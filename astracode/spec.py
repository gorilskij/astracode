"""Astracode format: geometry, symbol mapping and error-correcting code.

All lengths are in model units, with the ring of star centres at radius 100
around the origin; x to the right, y down.

Layout
  - A code has 1 to MAX_RINGS rings of stars. Ring k has 12 + 6k stars at
    radius 100 + RING_STEP * k, star 0 at 12 o'clock, then clockwise. Every
    ring has a star on each of the six spokes at multiples of 60 degrees.
  - Every star is the same size and upright (never rotated): a complete
    regular pentagram. Its centre pentagon is always solid ink. Each of its
    five limbs is either solid (bit 1) or a lighter shade, SHADE of the way
    from background to ink (bit 0). The top limb is bit 4 of the star's 5-bit
    symbol, then clockwise bits 3, 2, 1, 0.
  - The centre may hold any picture (or nothing) inside a circle of radius
    CENTER_R. Readers do not look at it. The band between that circle and
    the stars must stay background colour. The Astra logo is drawn in a box
    of side LOGO_D and reaches radius 72.9.

What a code carries
  - A string of bits, usually whole bytes, of any length up to what 24 rings
    hold. The format gives the bits no meaning; that is up to whoever uses it
    (astra.py holds Astra's own use). A code has as many rings as its payload
    needs: up to 38 bits fit in one ring.
  - The payload is laid out as a stream: the payload's bits, then a 1 bit,
    then zeros to fill the code.

Ring 0
  - Its 12 stars hold a 40-bit value in 8 symbols, most significant first,
    then 4 Reed-Solomon check symbols over GF(32) (stars 8 to 11). The check
    symbols let a reader repair one misread star and still detect up to
    three, or repair two stars it already knows to be doubtful.
  - The value's top bit says whether there are more rings.
      0: this ring is the whole code. The other 39 bits are the stream.
      1: then 5 bits, the number of rings minus one (1 .. 23); 2 bits, the
         error-correction level of the outer rings; and the first 32 bits of
         the stream.
    So ring 0 always tells the reader how many rings to expect, and a ring 0
    cut out of a bigger code does not read as a code of its own.
  - Before it is split into symbols the value is scrambled by a fixed,
    public, reversible mixing step (MIX) so that codes for neighbouring
    values look unrelated and use both shades evenly. It is not encryption:
    anyone can undo it.
  - The 12 symbols are XORed with MASK before drawing, so no code is a single
    flat shade.

Rings 1 and up
  - Their stars, in order, are taken in pairs as 10-bit symbols (first star =
    high 5 bits) forming one Reed-Solomon codeword over GF(1024): the rest of
    the stream, then check symbols. Level 0..3 makes 1/8, 2/8, 3/8 or 4/8 of
    the symbols check symbols. The check symbols also cover ring 0's 40-bit
    value (as four leading symbols that are not drawn again), tying the rings
    together.
  - Outer symbols are XORed with a fixed pseudo-random sequence before drawing.

Earlier formats are in git history: v0.1.0 (limbs present or absent), v0.2.x
(one ring holding a number), v0.3.0 (a number in one ring or bytes in several).
"""
import math

R = 100.0
STAR_RO = 19.5
STAR_RATIO = math.sin(math.radians(18)) / math.sin(math.radians(54))  # regular pentagram, 0.38197
STAR_RI = STAR_RO * STAR_RATIO
LOGO_D = 180.0
CENTER_R = 73.0              # the centre picture must stay within this radius
MARGIN = 16.0
HALF = R + STAR_RO + MARGIN  # half side of the full code square
SHADE = 0.7                  # ink fraction of a light limb
N_STARS = 12
N_DATA = 8
N_CHECK = N_STARS - N_DATA

MASK = (0x15, 0x0A, 0x13, 0x1C, 0x09, 0x06, 0x1E, 0x01, 0x17, 0x08, 0x1B, 0x04)

# ---------------------------------------------------------------- GF(32) / RS

_EXP = [0] * 62
_LOG = [0] * 32
_x = 1
for _i in range(31):
    _EXP[_i] = _EXP[_i + 31] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 32:
        _x ^= 0x25  # x^5 + x^2 + 1


def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def gf_div(a, b):
    return 0 if a == 0 else _EXP[_LOG[a] - _LOG[b] + 31]


def _generator():
    g = [1]
    for j in range(N_CHECK):          # roots a^0 .. a^3
        g = [x ^ gf_mul(y, _EXP[j]) for x, y in zip(g + [0], [0] + g)]
    return g                           # highest degree first, g[0] == 1


_GEN = _generator()


def rs_parity(data):
    """Remainder of data(x) * x^N_CHECK modulo the generator polynomial."""
    rem = [0] * N_CHECK
    for d in data:
        f = d ^ rem[0]
        rem = [r ^ gf_mul(f, g) for r, g in zip(rem[1:] + [0], _GEN[1:])]
    return rem


def rs_syndromes(cw):
    """S0..S3 for a 12-symbol codeword; all zero when it is valid."""
    n = len(cw)
    out = []
    for j in range(N_CHECK):
        s = 0
        for i, c in enumerate(cw):
            s ^= gf_mul(c, _EXP[(j * (n - 1 - i)) % 31])
        out.append(s)
    return out


def rs_correct(cw):
    """Return (codeword, stars repaired) with at most one symbol repaired, or None.

    Four check symbols could repair two stars, but that would leave nothing to
    catch a third; repairing one keeps the code able to detect up to three.
    """
    s = rs_syndromes(cw)
    if not any(s):
        return list(cw), 0
    if 0 in s:
        return None
    x = gf_div(s[1], s[0])                       # a^position of a single error
    if gf_mul(s[1], x) != s[2] or gf_mul(s[2], x) != s[3]:
        return None
    pos = _LOG[x]
    if pos >= len(cw):
        return None
    fixed = list(cw)
    fixed[len(cw) - 1 - pos] ^= s[0]
    return fixed, 1


def rs_fill(cw, i1, i2):
    """Treat symbols i1 and i2 as unknown and solve for them; None unless the result checks out.

    Two check symbols rebuild the two unknowns and the other two verify, so this
    is only as trustworthy as the caller's reason for doubting exactly these two.
    """
    n = len(cw)
    s = rs_syndromes(cw)
    x1, x2 = _EXP[n - 1 - i1], _EXP[n - 1 - i2]
    e2 = gf_div(s[1] ^ gf_mul(s[0], x1), x1 ^ x2)
    e1 = s[0] ^ e2
    x1s, x2s = gf_mul(x1, x1), gf_mul(x2, x2)
    if gf_mul(e1, x1s) ^ gf_mul(e2, x2s) != s[2]:
        return None
    if gf_mul(e1, gf_mul(x1s, x1)) ^ gf_mul(e2, gf_mul(x2s, x2)) != s[3]:
        return None
    fixed = list(cw)
    fixed[i1] ^= e1
    fixed[i2] ^= e2
    return fixed


# ---------------------------------------------------------------- mixing

_MIX_KEYS = (0x3C6EF, 0x1B873, 0x5A827, 0x6ED9E)


def _mix_round(x, key):
    """20 bits -> 20 bits; 32-bit integer arithmetic so every implementation agrees."""
    t = ((x + key) * 0x9E3779B1) & 0xFFFFFFFF
    t ^= t >> 15
    t = (t * 0x85EBCA77) & 0xFFFFFFFF
    t ^= t >> 13
    return t & 0xFFFFF


def mix(value):
    """Scramble a 40-bit value: a 4-round Feistel network on its two 20-bit halves."""
    left, right = value >> 20, value & 0xFFFFF
    for key in _MIX_KEYS:
        left, right = right, left ^ _mix_round(right, key)
    return left << 20 | right


def unmix(value):
    left, right = value >> 20, value & 0xFFFFF
    for key in reversed(_MIX_KEYS):
        left, right = right ^ _mix_round(left, key), left
    return left << 20 | right


# ---------------------------------------------------------------- ring 0

MAX_RINGS = 24            # 1932 outer stars = 966 symbols, within GF(1024)'s 1023
LEVELS = 4                # error-correction levels 0..3
ONE_RING_BITS = 39        # stream bits in ring 0 when it is the whole code
HEAD_BITS = 32            # stream bits in ring 0 when more rings follow


def ring0_value(rings, level, head):
    """Ring 0's 40-bit value: the number of rings, the level, and the first bits of the stream."""
    if rings == 1:
        return head
    return 1 << 39 | (rings - 1) << 34 | level << 32 | head


def encode_ring0(value):
    """A 40-bit value -> the 12 symbols of ring 0 as drawn (masked)."""
    mixed = mix(value)
    data = [(mixed >> (5 * (N_DATA - 1 - i))) & 31 for i in range(N_DATA)]
    cw = data + rs_parity(data)
    return [c ^ m for c, m in zip(cw, MASK)]


def decode_ring0(symbols, doubtful=None):
    """The 12 symbols of ring 0 as drawn -> (40-bit value, stars repaired), or None.

    Repairs one misread star anywhere. If that fails and `doubtful` names two
    stars the reader is unsure of, repairs exactly those two instead.
    """
    cw = [s ^ m for s, m in zip(symbols, MASK)]
    res = rs_correct(cw)
    if res is None and doubtful is not None:
        filled = rs_fill(cw, *doubtful)
        res = None if filled is None else (filled, 2)
    if res is None:
        return None
    cw, repaired = res
    mixed = 0
    for d in cw[:N_DATA]:
        mixed = (mixed << 5) | d
    return unmix(mixed), repaired


def read_ring0(symbols, doubtful=None):
    """Ring 0 as drawn -> (rings, level, 40-bit value, stars repaired), or None.

    The level is 0 for a one-ring code, which has no outer check symbols.
    """
    res = decode_ring0(symbols, doubtful)
    if res is None:
        return None
    value, repaired = res
    if not value >> 39:
        return (1, 0, value, repaired) if value else None      # an all-zero stream has no end
    rings, level = (value >> 34 & 31) + 1, value >> 32 & 3
    return (rings, level, value, repaired) if rings <= MAX_RINGS else None


# ---------------------------------------------------------------- more rings

RING_STEP = 40.0          # radius added per ring

_EXP10 = [0] * 2046
_LOG10 = [0] * 1024
_x = 1
for _i in range(1023):
    _EXP10[_i] = _EXP10[_i + 1023] = _x
    _LOG10[_x] = _i
    _x <<= 1
    if _x & 1024:
        _x ^= 0x409  # x^10 + x^3 + 1


def gf10_mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP10[_LOG10[a] + _LOG10[b]]


def gf10_div(a, b):
    return 0 if a == 0 else _EXP10[_LOG10[a] - _LOG10[b] + 1023]


def ring_stars(k):
    return 12 + 6 * k


def ring_radius(k):
    return R + RING_STEP * k


def half_side(rings):
    """Half side of the full code square for a code with this many rings."""
    return ring_radius(rings - 1) + STAR_RO + MARGIN


def outer_symbols(rings):
    """Number of 10-bit symbols in rings 1 and up."""
    return sum(ring_stars(k) for k in range(1, rings)) // 2


def check_symbols(rings, level):
    """How many of the outer symbols are check symbols. Never fewer than two; none in a one-ring code."""
    return max(2, -(-outer_symbols(rings) * (level + 1) // 8)) if rings > 1 else 0


def stream_bits(rings, level=0):
    """Length of the stream: ring 0's share plus the outer rings' data symbols."""
    if rings == 1:
        return ONE_RING_BITS
    return HEAD_BITS + 10 * (outer_symbols(rings) - check_symbols(rings, level))


def capacity_bits(rings, level=0):
    """Longest payload, in bits, a code with this many rings holds."""
    return stream_bits(rings, level) - 1


def capacity(rings, level=0):
    """Longest payload, in whole bytes, a code with this many rings holds."""
    return capacity_bits(rings, level) // 8


def rings_for(nbits, level=1):
    """The fewest rings that hold a payload of this many bits."""
    for rings in range(1, MAX_RINGS + 1):
        if capacity_bits(rings, level) >= nbits:
            return rings
    raise ValueError("payload too long: %d rings hold at most %d bits at level %d"
                     % (MAX_RINGS, capacity_bits(MAX_RINGS, level), level))


def _whiten(j):
    return _mix_round(j, 0x2545F) & 1023


_gen10 = {}


def _generator10(p):
    if p not in _gen10:
        g = [1]
        for j in range(p):            # roots a^0 .. a^(p-1)
            g = [x ^ gf10_mul(y, _EXP10[j]) for x, y in zip(g + [0], [0] + g)]
        _gen10[p] = g
    return _gen10[p]


def rs10_parity(data, p):
    gen = _generator10(p)
    rem = [0] * p
    for d in data:
        f = d ^ rem[0]
        rem = rem[1:] + [0]
        if f:
            lf = _LOG10[f]
            rem = [r ^ (_EXP10[lf + _LOG10[g]] if g else 0) for r, g in zip(rem, gen[1:])]
    return rem


def rs10_decode(cw, p, max_errors=None):
    """Repair up to p // 2 wrong symbols (or max_errors). Returns (codeword, count) or None."""
    n = len(cw)
    synd = []
    for j in range(p):
        s = 0
        for i, c in enumerate(cw):
            if c:
                s ^= _EXP10[(_LOG10[c] + j * (n - 1 - i)) % 1023]
        synd.append(s)
    if not any(synd):
        return list(cw), 0
    # Berlekamp-Massey: error locator, lowest degree first
    loc, prev, shift, last, errs = [1], [1], 1, 1, 0
    for i in range(p):
        delta = synd[i]
        for j in range(1, errs + 1):
            if j < len(loc):
                delta ^= gf10_mul(loc[j], synd[i - j])
        if delta == 0:
            shift += 1
            continue
        coef = gf10_div(delta, last)
        nxt = loc + [0] * max(0, len(prev) + shift - len(loc))
        for j, c in enumerate(prev):
            nxt[j + shift] ^= gf10_mul(coef, c)
        if 2 * errs <= i:
            prev, last, errs, shift = loc, delta, i + 1 - errs, 1
        else:
            shift += 1
        loc = nxt
    while loc and loc[-1] == 0:
        loc.pop()
    if len(loc) - 1 != errs or errs > (p // 2 if max_errors is None else min(max_errors, p // 2)):
        return None
    # error evaluator: synd(x) * loc(x) mod x^p
    omega = [0] * p
    for i, a in enumerate(loc):
        for j in range(p - i):
            omega[i + j] ^= gf10_mul(a, synd[j])
    fixed, found = list(cw), 0
    for pos in range(n):
        xinv = _EXP10[(1023 - pos) % 1023]
        val, power = 0, 1
        for c in loc:
            val ^= gf10_mul(c, power)
            power = gf10_mul(power, xinv)
        if val:
            continue
        num, power = 0, 1
        for c in omega:
            num ^= gf10_mul(c, power)
            power = gf10_mul(power, xinv)
        den, power = 0, 1
        x2 = gf10_mul(xinv, xinv)
        for c in loc[1::2]:               # formal derivative keeps the odd terms
            den ^= gf10_mul(c, power)
            power = gf10_mul(power, x2)
        if den == 0:
            return None
        fixed[n - 1 - pos] ^= gf10_mul(_EXP10[pos], gf10_div(num, den))
        found += 1
    return (fixed, errs) if found == errs else None


def _virtual(value):
    return [(value >> s) & 1023 for s in (30, 20, 10, 0)]


def encode_bits(value, nbits, level=1, rings=None):
    """A payload of nbits bits, given as an integer -> the symbols of every ring as drawn.

    Returns [ring 0's 12 symbols, ring 1's 18, ...]. The code has the fewest
    rings that hold the payload, or `rings` if given, which must be enough.
    `level` is the error-correction level of the outer rings, if there are any.
    """
    if not 0 <= level < LEVELS:
        raise ValueError("level must be 0..%d" % (LEVELS - 1))
    if nbits < 0 or value < 0 or value >> nbits:
        raise ValueError("the payload does not fit in %d bits" % nbits)
    if rings is None:
        rings = rings_for(nbits, level)
    elif not 1 <= rings <= MAX_RINGS:
        raise ValueError("rings must be 1..%d" % MAX_RINGS)
    elif nbits > capacity_bits(rings, level):
        raise ValueError("payload too long: at most %d bits" % capacity_bits(rings, level))
    total = stream_bits(rings, level)
    stream = (value << 1 | 1) << (total - nbits - 1)
    if rings == 1:
        return [encode_ring0(ring0_value(1, 0, stream))]
    p = check_symbols(rings, level)
    d = outer_symbols(rings) - p
    data = [(stream >> (10 * (d - 1 - j))) & 1023 for j in range(d)]
    value0 = ring0_value(rings, level, stream >> (10 * d))
    outer = [c ^ _whiten(j) for j, c in enumerate(data + rs10_parity(_virtual(value0) + data, p))]
    stars = [half for sym in outer for half in (sym >> 5, sym & 31)]
    out, at = [encode_ring0(value0)], 0
    for k in range(1, rings):
        out.append(stars[at:at + ring_stars(k)])
        at += ring_stars(k)
    return out


def encode(data, level=1, rings=None):
    """Byte string -> the symbols of every ring as drawn; see encode_bits."""
    return encode_bits(int.from_bytes(data, "big"), 8 * len(data), level, rings)


def decode_bits(ring_symbols):
    """Symbols of every ring as drawn -> (payload as an integer, its length in bits, outer symbols repaired), or None.

    None if ring 0 does not check out, if the number or size of the rings is
    not what ring 0 says, or if the outer rings cannot be repaired.
    """
    header = read_ring0(ring_symbols[0])
    if header is None:
        return None
    rings, level, value, _ = header
    if len(ring_symbols) != rings or any(len(ring_symbols[k]) != ring_stars(k) for k in range(rings)):
        return None
    if rings == 1:
        stream, repaired = value, 0
    else:
        stars = [s for ring in ring_symbols[1:] for s in ring]
        outer = [(stars[2 * j] << 5 | stars[2 * j + 1]) ^ _whiten(j) for j in range(len(stars) // 2)]
        p = check_symbols(rings, level)
        # Two check symbols are always kept back for detection, so that a badly misread
        # code is refused rather than "repaired" into a different payload.
        res = rs10_decode(_virtual(value) + outer, p, max_errors=(p - 2) // 2)
        if res is None or res[0][:4] != _virtual(value):
            return None
        cw, repaired = res
        stream = value & ((1 << HEAD_BITS) - 1)
        for sym in cw[4:4 + len(outer) - p]:
            stream = stream << 10 | sym
    if stream == 0:
        return None
    pad = (stream & -stream).bit_length()         # the final 1 bit and the zeros after it
    return stream >> pad, stream_bits(rings, level) - pad, repaired


def as_bytes(value, nbits):
    """A payload as a byte string, or None if it is not a whole number of bytes."""
    return None if nbits % 8 else value.to_bytes(nbits // 8, "big")


def decode(ring_symbols):
    """Symbols of every ring as drawn -> (payload bytes, outer symbols repaired); None if it is not whole bytes."""
    res = decode_bits(ring_symbols)
    if res is None or res[1] % 8:
        return None
    return as_bytes(res[0], res[1]), res[2]


# ---------------------------------------------------------------- geometry

def star_center(i, ring=0):
    a = math.radians(-90 + 360.0 * i / ring_stars(ring))
    r = ring_radius(ring)
    return r * math.cos(a), r * math.sin(a)


def limb_solid(symbol, k):
    """k = 0 is the top limb, 1..4 go clockwise."""
    return bool((symbol >> (4 - k)) & 1)


def limb_polygon(k, cx=0.0, cy=0.0):
    """Limb k as a quadrilateral closed through the star centre, so it tucks under the core."""
    pts = []
    for a, r in ((-90 - 36 + 72 * k, STAR_RI), (-90 + 72 * k, STAR_RO), (-90 + 36 + 72 * k, STAR_RI)):
        pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    pts.append((cx, cy))
    return pts


def star_polygon(symbol, cx=0.0, cy=0.0, ro=STAR_RO, ratio=STAR_RATIO):
    """Outline of the solid part of a star: the centre pentagon plus its solid limbs."""
    ri = ro * ratio
    pts = []
    for k in range(5):
        a = math.radians(-90 - 36 + 72 * k)
        pts.append((cx + ri * math.cos(a), cy + ri * math.sin(a)))
        if limb_solid(symbol, k):
            a = math.radians(-90 + 72 * k)
            pts.append((cx + ro * math.cos(a), cy + ro * math.sin(a)))
    return pts


LOGO_SRC = 3840
LOGO_TRANSFORM = "translate(744.90 729.60) scale(2.671645) translate(-14.05 -91.42)"
LOGO_PATHS = (
    '<path d="M338.37,465.61c23-62.6,49.94-122.56,81.41-172.56,19.82,31.52,37.88,67,54.29,104.62a1242.84,1242.84,0,0,1,128.34-46.05C485.58,134.07,419.78,91.42,419.78,91.42s-78.92,51.16-217.27,327C169,485.21,134,563.55,99.17,654.68,152.15,594.63,228.47,529.46,338.37,465.61Z"/>'
    '<path d="M590.17,543.37c-16.66,11.13-34.06,23.38-51.87,36.74,49.12,170.74,70.06,328.47,70.06,328.47H825.5C776.38,741,723.47,603.64,672.6,492.63,646.63,507.34,618.78,524.25,590.17,543.37Z"/>'
    '<path d="M982.17,298.05l-98-5.3-38.63-90.22a4,4,0,0,0-7.39.13l-35.32,91.58-97.75,8.86a4,4,0,0,0-2.16,7.07l51.57,41.89a1245.36,1245.36,0,0,0-135.25,31.47,1205.4,1205.4,0,0,0-131.61,46.65A1107.27,1107.27,0,0,0,384.41,480q-34.05,18.71-64,37.47c-149,93.34-225.58,186.86-264.91,259-42.9,78.78-41.47,132.11-41.47,132.11H231.19c58.29-158.17,179.5-278.94,296.43-364.38a1326.91,1326.91,0,0,1,130-83.58c46.2-26.07,86.34-45.26,114.42-57.74l-14.77,64.88a4,4,0,0,0,6.06,4.24l82.4-53.33L930.06,469a4,4,0,0,0,5.9-4.46l-25.25-94.84L984.58,305A4,4,0,0,0,982.17,298.05Z"/>'
)


def blend(background, ink, t):
    """Hex colour t of the way from background to ink."""
    a = [int(background[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(ink[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def svg_rings(ring_symbols, ink="#021639", background="#febc5c", size=512):
    """A code as an SVG string, from the symbols of each ring (see encode)."""
    half = half_side(len(ring_symbols))
    poly = lambda pts: '<polygon points="%s"/>' % " ".join("%.3f,%.3f" % p for p in pts)
    light = solid = ""
    for ring, symbols in enumerate(ring_symbols):
        for i, sym in enumerate(symbols):
            cx, cy = star_center(i, ring)
            light += "".join(poly(limb_polygon(k, cx, cy)) for k in range(5) if not limb_solid(sym, k))
            solid += poly(star_polygon(sym, cx, cy))
    k = LOGO_D / LOGO_SRC
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{-half} {-half} {2 * half} {2 * half}">'
        f'<rect x="{-half}" y="{-half}" width="{2 * half}" height="{2 * half}" fill="{background}"/>'
        f'<g fill="{blend(background, ink, SHADE)}">{light}</g><g fill="{ink}">{solid}'
        f'<g transform="scale({k}) translate({-LOGO_SRC / 2} {-LOGO_SRC / 2}) {LOGO_TRANSFORM}">'
        f'{LOGO_PATHS}</g></g></svg>')
