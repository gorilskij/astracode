"""Astracode format, version 2: geometry, symbol mapping and error-correcting code.

All lengths are in model units, with the ring of star centres at radius 100
around the origin; x to the right, y down.

Layout
  - 12 stars, upright (never rotated), centred on the ring at angles
    -90 + 30*i degrees: star 0 at 12 o'clock, then clockwise.
  - Each star is a complete regular pentagram. Its centre pentagon is always
    solid ink. Each of its five limbs is either solid (bit 1) or a lighter
    shade, SHADE of the way from background to ink (bit 0). The top limb is
    bit 4 of the star's 5-bit symbol, then clockwise bits 3, 2, 1, 0.
  - The logo sits inside a circle of diameter LOGO_D at the centre.

Payload
  - A number 0 .. 10^12-1, as 8 symbols (40 bits), most significant first.
  - 4 Reed-Solomon check symbols over GF(32) follow (stars 8 to 11). They let
    a reader repair one misread star and still detect up to three, or repair
    two stars it already knows to be doubtful.
  - The 12 symbols are XORed with MASK before drawing, so no code is a single
    flat shade.

Version 1 (limbs present or absent, top limb fixed, 2 check stars) is in git
history, tagged v1.0.
"""
import math

R = 100.0
STAR_RO = 19.5
STAR_RATIO = math.sin(math.radians(18)) / math.sin(math.radians(54))  # regular pentagram, 0.38197
STAR_RI = STAR_RO * STAR_RATIO
LOGO_D = 180.0
MARGIN = 16.0
HALF = R + STAR_RO + MARGIN  # half side of the full code square
SHADE = 0.7                  # ink fraction of a light limb
N_STARS = 12
N_DATA = 8
N_CHECK = N_STARS - N_DATA
MAX_NUMBER = 10 ** 12

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


# ---------------------------------------------------------------- payload

def encode(number):
    """Number -> the 12 symbols as drawn (masked)."""
    if not 0 <= number < MAX_NUMBER:
        raise ValueError("number must have at most 12 digits")
    data = [(number >> (5 * (N_DATA - 1 - i))) & 31 for i in range(N_DATA)]
    cw = data + rs_parity(data)
    return [c ^ m for c, m in zip(cw, MASK)]


def decode_symbols(symbols, doubtful=None):
    """The 12 symbols as drawn -> (number, stars repaired), or None if the check fails.

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
    number = 0
    for d in cw[:N_DATA]:
        number = (number << 5) | d
    return (number, repaired) if number < MAX_NUMBER else None


# ---------------------------------------------------------------- geometry

def star_center(i):
    a = math.radians(-90 + 30 * i)
    return R * math.cos(a), R * math.sin(a)


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


def mix(background, ink, t):
    """Hex colour t of the way from background to ink."""
    a = [int(background[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(ink[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def svg(number, ink="#021639", background="#febc5c", size=512):
    """The code as an SVG string."""
    poly = lambda pts: '<polygon points="%s"/>' % " ".join("%.3f,%.3f" % p for p in pts)
    light = solid = ""
    for i, sym in enumerate(encode(number)):
        cx, cy = star_center(i)
        light += "".join(poly(limb_polygon(k, cx, cy)) for k in range(5) if not limb_solid(sym, k))
        solid += poly(star_polygon(sym, cx, cy))
    k = LOGO_D / LOGO_SRC
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{-HALF} {-HALF} {2 * HALF} {2 * HALF}">'
        f'<rect x="{-HALF}" y="{-HALF}" width="{2 * HALF}" height="{2 * HALF}" fill="{background}"/>'
        f'<g fill="{mix(background, ink, SHADE)}">{light}</g><g fill="{ink}">{solid}'
        f'<g transform="scale({k}) translate({-LOGO_SRC / 2} {-LOGO_SRC / 2}) {LOGO_TRANSFORM}">'
        f'{LOGO_PATHS}</g></g></svg>')
