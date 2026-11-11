"""Astracode format: geometry, symbol mapping and error-check code.

All lengths are in model units, with the ring of star centres at radius 100
around the origin; x to the right, y down.

Layout
  - 12 stars, upright (never rotated), centred on the ring at angles
    -90 + 30*i degrees: star 0 at 12 o'clock, then clockwise.
  - Each star is a regular pentagram. The top limb is always present; the other
    four limbs, clockwise from the upper-right one, carry bits 3,2,1,0 of the
    star's 4-bit symbol (1 = limb present).
  - The logo sits inside a circle of diameter LOGO_D at the centre.

Payload
  - A number 0 .. 10^12-1, as 10 symbols, most significant first.
  - 2 Reed-Solomon parity symbols over GF(16) follow (stars 10 and 11).
  - The 12 symbols are XORed with MASK before drawing.
"""
import math

R = 100.0
STAR_RO = 19.5
STAR_RATIO = math.sin(math.radians(18)) / math.sin(math.radians(54))  # regular pentagram, 0.38197
STAR_RI = STAR_RO * STAR_RATIO
LOGO_D = 180.0
MARGIN = 16.0
HALF = R + STAR_RO + MARGIN  # half side of the full code square
N_STARS = 12
N_DATA = 10
MAX_NUMBER = 10 ** 12

MASK = (0x5, 0xA, 0x3, 0xC, 0x9, 0x6, 0xE, 0x1, 0x7, 0x8, 0xB, 0x4)

# ---------------------------------------------------------------- GF(16) / RS

_EXP = [0] * 30
_LOG = [0] * 16
_x = 1
for _i in range(15):
    _EXP[_i] = _EXP[_i + 15] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 16:
        _x ^= 0x13  # x^4 + x + 1


def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def rs_parity(data):
    """Remainder of data(x) * x^2 modulo g(x) = (x + 1)(x + a) = x^2 + 3x + 2."""
    r1 = r0 = 0
    for d in data:
        f = d ^ r1
        r1 = r0 ^ gf_mul(f, 3)
        r0 = gf_mul(f, 2)
    return [r1, r0]


def rs_syndromes(cw):
    """(S0, S1) for a 12-symbol codeword; both zero when it is valid."""
    s0 = s1 = 0
    n = len(cw)
    for i, c in enumerate(cw):
        s0 ^= c
        s1 ^= gf_mul(c, _EXP[(n - 1 - i) % 15])
    return s0, s1


# ---------------------------------------------------------------- payload

def encode(number):
    """Number -> the 12 symbols as drawn (masked)."""
    if not 0 <= number < MAX_NUMBER:
        raise ValueError("number must have at most 12 digits")
    data = [(number >> (4 * (N_DATA - 1 - i))) & 15 for i in range(N_DATA)]
    cw = data + rs_parity(data)
    return [c ^ m for c, m in zip(cw, MASK)]


def decode_symbols(symbols):
    """The 12 symbols as drawn -> number, or None if the check fails."""
    cw = [s ^ m for s, m in zip(symbols, MASK)]
    if rs_syndromes(cw) != (0, 0):
        return None
    number = 0
    for d in cw[:N_DATA]:
        number = (number << 4) | d
    return number if number < MAX_NUMBER else None


# ---------------------------------------------------------------- geometry

def star_center(i):
    a = math.radians(-90 + 30 * i)
    return R * math.cos(a), R * math.sin(a)


def limb_present(symbol, k):
    """k = 0 is the top limb, 1..4 go clockwise."""
    return k == 0 or bool((symbol >> (4 - k)) & 1)


def star_polygon(symbol, cx=0.0, cy=0.0, ro=STAR_RO, ratio=STAR_RATIO):
    ri = ro * ratio
    pts = []
    for k in range(5):
        a = math.radians(-90 - 36 + 72 * k)
        pts.append((cx + ri * math.cos(a), cy + ri * math.sin(a)))
        if limb_present(symbol, k):
            a = math.radians(-90 + 72 * k)
            pts.append((cx + ro * math.cos(a), cy + ro * math.sin(a)))
    return pts


def polygon_centroid(pts):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    return cx / (3 * a), cy / (3 * a)


# Centroid of each of the 16 star shapes relative to the star centre.
SHAPE_CENTROID = [polygon_centroid(star_polygon(s)) for s in range(16)]

LOGO_SRC = 3840
LOGO_TRANSFORM = "translate(744.90 729.60) scale(2.671645) translate(-14.05 -91.42)"
LOGO_PATHS = (
    '<path d="M338.37,465.61c23-62.6,49.94-122.56,81.41-172.56,19.82,31.52,37.88,67,54.29,104.62a1242.84,1242.84,0,0,1,128.34-46.05C485.58,134.07,419.78,91.42,419.78,91.42s-78.92,51.16-217.27,327C169,485.21,134,563.55,99.17,654.68,152.15,594.63,228.47,529.46,338.37,465.61Z"/>'
    '<path d="M590.17,543.37c-16.66,11.13-34.06,23.38-51.87,36.74,49.12,170.74,70.06,328.47,70.06,328.47H825.5C776.38,741,723.47,603.64,672.6,492.63,646.63,507.34,618.78,524.25,590.17,543.37Z"/>'
    '<path d="M982.17,298.05l-98-5.3-38.63-90.22a4,4,0,0,0-7.39.13l-35.32,91.58-97.75,8.86a4,4,0,0,0-2.16,7.07l51.57,41.89a1245.36,1245.36,0,0,0-135.25,31.47,1205.4,1205.4,0,0,0-131.61,46.65A1107.27,1107.27,0,0,0,384.41,480q-34.05,18.71-64,37.47c-149,93.34-225.58,186.86-264.91,259-42.9,78.78-41.47,132.11-41.47,132.11H231.19c58.29-158.17,179.5-278.94,296.43-364.38a1326.91,1326.91,0,0,1,130-83.58c46.2-26.07,86.34-45.26,114.42-57.74l-14.77,64.88a4,4,0,0,0,6.06,4.24l82.4-53.33L930.06,469a4,4,0,0,0,5.9-4.46l-25.25-94.84L984.58,305A4,4,0,0,0,982.17,298.05Z"/>'
)


def svg(number, ink="#021639", background="#febc5c", size=512):
    """The code as an SVG string. background=None leaves it transparent."""
    symbols = encode(number)
    stars = "".join(
        '<polygon points="%s"/>' % " ".join(
            "%.3f,%.3f" % p for p in star_polygon(s, *star_center(i)))
        for i, s in enumerate(symbols))
    k = LOGO_D / LOGO_SRC
    rect = "" if background is None else (
        f'<rect x="{-HALF}" y="{-HALF}" width="{2 * HALF}" height="{2 * HALF}" fill="{background}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{-HALF} {-HALF} {2 * HALF} {2 * HALF}">{rect}<g fill="{ink}">{stars}'
        f'<g transform="scale({k}) translate({-LOGO_SRC / 2} {-LOGO_SRC / 2}) {LOGO_TRANSFORM}">'
        f'{LOGO_PATHS}</g></g></svg>')
