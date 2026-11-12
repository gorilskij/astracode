"""Raster rendering of a code, used by the tests and to build the scanner's logo template."""
import os

import cv2
import numpy as np

from . import spec

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_logo = None


def logo_alpha():
    """Logo ink coverage, 0..1, as a square image spanning the LOGO_D crop circle's box."""
    global _logo
    if _logo is None:
        g = cv2.imread(os.path.join(_ASSETS, "logo_mask.png"), cv2.IMREAD_GRAYSCALE)
        _logo = 1.0 - g.astype(np.float32) / 255.0
    return _logo


def alpha(number, px_per_unit=3.0, supersample=3):
    """Ink coverage (0..1 float32) of the full code square, margin included."""
    s = px_per_unit * supersample
    side = int(round(2 * spec.HALF * s))
    canvas = np.zeros((side, side), np.uint8)
    shift = 6
    to_px = lambda pts: np.round((np.array(pts) + spec.HALF) * s * (1 << shift)).astype(np.int32)
    symbols = spec.encode(number)
    for i, sym in enumerate(symbols):          # light limbs first, solid parts over them
        for k in range(5):
            if not spec.limb_solid(sym, k):
                cv2.fillPoly(canvas, [to_px(spec.limb_polygon(k, *spec.star_center(i)))],
                             int(round(255 * spec.SHADE)), lineType=cv2.LINE_AA, shift=shift)
    for i, sym in enumerate(symbols):
        cv2.fillPoly(canvas, [to_px(spec.star_polygon(sym, *spec.star_center(i)))], 255, lineType=cv2.LINE_AA, shift=shift)
    a = canvas.astype(np.float32) / 255.0
    lside = int(round(spec.LOGO_D * s))
    logo = cv2.resize(logo_alpha(), (lside, lside), interpolation=cv2.INTER_AREA)
    o = (side - lside) // 2
    a[o:o + lside, o:o + lside] = np.maximum(a[o:o + lside, o:o + lside], logo)
    out = int(round(2 * spec.HALF * px_per_unit))
    return cv2.resize(a, (out, out), interpolation=cv2.INTER_AREA)


def render(number, px_per_unit=3.0, ink=(57, 22, 2), background=(92, 188, 254)):
    """BGR uint8 image of the code. Colours are BGR tuples."""
    a = alpha(number, px_per_unit)[..., None]
    img = np.array(background, np.float32) * (1 - a) + np.array(ink, np.float32) * a
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)
