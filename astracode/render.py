"""Raster rendering of a code, used by the benchmark to make its frames."""
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


def center_alpha(picture="logo", scale=1.0, rotation=0.0, size=512):
    """Ink coverage of the centre picture as a square image spanning the 2 * CENTER_R box.

    picture: "logo", None, or a square 0..1 float image to fit inside the circle.
    """
    out = np.zeros((size, size), np.float32)
    if picture is None:
        return out
    if isinstance(picture, str):
        src, span = logo_alpha(), spec.LOGO_D
    else:
        src, span = np.asarray(picture, np.float32), 2 * spec.CENTER_R
    k = scale * span / (2 * spec.CENTER_R) * size / src.shape[0]     # source px -> output px
    M = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), rotation, k)
    M[:, 2] += size / 2 - src.shape[1] / 2, size / 2 - src.shape[0] / 2
    out = cv2.warpAffine(src, M, (size, size), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:size, 0:size]
    out[np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) > size / 2] = 0
    return out


def alpha(code, px_per_unit=3.0, supersample=3, center="logo", center_scale=1.0, center_rotation=0.0):
    """Ink coverage (0..1 float32) of the full code square, margin included.

    code: the symbols of every ring, as spec.encode returns them.
    """
    rings = code
    half = spec.half_side(len(rings))
    s = px_per_unit * supersample
    side = int(round(2 * half * s))
    canvas = np.zeros((side, side), np.uint8)
    shift = 6
    to_px = lambda pts: np.round((np.array(pts) + half) * s * (1 << shift)).astype(np.int32)
    stars = [(sym, spec.star_center(i, k)) for k, ring in enumerate(rings) for i, sym in enumerate(ring)]
    for sym, c in stars:                       # light limbs first, solid parts over them
        for k in range(5):
            if not spec.limb_solid(sym, k):
                cv2.fillPoly(canvas, [to_px(spec.limb_polygon(k, *c))],
                             int(round(255 * spec.SHADE)), lineType=cv2.LINE_AA, shift=shift)
    for sym, c in stars:
        cv2.fillPoly(canvas, [to_px(spec.star_polygon(sym, *c))], 255, lineType=cv2.LINE_AA, shift=shift)
    a = canvas.astype(np.float32) / 255.0
    lside = int(round(2 * spec.CENTER_R * s))
    pic = center_alpha(center, center_scale, center_rotation, lside)
    o = (side - lside) // 2
    a[o:o + lside, o:o + lside] = np.maximum(a[o:o + lside, o:o + lside], pic)
    out = int(round(2 * half * px_per_unit))
    return cv2.resize(a, (out, out), interpolation=cv2.INTER_AREA)


def render(code, px_per_unit=3.0, ink=(57, 22, 2), background=(92, 188, 254), **center):
    """BGR uint8 image of the code. Colours are BGR tuples; see alpha for the rest."""
    a = alpha(code, px_per_unit, **center)[..., None]
    img = np.array(background, np.float32) * (1 - a) + np.array(ink, np.float32) * a
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)
