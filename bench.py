"""Synthetic reliability test for the scanner.

Renders random codes, places them in a camera-like frame with perspective, blur,
noise, uneven light and JPEG damage, scans them, and reports hit / miss / wrong.

    .venv/bin/python bench.py [count] [level] [--rings N | --rings LOW-HIGH] [--wavy STRAIN] [--save-fails DIR]

--wavy 0.1 prints the code on a surface that is not flat: smooth waves that stretch
and squeeze it locally by up to 10%.

With --rings above 1 the frame is larger (1280 x 960, or 2560 x 1920 above 12 rings),
since those codes need more pixels.
"""
import glob
import math
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from astracode import decode, render, spec

# ASTRA_SHADE=0.5 renders the light limbs at another level (the scanner works it out for itself).
spec.SHADE = float(os.environ.get("ASTRA_SHADE", spec.SHADE))

FRAME_W, FRAME_H = 960, 720
SRC_PPU = 3.0
ROOT = os.path.dirname(os.path.abspath(__file__))
_bgs = None


def backgrounds():
    global _bgs
    if _bgs is None:
        _bgs = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(ROOT, "assets/bg/*.jpg")))]
    return _bgs


def luma(c):
    return 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]


def random_colours(rng):
    kind = rng.random()
    if kind < 0.3:
        ink, bg = (57, 22, 2), (92, 188, 254)
    elif kind < 0.45:
        ink, bg = (92, 188, 254), (57, 22, 2)
    else:
        while True:
            ink, bg = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            if abs(luma(ink) - luma(bg)) > 50:
                break
    return tuple(int(v) for v in ink), tuple(int(v) for v in bg)


def scene_background(rng):
    kind = rng.random()
    if kind < 0.6 and backgrounds():
        src = backgrounds()[rng.integers(len(backgrounds()))]
        s = rng.uniform(0.6, 1.0) * min(1.0, (src.shape[1] - 1) / FRAME_W, (src.shape[0] - 1) / FRAME_H)
        w, h = int(FRAME_W * s), int(FRAME_H * s)
        x, y = rng.integers(0, src.shape[1] - w), rng.integers(0, src.shape[0] - h)
        return cv2.resize(src[y:y + h, x:x + w], (FRAME_W, FRAME_H))
    if kind < 0.8:
        return np.full((FRAME_H, FRAME_W, 3), rng.integers(0, 256, 3), np.uint8)
    # Busy synthetic clutter: random blobs and "text-like" marks.
    img = np.full((FRAME_H, FRAME_W, 3), rng.integers(120, 256, 3), np.uint8)
    for _ in range(400):
        p = (int(rng.integers(0, FRAME_W)), int(rng.integers(0, FRAME_H)))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(3, 25)), col, -1, cv2.LINE_AA)
        else:
            cv2.putText(img, chr(int(rng.integers(33, 123))), p, cv2.FONT_HERSHEY_SIMPLEX,
                        rng.uniform(0.4, 2.5), col, int(rng.integers(1, 5)), cv2.LINE_AA)
    return img


def placement(rng, diameter, tilt_deg, half=spec.HALF):
    """Homography from code model units to frame pixels, for a tilted plane seen by a pinhole camera."""
    f = 1.1 * FRAME_W
    roll, axis, tilt = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), math.radians(tilt_deg)
    dist = f * 2 * half / diameter
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    ca, sa = math.cos(axis), math.sin(axis)
    A = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ct, st = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    Rm = A @ Rx @ A.T @ Rz
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    H = K @ np.column_stack([Rm[:, 0], Rm[:, 1], [0, 0, dist]])
    H = H / H[2, 2]
    corners = decode._apply(H, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * half)
    lo, hi = corners.min(0), corners.max(0)
    span = hi - lo
    if span[0] > FRAME_W - 8 or span[1] > FRAME_H - 8:
        return None
    tx = rng.uniform(4 - lo[0], FRAME_W - 4 - hi[0])
    ty = rng.uniform(4 - lo[1], FRAME_H - 4 - hi[1])
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]) @ H


def random_center(rng):
    """What sits in the middle of the code: (label, keyword arguments for render)."""
    kind = rng.random()
    if os.environ.get("ASTRA_CENTER") == "star":     # a single upright star, from code-star size to filling the circle
        pic = np.zeros((256, 256), np.uint8)
        r = rng.uniform(0.27, 0.95) * 128
        pts = np.array(spec.star_polygon(31)) * r / spec.STAR_RO + 128
        cv2.fillPoly(pic, [np.round(pts * 64).astype(np.int32)], 255, cv2.LINE_AA, shift=6)
        return "star %s" % ("small" if r < 0.4 * 128 else "big"), dict(center=pic.astype(np.float32) / 255)
    if kind < 0.35:
        return "logo", {}
    if kind < 0.5:
        return "none", dict(center=None)
    if kind < 0.65:
        return "logo moved", dict(center_scale=float(rng.uniform(0.35, 1.0)), center_rotation=float(rng.uniform(0, 360)))
    pic = np.zeros((256, 256), np.uint8)
    if kind < 0.8:                                   # letters
        for _ in range(int(rng.integers(1, 4))):
            cv2.putText(pic, chr(int(rng.integers(48, 91))), (int(rng.integers(10, 130)), int(rng.integers(110, 230))),
                        cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(2, 6), 255, int(rng.integers(4, 18)), cv2.LINE_AA)
        label = "letters"
    elif kind < 0.9 or not backgrounds():            # star-sized blobs, right up to the edge of the allowed circle
        for _ in range(int(rng.integers(3, 14))):
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 100)
            cv2.circle(pic, (int(128 + r * math.cos(a)), int(128 + r * math.sin(a))), int(rng.integers(10, 34)), 255, -1, cv2.LINE_AA)
        label = "blobs"
    else:                                            # a photograph
        src = backgrounds()[rng.integers(len(backgrounds()))]
        x, y = rng.integers(0, src.shape[1] - 400), rng.integers(0, src.shape[0] - 400)
        pic = 255 - cv2.resize(cv2.cvtColor(src[y:y + 400, x:x + 400], cv2.COLOR_BGR2GRAY), (256, 256))
        label = "photo"
    return label, dict(center=pic.astype(np.float32) / 255, center_rotation=float(rng.uniform(0, 360)))


def wave(code, seed, strain, px_per_unit):
    """Bend the code image with two smooth waves, as on a sheet that does not lie flat."""
    rng = np.random.default_rng(seed + 7 * 10 ** 8)
    h, w = code.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = xx.copy(), yy.copy()
    for _ in range(2):
        lam = rng.uniform(150, 700) * px_per_unit            # wavelength
        amp = rng.uniform(0, strain) * lam / (2 * math.pi)   # so the local stretch is at most `strain`
        th, psi, ph = rng.uniform(0, 2 * math.pi, 3)
        d = amp * np.sin(2 * math.pi * (xx * math.cos(th) + yy * math.sin(th)) / lam + ph)
        mx += (d * math.cos(psi)).astype(np.float32)
        my += (d * math.sin(psi)).astype(np.float32)
    return cv2.remap(code, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_case(seed, level=1.0, rings=(1, 1), wavy=0.0):
    """level scales how harsh the damage is; 1.0 is the default mix. rings: (fewest, most)."""
    global FRAME_W, FRAME_H
    FRAME_W, FRAME_H = (960, 720) if rings[1] == 1 else (1280, 960) if rings[1] <= 12 else (2560, 1920)
    rng = np.random.default_rng(seed)
    extra = np.random.default_rng(seed + 5 * 10 ** 8)
    center_label, center = random_center(extra)
    mirrored = extra.random() < 0.15
    n_rings = int(extra.integers(rings[0], rings[1] + 1))
    ec_level = int(extra.integers(0, spec.LEVELS))
    half = spec.half_side(n_rings)
    number = int(rng.integers(0, spec.MAX_NUMBER))
    ink, bgc = random_colours(rng)
    while True:
        # The same range of star sizes whatever the number of rings, as far as the frame allows.
        diameter = float(np.exp(rng.uniform(math.log(90), math.log(560)))) * half / spec.HALF
        tilt = float(rng.uniform(0, 55))
        H = placement(rng, diameter, tilt, half) if diameter < FRAME_H else None
        if H is not None:
            break
    px_per_unit = diameter / (2 * half)
    if n_rings == 1:
        expected, code = number, number
    else:
        expected = extra.bytes(int(extra.integers(0, spec.capacity(n_rings, ec_level) + 1)))
        code = spec.encode_message(expected, n_rings, ec_level)
    SRC_PPU = 3.0 if n_rings <= 8 else 1.5         # big codes are rendered smaller, to bound memory
    code = render.render(code, SRC_PPU, ink, bgc, supersample=3 if n_rings <= 8 else 2, **center)
    if wavy:
        code = wave(code, seed, wavy, SRC_PPU)
    if mirrored:
        code = code[:, ::-1].copy()
    # Optionally the code is printed on a larger surface of the same background colour.
    pad = int(rng.choice([0, 0, 60, 400]))
    if pad:
        code = cv2.copyMakeBorder(code, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bgc)
    S = np.array([[1 / SRC_PPU, 0, -half - pad / SRC_PPU], [0, 1 / SRC_PPU, -half - pad / SRC_PPU], [0, 0, 1]])
    M = H @ S
    shrink = SRC_PPU / px_per_unit
    if shrink > 1.5:
        code = cv2.GaussianBlur(code, (0, 0), 0.45 * shrink)
    frame = scene_background(rng)
    warped = cv2.warpPerspective(code, M, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(code.shape[:2], 255, np.uint8), M, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)
    a = (mask.astype(np.float32) / 255)[..., None]
    img = frame.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a

    # Uneven light: a linear gradient, sometimes a hard shadow edge across the frame.
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    ramp = (xx * math.cos(ang) + yy * math.sin(ang))
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    lo = 1 - level * rng.uniform(0, 0.5)
    img *= (lo + (1 - lo) * ramp)[..., None]
    if rng.random() < 0.25 * level:
        ang = rng.uniform(0, 2 * math.pi)
        edge = (xx - FRAME_W / 2 + rng.uniform(-200, 200)) * math.cos(ang) + (yy - FRAME_H / 2) * math.sin(ang)
        shade = 1 - rng.uniform(0.2, 0.5) / (1 + np.exp(-edge / 6))
        img *= shade[..., None]
    if rng.random() < 0.2 * level:   # glare spot
        cx, cy = rng.uniform(0, FRAME_W), rng.uniform(0, FRAME_H)
        rad = rng.uniform(40, 200)
        img += (rng.uniform(40, 120) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad * rad)))[..., None]

    # Camera tone curve: a random gamma, which moves the light shade relative to solid and background.
    gamma = 1.0
    if rng.random() < 0.6:
        gamma = float(np.exp(rng.uniform(math.log(0.6), math.log(1.7))))
        img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** gamma

    # Blur, expressed in model units so it is equally severe at any code size.
    blur_units = 0.0
    if rng.random() < 0.7:
        blur_units = level * rng.uniform(0, 2.2)
        sigma = max(blur_units * px_per_unit, 0.01)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    motion_units = 0.0
    if rng.random() < 0.3:
        motion_units = level * rng.uniform(1, 6)
        n = max(1, int(round(motion_units * px_per_unit)))
        if n > 1:
            k = np.zeros((n, n), np.float32)
            k[n // 2, :] = 1.0 / n
            rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1)
            k = cv2.warpAffine(k, rot, (n, n))
            img = cv2.filter2D(img, -1, k / k.sum())
    noise = level * rng.uniform(0, 10)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.uniform(100 - 65 * min(level, 1.0), 95))
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    meta = dict(seed=seed, number=expected, rings=n_rings, ec=ec_level, star_px=39.0 * px_per_unit, diameter=diameter, tilt=tilt, blur=blur_units, motion=motion_units,
                noise=noise, jpeg=q, gamma=gamma, center=center_label, mirrored=mirrored, contrast=abs(luma(ink) - luma(bgc)), H=H)
    return img, meta


def make_negative(seed):
    rng = np.random.default_rng(seed)
    img = scene_background(rng).astype(np.float32)
    img += rng.normal(0, rng.uniform(0, 10), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_case(args):
    seed, level, rings, wavy = args
    img, meta = make_case(seed, level, rings, wavy)
    found = decode.scan(img)
    if not found:
        meta["result"] = "miss"
    elif len(found) == 1 and found[0]["key"] == meta["number"]:
        meta["result"] = "ok"
        meta["repaired"] = found[0]["repaired"]
        if found[0]["mirrored"] != meta["mirrored"]:
            meta["result"] = "wrong"
    else:
        meta["result"] = "wrong"
    return meta


def run_negative(seed):
    return len(decode.scan(make_negative(seed)))


def groups(rows, key, label):
    print(f"\n  by {label}")
    for value in sorted({r[key] for r in rows}, key=str):
        sel = [r for r in rows if r[key] == value]
        ok = sum(r["result"] == "ok" for r in sel)
        print(f"    {str(value):<14} {100 * ok / len(sel):5.1f}%  ({ok}/{len(sel)})")


def table(rows, key, edges, label):
    print(f"\n  by {label}")
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel:
            ok = sum(r["result"] == "ok" for r in sel)
            print(f"    {lo:>6g} - {hi:<6g} {100 * ok / len(sel):5.1f}%  ({ok}/{len(sel)})")


def main():
    args = sys.argv[1:]
    save = None
    if "--save-fails" in args:
        i = args.index("--save-fails")
        save = args[i + 1]
        del args[i:i + 2]
    rings = (1, 1)
    if "--rings" in args:
        i = args.index("--rings")
        lo, _, hi = args[i + 1].partition("-")
        rings = (int(lo), int(hi or lo))
        del args[i:i + 2]
    wavy = 0.0
    if "--wavy" in args:
        i = args.index("--wavy")
        wavy = float(args[i + 1])
        del args[i:i + 2]
    count = int(args[0]) if args else 1000
    level = float(args[1]) if len(args) > 1 else 1.0
    with Pool() as pool:
        rows = pool.map(run_case, [(s, level, rings, wavy) for s in range(count)], chunksize=4)
        neg = pool.map(run_negative, range(10 ** 6, 10 ** 6 + count // 2), chunksize=8)
    ok = sum(r["result"] == "ok" for r in rows)
    wrong = [r for r in rows if r["result"] == "wrong"]
    print(f"decoded {ok}/{count} ({100 * ok / count:.1f}%), wrong number {len(wrong)}, "
          f"needed a star repaired {sum(r.get('repaired', 0) for r in rows)}")
    print(f"false positives on {len(neg)} code-free frames: {sum(n > 0 for n in neg)}")
    if rings[1] == 1:
        table(rows, "diameter", [90, 130, 180, 250, 350, 560], "code width in px")
    else:
        groups(rows, "rings", "rings")
        groups(rows, "ec", "error-correction level")
    table(rows, "star_px", [10, 19, 26, 36, 50, 90], "star height in px")
    table(rows, "tilt", [0, 15, 30, 45, 55], "tilt in degrees")
    table(rows, "blur", [0, 0.01, 0.7, 1.4, 2.2, 9], "blur sigma, model units")
    table(rows, "motion", [0, 0.01, 3, 6, 99], "motion blur length, model units")
    table(rows, "contrast", [50, 80, 120, 256], "ink/background luma difference")
    table(rows, "gamma", [0.6, 0.8, 0.999, 1.001, 1.3, 1.71], "camera gamma")
    groups(rows, "center", "centre picture")
    groups(rows, "mirrored", "mirror image")
    fails = [r for r in rows if r["result"] != "ok"]
    print("\n  wrong seeds:", [r["seed"] for r in wrong])
    print("  missed seeds:", [r["seed"] for r in fails if r["result"] == "miss"][:60])
    if save:
        os.makedirs(save, exist_ok=True)
        for r in fails[:40]:
            cv2.imwrite(os.path.join(save, f"{r['result']}_{r['seed']}.jpg"), make_case(r["seed"], level, rings, wavy)[0])


if __name__ == "__main__":
    main()
