"""Synthetic reliability test for the scanner.

Renders random codes, places them in a camera-like frame with perspective, blur,
noise, uneven light and JPEG damage, scans them, and reports hit / miss / wrong.

    .venv/bin/python bench.py [count] [level] [--save-fails DIR]
"""
import glob
import math
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from astracode import decode, render, spec

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
        s = rng.uniform(0.6, 1.0)
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


def placement(rng, diameter, tilt_deg):
    """Homography from code model units to frame pixels, for a tilted plane seen by a pinhole camera."""
    f = 1.1 * FRAME_W
    roll, axis, tilt = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), math.radians(tilt_deg)
    dist = f * 2 * spec.HALF / diameter
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
    corners = decode._apply(H, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * spec.HALF)
    lo, hi = corners.min(0), corners.max(0)
    span = hi - lo
    if span[0] > FRAME_W - 8 or span[1] > FRAME_H - 8:
        return None
    tx = rng.uniform(4 - lo[0], FRAME_W - 4 - hi[0])
    ty = rng.uniform(4 - lo[1], FRAME_H - 4 - hi[1])
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]) @ H


def make_case(seed, level=1.0):
    """level scales how harsh the damage is; 1.0 is the default mix."""
    rng = np.random.default_rng(seed)
    number = int(rng.integers(0, spec.MAX_NUMBER))
    ink, bgc = random_colours(rng)
    while True:
        diameter = float(np.exp(rng.uniform(math.log(90), math.log(560))))
        tilt = float(rng.uniform(0, 55))
        H = placement(rng, diameter, tilt)
        if H is not None:
            break
    px_per_unit = diameter / (2 * spec.HALF)
    code = render.render(number, SRC_PPU, ink, bgc)
    # Optionally the code is printed on a larger surface of the same background colour.
    pad = int(rng.choice([0, 0, 60, 400]))
    if pad:
        code = cv2.copyMakeBorder(code, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bgc)
    S = np.array([[1 / SRC_PPU, 0, -spec.HALF - pad / SRC_PPU], [0, 1 / SRC_PPU, -spec.HALF - pad / SRC_PPU], [0, 0, 1]])
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
    meta = dict(seed=seed, number=number, diameter=diameter, tilt=tilt, blur=blur_units, motion=motion_units,
                noise=noise, jpeg=q, gamma=gamma, contrast=abs(luma(ink) - luma(bgc)), H=H)
    return img, meta


def make_negative(seed):
    rng = np.random.default_rng(seed)
    img = scene_background(rng).astype(np.float32)
    img += rng.normal(0, rng.uniform(0, 10), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_case(args):
    seed, level = args
    img, meta = make_case(seed, level)
    found = decode.scan(img)
    if not found:
        meta["result"] = "miss"
    elif len(found) == 1 and found[0]["number"] == meta["number"]:
        meta["result"] = "ok"
        meta["repaired"] = found[0]["repaired"]
    else:
        meta["result"] = "wrong"
    return meta


def run_negative(seed):
    return len(decode.scan(make_negative(seed)))


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
    count = int(args[0]) if args else 1000
    level = float(args[1]) if len(args) > 1 else 1.0
    with Pool() as pool:
        rows = pool.map(run_case, [(s, level) for s in range(count)], chunksize=8)
        neg = pool.map(run_negative, range(10 ** 6, 10 ** 6 + count // 2), chunksize=8)
    ok = sum(r["result"] == "ok" for r in rows)
    wrong = [r for r in rows if r["result"] == "wrong"]
    print(f"decoded {ok}/{count} ({100 * ok / count:.1f}%), wrong number {len(wrong)}, "
          f"needed a star repaired {sum(r.get('repaired', 0) for r in rows)}")
    print(f"false positives on {len(neg)} code-free frames: {sum(n > 0 for n in neg)}")
    table(rows, "diameter", [90, 130, 180, 250, 350, 560], "code width in px")
    table(rows, "tilt", [0, 15, 30, 45, 55], "tilt in degrees")
    table(rows, "blur", [0, 0.01, 0.7, 1.4, 2.2, 9], "blur sigma, model units")
    table(rows, "motion", [0, 0.01, 3, 6, 99], "motion blur length, model units")
    table(rows, "contrast", [50, 80, 120, 256], "ink/background luma difference")
    table(rows, "gamma", [0.6, 0.8, 0.999, 1.001, 1.3, 1.71], "camera gamma")
    fails = [r for r in rows if r["result"] != "ok"]
    print("\n  wrong seeds:", [r["seed"] for r in wrong])
    print("  missed seeds:", [r["seed"] for r in fails if r["result"] == "miss"][:60])
    if save:
        os.makedirs(save, exist_ok=True)
        for r in fails[:40]:
            cv2.imwrite(os.path.join(save, f"{r['result']}_{r['seed']}.jpg"), make_case(r["seed"], level)[0])


if __name__ == "__main__":
    main()
