"""Decode rate against code size and blur, head-on and tilted, on photo backgrounds. Light noise only."""
import math
from multiprocessing import Pool

import cv2
import numpy as np

import bench
from astracode import decode, render, spec


def case(args):
    width, tilt, blur, seed = args
    rng = np.random.default_rng(seed)
    number = int(rng.integers(0, spec.MAX_NUMBER))
    H = None
    while H is None:
        H = bench.placement(rng, width, tilt)
    code = render.render(number, bench.SRC_PPU)
    S = np.array([[1 / bench.SRC_PPU, 0, -spec.HALF], [0, 1 / bench.SRC_PPU, -spec.HALF], [0, 0, 1]])
    ppu = width / (2 * spec.HALF)
    code = cv2.GaussianBlur(code, (0, 0), max(0.45 * bench.SRC_PPU / ppu, 0.01))
    frame = bench.scene_background(rng).astype(np.float32)
    size = (bench.FRAME_W, bench.FRAME_H)
    warped = cv2.warpPerspective(code, H @ S, size).astype(np.float32)
    a = cv2.warpPerspective(np.full(code.shape[:2], 1.0, np.float32), H @ S, size)[..., None]
    img = frame * (1 - a) + warped * a
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur * ppu)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    found = decode.scan(img)
    return (width, tilt, blur), ("ok" if [r["number"] for r in found] == [number] else "wrong" if found else "miss")


if __name__ == "__main__":
    widths, n = [50, 60, 70, 80, 100, 140], 60
    jobs = [(w, t, b, 7000 + i) for w in widths for t in (0, 45) for b in (0, 2.5, 3.5) for i in range(n)]
    with Pool() as pool:
        res = pool.map(case, jobs, chunksize=4)
    print("wrong reads:", sum(r == "wrong" for _, r in res))
    for t in (0, 45):
        for b in (0, 2.5, 3.5):
            row = [100 * sum(r == "ok" for k, r in res if k == (w, t, b)) / n for w in widths]
            print(f"tilt {t:2d}, blur {b:3.1f} units: " + "  ".join(f"{w}px {v:3.0f}%" for w, v in zip(widths, row)))
