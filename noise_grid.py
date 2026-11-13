"""How much noise the scanner takes, by number of rings, light-limb level, star size and polarity.

    (cd rust && cargo build --release)
    .venv/bin/python noise_grid.py results.jsonl [--codes 50] [--rings 1,2,3] [--light 50,70] [--stars 14,24] [--pilot]

For every combination of rings, light-limb level and star size, CODES random
codes are drawn clean (navy on yellow, a random rotation, a tilt of up to 20
degrees, a slight blur) and then scanned with more and more Gaussian noise
added, until fewer than half of them read. Each frame is scanned both as drawn
and inverted (the same code, light on dark). Noise is the only thing that
changes between scans of a frame.

Noise is in grey levels (0 to 255) on the brightness plane, which is what the
scanner works on; ink and background are 177 grey levels apart. Star size is
the star's height, tip to tip across its circle, in pixels (39 model units).

One line of JSON is appended to the results file per combination, polarity and
noise level, as it finishes, so a run can be watched, stopped and resumed:
combinations already in the file are skipped. `noise_report.py` turns the file
into tables.

Every combination runs in its own process, with a single-threaded scanner, so
scan times are per frame on one core. ASTRA_WORKERS sets how many run at once
(three quarters of the cores by default).
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np

from astracode import spec

ROOT = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(ROOT, "rust", "target", "release", "astracode")
WORKERS = int(os.environ.get("ASTRA_WORKERS", max(1, os.cpu_count() * 3 // 4)))

RINGS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 22, 24)
LIGHT = (50, 60, 70, 80, 90)
STARS = (14, 18, 24, 32, 44)
NOISE = (0, 4, 8, 12, 16, 20, 25, 30, 36, 43, 50, 60, 70, 85, 100, 120, 140)
LEVEL = 1                      # error-correction level of every code with more than one ring
INK, BACKGROUND = "#021639", "#febc5c"
STAR_UNITS = 2 * spec.STAR_RO  # a star's height in model units
PHONE_FRAME = 1440             # the shorter side of the app's camera frame


def code_px(rings, star_px):
    """Side of the code's square in pixels when its stars are star_px tall."""
    return 2 * spec.half_side(rings) * star_px / STAR_UNITS


def make_frame(rings, light, star_px, seed, folder):
    """Draw one clean frame. Returns (path, payload as hex)."""
    rng = np.random.default_rng(seed)
    payload = rng.bytes(spec.capacity(rings, LEVEL)).hex()
    side = code_px(rings, star_px)
    # Drawn half as large again as it will appear, then warped down.
    drawn = int(math.ceil(1.5 * side))
    source = os.path.join(folder, "source_%d.png" % seed)
    subprocess.run([BINARY, "encode", "--hex", payload, "--rings", str(rings), "--level", str(LEVEL), "--light", str(light),
                    "--ink", INK, "--background", BACKGROUND, "--size", str(drawn), "-o", source],
                   check=True, capture_output=True)
    code = cv2.imread(source)
    os.remove(source)
    # The brightness plane, with the scanner's own weights.
    b, g, r = (code[..., i].astype(np.uint32) for i in range(3))
    gray = ((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14).astype(np.uint8)

    # Where the code lands: any rotation, a tilt of up to 20 degrees about a random axis,
    # in a frame with a quarter of the code's width to spare.
    frame = int(math.ceil(1.25 * side)) + 8
    spin, tilt, axis = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(0, 20)), rng.uniform(0, 2 * math.pi)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2
    rot = np.array([[math.cos(spin), -math.sin(spin)], [math.sin(spin), math.cos(spin)]])
    flat = corners @ rot.T
    # Tilt: points move away along the axis direction and shrink with distance (camera 3 code widths away).
    along = flat @ np.array([math.cos(axis), math.sin(axis)])
    depth = 3 * side + along * math.sin(tilt)
    across = flat - np.outer(along, [math.cos(axis), math.sin(axis)])
    seen = (across + np.outer(along * math.cos(tilt), [math.cos(axis), math.sin(axis)])) * (3 * side / depth)[:, None]
    target = seen + frame / 2 + rng.uniform(-0.5, 0.5, 2)
    src = np.array([[0, 0], [drawn, 0], [drawn, drawn], [0, drawn]], np.float32)
    H = cv2.getPerspectiveTransform(src, target.astype(np.float32))
    background = int(gray[2, 2])
    # Smooth before shrinking so that the warp does not alias.
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.6)
    out = cv2.warpPerspective(smooth, H, (frame, frame), flags=cv2.INTER_LINEAR, borderValue=background)
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    path = os.path.join(folder, "frame_%d.png" % seed)
    cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return path, payload


def scan(listing, noise, invert):
    """Scan every frame of a listing with this much noise. Returns per frame (seconds, [(bits, hex, rings, mirrored)])."""
    command = [BINARY, "batch", listing, "--threads", "1", "--gray", "--noise", str(noise), "--seed", str(noise)]
    out = subprocess.run(command + (["--invert"] if invert else []), check=True, capture_output=True, text=True)
    results = []
    for line in out.stdout.splitlines():
        fields = line.split("\t")
        found = [(int(fields[at]), fields[at + 1], int(fields[at + 2]), fields[at + 4] == "1") for at in range(2, len(fields), 7)]
        results.append((float(fields[1]), found))
    return results


def run_cell(args):
    """One combination of rings, light level and star size: draw its codes, then sweep the noise in both polarities."""
    rings, light, star_px, codes = args
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        frames = [make_frame(rings, light, star_px, 1000 * (1000 * rings + light) + 100 * star_px + i, folder) for i in range(codes)]
        listing = os.path.join(folder, "frames.txt")
        with open(listing, "w") as f:
            f.write("\n".join(path for path, _ in frames) + "\n")
        for invert in (False, True):
            below = 0
            for noise in NOISE:
                ok = wrong = 0
                seconds = []
                for (_, payload), (took, found) in zip(frames, scan(listing, noise, invert)):
                    seconds.append(took)
                    expected = (4 * len(payload), payload or "-", rings, False)
                    if len(found) == 1 and found[0] == expected:
                        ok += 1
                    elif found:
                        wrong += 1
                rows.append(dict(rings=rings, light=light, star=star_px, dark_on_light=not invert, noise=noise, codes=codes, ok=ok,
                                 wrong=wrong, median_ms=round(1000 * float(np.median(seconds)), 2),
                                 slowest_ms=round(1000 * max(seconds), 2), code_px=round(code_px(rings, star_px))))
                # Carry on one level past the first that reads fewer than half, to see that it stays down.
                below = below + 1 if ok < codes / 2 else 0
                if below == 2 or (ok == 0 and wrong == 0):
                    break
    return rows


def run_blank(args):
    """Frames with no code, only noise: anything found is a false positive."""
    noise, count = args
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for i in range(count):
            path = os.path.join(folder, "blank_%d.png" % i)
            cv2.imwrite(path, np.full((720, 960), 197 if i % 2 else 20, np.uint8))
            paths.append(path)
        listing = os.path.join(folder, "frames.txt")
        with open(listing, "w") as f:
            f.write("\n".join(paths) + "\n")
        results = scan(listing, noise, False)
    return dict(blank=True, noise=noise, codes=count, found=sum(bool(found) for _, found in results),
                median_ms=round(1000 * float(np.median([t for t, _ in results])), 2))


def numbers(args, flag, default):
    if flag not in args:
        return default
    i = args.index(flag)
    values = tuple(int(v) for v in args[i + 1].split(","))
    del args[i:i + 2]
    return values


def main():
    args = sys.argv[1:]
    rings, light, stars = numbers(args, "--rings", RINGS), numbers(args, "--light", LIGHT), numbers(args, "--stars", STARS)
    codes = numbers(args, "--codes", (50,))[0]
    results = args[0]
    if not os.path.exists(BINARY):
        sys.exit("build the scanner first: (cd rust && cargo build --release)")
    done, blanks_done = set(), set()
    if os.path.exists(results):
        for line in open(results):
            row = json.loads(line)
            if row.get("blank"):
                blanks_done.add(row["noise"])
            elif not row["dark_on_light"]:             # a combination's inverted rows are written last
                done.add((row["rings"], row["light"], row["star"]))
    cells = [(r, l, s, codes) for r in rings for l in light for s in stars if (r, l, s) not in done]
    # In a scrambled but repeatable order, so that a run stopped part way has covered a bit of everything.
    np.random.default_rng(7).shuffle(cells)
    print(f"{len(cells)} combinations to run, {len(done)} already done; {WORKERS} at a time", flush=True)
    with Pool(WORKERS) as pool, open(results, "a") as out:
        for row in pool.imap_unordered(run_blank, [(noise, 200) for noise in NOISE if noise not in blanks_done]):
            out.write(json.dumps(row) + "\n")
        out.flush()
        for n, rows in enumerate(pool.imap_unordered(run_cell, cells), 1):
            # Only whole combinations are written, so that a stopped run leaves nothing half done.
            out.writelines(json.dumps(row) + "\n" for row in rows)
            out.flush()
            first = rows[0]
            print(f"{n}/{len(cells)}  {first['rings']} rings, light {first['light']}%, stars {first['star']} px", flush=True)


if __name__ == "__main__":
    main()
