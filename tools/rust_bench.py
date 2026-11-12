"""Run the Rust scanner on the same synthetic frames as bench.py, and compare it with the Python one.

    .venv/bin/python tools/rust_bench.py [count] [level] [--rings N | --rings LOW-HIGH] [--wavy STRAIN] [--no-python]

Frames come from bench.make_case with the same seeds as bench.py, are written
as PNG files to a temporary directory a few dozen at a time, and are scanned by `astracode batch`
(build it first: `cargo build --release` in rust/). Both stages use
bench.WORKERS processes or threads (three quarters of the cores unless
ASTRA_WORKERS says otherwise). Scan times for the Rust
scanner are measured one frame at a time inside that process; the Python
scanner's are measured while all cores are busy, so they are on the high side.
"""
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bench  # noqa: E402
from astracode import decode  # noqa: E402

BINARY = os.path.join(ROOT, "rust", "target", "release", "astracode")
BATCH = 36      # frames on disk at any one time


def prepare(args):
    """Make one frame, save it, and scan it with the Python scanner if asked."""
    kind, seed, level, rings, wavy, folder, with_python = args
    if kind == "code":
        img, meta = bench.make_case(seed, level, rings, wavy)
        path = os.path.join(folder, "code_%05d.png" % seed)
    else:
        img, meta = bench.make_negative(seed), dict(number=None, mirrored=False, rings=0, star_px=0.0)
        path = os.path.join(folder, "none_%07d.png" % seed)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    row = dict(path=path, expected=meta["number"], mirrored=meta["mirrored"], rings=meta["rings"], star_px=meta["star_px"])
    if with_python:
        start = time.perf_counter()
        found = decode.scan(img)
        row["python"] = [(r["key"], r["mirrored"]) for r in found]
        row["python_seconds"] = time.perf_counter() - start
    return row


def verdict(found, row):
    """ok / miss / wrong for a list of (value, mirrored) results."""
    if not found:
        return "miss"
    return "ok" if len(found) == 1 and found[0] == (row["expected"], row["mirrored"]) else "wrong"


def parse(line):
    """One line of `astracode batch` output -> (path, seconds, [(value, mirrored)])."""
    fields = line.rstrip("\n").split("\t")
    found = []
    for at in range(2, len(fields), 7):
        kind, value, _rings, _level, mirrored = fields[at:at + 5]
        value = int(value) if kind == "number" else bytes.fromhex("" if value == "-" else value)
        found.append((value, mirrored == "1"))
    return fields[0], float(fields[1]), found


def timing(label, rows, key, which):
    for name in ("ok", "miss"):
        t = np.array([r[key] for r in rows if r[which] == name])
        if len(t):
            print(f"  {label} scan time, {'read' if name == 'ok' else 'missed'}: median {np.median(t) * 1000:.0f} ms, "
                  f"90% under {np.percentile(t, 90) * 1000:.0f} ms, slowest {t.max() * 1000:.0f} ms")


def main():
    args = sys.argv[1:]
    with_python = "--no-python" not in args
    args = [a for a in args if a != "--no-python"]
    rings, wavy = (1, 1), 0.0
    if "--rings" in args:
        i = args.index("--rings")
        lo, _, hi = args[i + 1].partition("-")
        rings = (int(lo), int(hi or lo))
        del args[i:i + 2]
    if "--wavy" in args:
        i = args.index("--wavy")
        wavy = float(args[i + 1])
        del args[i:i + 2]
    count = int(args[0]) if args else 300
    level = float(args[1]) if len(args) > 1 else 1.0
    if not os.path.exists(BINARY):
        sys.exit("build the Rust scanner first: (cd rust && cargo build --release)")

    # Frames go through in small batches that are deleted as soon as they have been
    # scanned: large frames are 10 MB each as PNG, and hundreds of them fill a disk.
    rows, elapsed = [], 0.0
    with tempfile.TemporaryDirectory() as folder, Pool(bench.workers(rings)) as pool:
        jobs = [("code", s, level, rings, wavy, folder, with_python) for s in range(count)]
        jobs += [("none", s, level, rings, wavy, folder, with_python) for s in range(10 ** 6, 10 ** 6 + count // 2)]
        listing = os.path.join(folder, "frames.txt")
        for at in range(0, len(jobs), BATCH):
            batch = pool.map(prepare, jobs[at:at + BATCH], chunksize=1)
            with open(listing, "w") as f:
                f.write("\n".join(r["path"] for r in batch) + "\n")
            start = time.perf_counter()
            out = subprocess.run([BINARY, "batch", listing, "--threads", str(bench.WORKERS)], capture_output=True, text=True)
            elapsed += time.perf_counter() - start
            if out.returncode:
                sys.exit(out.stderr)
            for row, line in zip(batch, out.stdout.splitlines()):
                path, seconds, found = parse(line)
                assert path == row["path"]
                row["rust"], row["rust_seconds"] = found, seconds
                os.remove(path)
            rows += batch

    codes = [r for r in rows if r["expected"] is not None]
    blanks = [r for r in rows if r["expected"] is None]
    for name in ("rust",) + (("python",) if with_python else ()):
        for r in codes:
            r[name + "_verdict"] = verdict(r[name], r)
        ok = sum(r[name + "_verdict"] == "ok" for r in codes)
        wrong = sum(r[name + "_verdict"] == "wrong" for r in codes)
        false = sum(bool(r[name]) for r in blanks)
        print(f"{name:>6}: decoded {ok}/{len(codes)} ({100 * ok / len(codes):.1f}%), wrong {wrong}, "
              f"false positives {false}/{len(blanks)}")
        timing(name, codes, name + "_seconds", name + "_verdict")
        if blanks:
            t = np.array([r[name + "_seconds"] for r in blanks])
            print(f"  {name} scan time, no code in frame: median {np.median(t) * 1000:.0f} ms, slowest {t.max() * 1000:.0f} ms")
    print(f"  the Rust scanner took {elapsed:.1f}s for all {len(rows)} frames, PNG decoding included")
    if with_python:
        both = sum(r["rust_verdict"] == "ok" and r["python_verdict"] == "ok" for r in codes)
        only_rust = [os.path.basename(r["path"]) for r in codes if r["rust_verdict"] == "ok" and r["python_verdict"] != "ok"]
        only_python = [os.path.basename(r["path"]) for r in codes if r["python_verdict"] == "ok" and r["rust_verdict"] != "ok"]
        print(f"both read {both}; only Rust {len(only_rust)}; only Python {len(only_python)}")
        if only_python:
            print("  read by Python only:", " ".join(only_python[:20]))
        if only_rust:
            print("  read by Rust only:", " ".join(only_rust[:20]))
    print("  by star height in px (Rust):")
    for lo, hi in ((10, 19), (19, 26), (26, 36), (36, 50), (50, 90)):
        sel = [r for r in codes if lo <= r["star_px"] < hi]
        if sel:
            ok = sum(r["rust_verdict"] == "ok" for r in sel)
            print(f"    {lo:>3} - {hi:<3} {100 * ok / len(sel):5.1f}%  ({ok}/{len(sel)})")


if __name__ == "__main__":
    main()
