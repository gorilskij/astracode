"""Tables from the results of noise_grid.py.

    .venv/bin/python noise_report.py results.jsonl [--at 50] [--csv out.csv] [--against earlier.jsonl]

For every star size, a table of rings against light-limb level giving the noise
(standard deviation in grey levels; ink and background are 177 apart) at which
the read rate falls to 50%, or whatever --at says, interpolated between the
levels that were run. Dark on light first, then light on dark. Marks:

    >140   still above the rate at the heaviest noise run
    -      below the rate with no noise at all
    *      the code is wider than a phone camera frame (1440 px) at this star size
    !      some scan of this combination returned a wrong payload

With --against, each entry is followed by the same entry from an earlier run
in brackets, where the two differ by more than 2.
"""
import collections
import csv
import json
import sys

import noise_grid


def crossing(rows, rate):
    """Noise at which the read rate first falls to `rate`, interpolating linearly; None if it starts below."""
    rows = sorted(rows, key=lambda r: r["noise"])
    rates = [r["ok"] / r["codes"] for r in rows]
    if rates[0] < rate:
        return None
    for (a, ra), (b, rb) in zip(zip(rows, rates), zip(rows[1:], rates[1:])):
        if rb < rate:
            return a["noise"] + (b["noise"] - a["noise"]) * (ra - rate) / (ra - rb)
    return float("inf")


def load(path):
    """The rows of a results file: per combination, and the noise-only ones."""
    cells, blanks = collections.defaultdict(list), []
    for line in open(path):
        row = json.loads(line)
        if row.get("blank"):
            blanks.append(row)
        else:
            cells[(row["dark_on_light"], row["star"], row["rings"], row["light"])].append(row)
    return cells, blanks


def show(rows, rate):
    """A combination's entry as text, and as a number for comparing."""
    at = crossing(rows, rate)
    if at is None:
        return "-", -1.0
    if at == float("inf"):
        return ">%d" % max(x["noise"] for x in rows), at
    return "%.0f" % at, at


def main():
    args = sys.argv[1:]
    rate, out, earlier = 0.5, None, {}
    if "--against" in args:
        i = args.index("--against")
        earlier = load(args[i + 1])[0]
        del args[i:i + 2]
    if "--at" in args:
        i = args.index("--at")
        rate = float(args[i + 1]) / 100
        del args[i:i + 2]
    if "--csv" in args:
        i = args.index("--csv")
        out = args[i + 1]
        del args[i:i + 2]
    cells, blanks = load(args[0])
    width = 12 if earlier else 8
    stars = sorted({k[1] for k in cells})
    rings = sorted({k[2] for k in cells})
    light = sorted({k[3] for k in cells})
    print(f"{len(cells)} combinations, {sum(len(v) for v in cells.values())} noise levels, "
          f"{sum(r['codes'] for v in cells.values() for r in v)} scans; wrong reads: {sum(r['wrong'] for v in cells.values() for r in v)}")
    if blanks:
        print(f"frames with only noise: {sum(r['codes'] for r in blanks)} scanned, {sum(r['found'] for r in blanks)} false positives")
    table = []
    for dark_on_light in (True, False):
        for star in stars:
            print(f"\n{'dark on light' if dark_on_light else 'light on dark'}, stars {star} px: noise at which {100 * rate:.0f}% still read")
            print("  rings  " + "".join(f"{str(l) + '%':>{width}}" for l in light) + "    code width, px")
            for r in rings:
                line = f"  {r:>5}  "
                for l in light:
                    rows = cells.get((dark_on_light, star, r, l))
                    if not rows:
                        line += f"{'':>{width}}"
                        continue
                    at = crossing(rows, rate)
                    text, value = show(rows, rate)
                    text += "!" if any(x["wrong"] for x in rows) else ""
                    before = earlier.get((dark_on_light, star, r, l))
                    if before:
                        was, was_value = show(before, rate)
                        if abs(value - was_value) > 2:
                            text += f" ({was})"
                    line += f"{text:>{width}}"
                    table.append(dict(dark_on_light=dark_on_light, star=star, rings=r, light=l, noise=at,
                                      wrong=sum(x["wrong"] for x in rows), code_px=rows[0]["code_px"],
                                      clean_ms=min(rows, key=lambda x: x["noise"])["median_ms"]))
                side = round(noise_grid.code_px(r, star))
                print(line + f"    {side}{' *' if side > noise_grid.PHONE_FRAME else ''}")
    if out:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main()
