"""Astracode scanner.

Pipeline (the same shape as a QR reader):
  1. Binarise with a local-mean threshold, in both polarities.
  2. Connected components -> candidate star blobs.
  3. Walk rings: from three neighbouring blobs, predict where the next star
     must be and step round until the ring closes after 12 (a few may be missing).
  4. Homography from the ring slots to the blob centres.
  5. Resolve which of the 12 ring rotations is "up" by correlating the fixed
     pattern (star cores, top limbs, the gaps between limbs, the logo).
  6. Refine: each star's core + top limb act as an alignment pattern; re-locate
     every star against the grey image and refit the homography. Repeat with
     the full shape of each star once its symbol has been read.
  7. Sample the four data limbs of each star, normalised by that star's own
     background and top limb, and threshold.
  8. Check with Reed-Solomon. No error correction: with only two check stars,
     "fixing" a read is how wrong numbers get through, and the next camera frame
     is a better second opinion.

Only step 1-3 use the binary image; everything after samples grey levels.
"""
import math
import sys

import cv2
import numpy as np

from . import spec
from .render import logo_alpha

# ------------------------------------------------------------ sample patterns

_BASE_W = 2 * spec.STAR_RI * math.sin(math.radians(36))   # limb width at its base
_BASE_R = spec.STAR_RI * math.cos(math.radians(36))       # distance of the base from the centre
_LIMB_LEN = spec.STAR_RO - _BASE_R


def _limb_points(k):
    a = math.radians(-90 + 72 * k)
    ax, ay = math.cos(a), math.sin(a)
    pts = []
    for r in (8.0, 9.5, 11.0, 12.5, 14.0):
        w = _BASE_W * (1 - (r - _BASE_R) / _LIMB_LEN)
        for t in (-0.25, 0.0, 0.25):
            pts.append((ax * r - ay * t * w, ay * r + ax * t * w))
    return pts


def _gap_points(k):
    a = math.radians(-90 + 36 + 72 * k)
    ax, ay = math.cos(a), math.sin(a)
    return [(ax * r - ay * t, ay * r + ax * t) for r in (15.0, 17.0, 19.0) for t in (-2.0, 0.0, 2.0)]


_CORE = np.array([(x, y) for x in np.arange(-3.75, 4, 1.5) for y in np.arange(-3.75, 4, 1.5)
                  if x * x + y * y <= 16.5], np.float32)
_LIMB = np.array([_limb_points(k) for k in range(5)], np.float32)        # 5 x 15 x 2
_GAP = np.array([_gap_points(k) for k in range(5)], np.float32)          # 5 x 9 x 2
_CENTERS = np.array([spec.star_center(i) for i in range(12)], np.float32)

# One star's sample points, concatenated: core | 5 limbs | 5 gaps.
_STAR_PTS = np.concatenate([_CORE, _LIMB.reshape(-1, 2), _GAP.reshape(-1, 2)])
_NC, _NL, _NG = len(_CORE), _LIMB.shape[1], _GAP.shape[1]
_ALL_STAR_PTS = (_CENTERS[:, None, :] + _STAR_PTS[None, :, :]).reshape(-1, 2)


def _build_fixed_pattern():
    """Model points whose ink value does not depend on the payload, with that value."""
    la = cv2.GaussianBlur(logo_alpha(), (0, 0), 6)
    ppu = la.shape[0] / spec.LOGO_D
    g = np.arange(-68, 69, 4.0)
    xy = np.array([(x, y) for x in g for y in g if x * x + y * y <= 70 * 70], np.float32)
    px = np.clip(np.round((xy + spec.LOGO_D / 2) * ppu).astype(int), 0, la.shape[0] - 1)
    logo_t = la[px[:, 1], px[:, 0]]
    star = np.concatenate([_CORE, _LIMB[0], _GAP.reshape(-1, 2)])
    star_t = np.concatenate([np.ones(_NC + _NL), np.zeros(5 * _NG)]).astype(np.float32)
    sxy = (_CENTERS[:, None, :] + star[None]).reshape(-1, 2)
    return (xy, logo_t), (sxy, np.tile(star_t, 12))


(_LOGO_XY, _LOGO_T), (_FIX_XY, _FIX_T) = _build_fixed_pattern()

# Dense star template for alignment: a grid over the star's disc, with a weight per
# grid point for each of the 16 shapes (+ inside, - outside), plus a shape-agnostic
# variant (index 16) that only uses what every star shares: core and top limb are ink,
# the wedges between limb positions are background, the four data limbs are ignored.
_SEARCH = np.array([(dx, dy) for dx in np.arange(-4, 4.1, 1.0) for dy in np.arange(-4, 4.1, 1.0)], np.float32)
_GRID = np.array([(x, y) for x in np.arange(-21, 21.1, 1.5) for y in np.arange(-21, 21.1, 1.5)
                  if x * x + y * y <= 22 * 22], np.float32)


def _build_align_weights():
    def inside(sym):
        poly = np.array(spec.star_polygon(sym), np.float32)
        return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in _GRID])
    def weights(ink, bg):
        return np.where(ink, 1.0 / ink.sum(), 0) - np.where(bg, 1.0 / bg.sum(), 0)
    full, bare = inside(15), inside(0)
    w = [weights(inside(sym), ~inside(sym)) for sym in range(16)]
    w.append(weights(bare, ~full))
    return np.array(w, np.float32)


_ALIGN_W = _build_align_weights()

# ------------------------------------------------------------ small geometry tools


def _apply(H, pts):
    p = pts @ H[:2, :2].T + H[:2, 2]
    w = pts @ H[2, :2] + H[2, 2]
    return p / w[:, None]


def _sample(gray, xy):
    """Bilinear samples of a float32 image at (n, 2) x,y positions."""
    h, w = gray.shape
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    a = gray[y0, x0] * (1 - fx) + gray[y0, x0 + 1] * fx
    b = gray[y0 + 1, x0] * (1 - fx) + gray[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def _homography(src, dst):
    """Least-squares homography (normalised DLT) mapping src -> dst, both (n, 2)."""
    def norm(p):
        c = p.mean(0)
        s = math.sqrt(2) / (np.sqrt(((p - c) ** 2).sum(1)).mean() + 1e-12)
        return (p - c) * s, np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    s, Ts = norm(np.asarray(src, np.float64))
    d, Td = norm(np.asarray(dst, np.float64))
    n = len(s)
    A = np.zeros((2 * n, 9))
    A[0::2, 0:2] = s
    A[0::2, 2] = 1
    A[0::2, 6:8] = -d[:, :1] * s
    A[0::2, 8] = -d[:, 0]
    A[1::2, 3:5] = s
    A[1::2, 5] = 1
    A[1::2, 6:8] = -d[:, 1:] * s
    A[1::2, 8] = -d[:, 1]
    h = np.linalg.svd(A)[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ h @ Ts
    return H / H[2, 2]


# ------------------------------------------------------------ ring detection

def _blobs(binary, max_area):
    n, _, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    w = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    h = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    ok = (area >= 9) & (area <= max_area) & (w < 3.5 * h) & (h < 3.5 * w) & (area > 0.12 * w * h)
    pts = cent[1:][ok].astype(np.float64)
    area = area[ok]
    if len(area) > 3500:
        keep = np.argsort(-area)[:3500]
        pts, area = pts[keep], area[keep]
    return pts, area


# Ring walking. Under an affine map, ring star j sits at C + A cos(30 j) + B sin(30 j),
# which is linear in C, A, B. So the next star is a fixed weighted sum of the last
# few: exact from three, least squares (less noise-sensitive) from four or five.
# Perspective bends this model over long arcs, so no more than five are used.
def _predict_weights(m):
    th = np.radians(30) * np.arange(m + 1)
    A = np.stack([np.ones(m + 1), np.cos(th), np.sin(th)], 1)
    return (A[m] @ np.linalg.pinv(A[:m])).tolist()


_PREDICT = {m: _predict_weights(m) for m in (3, 4, 5)}
_PREDICT_TOL = {3: 0.9, 4: 0.6, 5: 0.55}       # allowed miss, as a fraction of the last step
# A star has few similar blobs around it: its two ring neighbours and a little
# clutter. Blobs in a dense regular texture (screen pixel grids, halftone, fabric)
# have many, and would otherwise support endless false rings. Scan first with a
# strict limit, which is fast, and only if that finds nothing with a lenient one.
_MAX_NEIGHBOURS = (12, 24)
_WALK_BUDGET = 300000                           # ring-walk steps per binarised image


def _find_rings(pts, area, max_neighbours=_MAX_NEIGHBOURS[-1], max_missing=3):
    """Generate (blob indices, slots) for every closed ring of 12 evenly spaced, similar blobs.

    Starting from three neighbouring blobs, walk clockwise round the ring by
    prediction. A slot with no blob near its predicted place is carried as a
    virtual point; the walk must close onto its first two stars.
    """
    n = len(pts)
    if n < 12 - max_missing:
        return
    s = np.sqrt(area)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(2))
    smax = np.maximum(s[:, None], s[None, :])
    smin = np.minimum(s[:, None], s[None, :])
    adj = (d > 0.8 * smax) & (d < 6.5 * smax) & (smax < 2.3 * smin)
    nbrs = [np.nonzero(r)[0].tolist() for r in adj]
    # Wider neighbourhoods, for stepping across one missing star.
    far = [np.nonzero(r)[0].tolist() for r in (d > 0.8 * smax) & (d < 13 * smax) & (smax < 2.3 * smin)]
    P = [(float(x), float(y)) for x, y in pts]
    S = s.tolist()

    def near(x, y, radius, size, last, before):
        """Blobs of compatible size within radius of (x, y), closest first, taken from
        the neighbours of the previous star (or of the one before it, after a gap)."""
        pool = nbrs[last] if last >= 0 else far[before]
        out = []
        for i in pool:
            dd = (P[i][0] - x) ** 2 + (P[i][1] - y) ** 2
            if dd < radius * radius and size / 2.3 < S[i] < size * 2.3:
                out.append((dd, i))
        out.sort()
        return [i for _, i in out]

    budget = [_WALK_BUDGET]

    def walk(ids, pos, missing, size):
        """Extend a partial ring; yields complete id lists (-1 = missing star)."""
        budget[0] -= 1
        if budget[0] < 0:
            return
        step = len(ids)
        m = min(step, 5)
        w = _PREDICT[m]
        hist = pos[-m:]
        x = sum(wi * p[0] for wi, p in zip(w, hist))
        y = sum(wi * p[1] for wi, p in zip(w, hist))
        tol = _PREDICT_TOL[m] * math.hypot(pos[-1][0] - pos[-2][0], pos[-1][1] - pos[-2][1])
        if step >= 12:                               # must close onto the first two stars
            q = pos[step - 12]
            if math.hypot(q[0] - x, q[1] - y) <= tol:
                if step == 13:
                    yield ids[:12]
                else:
                    yield from walk(ids + [ids[0]], pos + [q], missing, size)
            return
        found = [i for i in near(x, y, tol, size, ids[-1], ids[-2]) if i not in ids]
        # Early on the prediction is loose, so try every blob in range; later only the closest.
        for i in found[:3] if step <= 4 else found[:1]:
            yield from walk(ids + [i], pos + [P[i]], missing, size)
        # The 4th star must be real, and no two may be missing in a row.
        if not found and step > 3 and ids[-1] >= 0 and missing < max_missing:
            yield from walk(ids + [-1], pos + [(x, y)], missing + 1, size)

    ring_of = [set() for _ in range(n)]
    seen = []
    crowded = [len(nb) > max_neighbours for nb in nbrs]
    # Largest blobs first, so real stars are tried before fine texture.
    for b in np.argsort(-s).tolist():
        if len(nbrs[b]) < 2 or crowded[b] or budget[0] < 0:
            continue
        bx, by = P[b]
        for a in nbrs[b]:
            if crowded[a]:
                continue
            ax, ay = P[a]
            l1 = math.hypot(bx - ax, by - ay)
            for c in nbrs[b]:
                if c == a or crowded[c] or (ring_of[a] & ring_of[b] & ring_of[c]):
                    continue
                cx, cy = P[c]
                l2 = math.hypot(cx - bx, cy - by)
                if not 0.55 < l2 / l1 < 1.8:
                    continue
                cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
                dot = (bx - ax) * (cx - bx) + (by - ay) * (cy - by)
                # Clockwise in image coordinates only, turning 5 to 85 degrees.
                if cross <= 0 or dot <= 0 or cross < 0.087 * dot:
                    continue
                for ids in walk([a, b, c], [P[a], P[b], P[c]], 0, (S[a] + S[b] + S[c]) / 3):
                    slots = [k for k, i in enumerate(ids) if i >= 0]
                    real = [ids[k] for k in slots]
                    key = frozenset(real)
                    if any(key <= k for k in seen):     # same ring, or part of one already reported
                        continue
                    seen.append(key)
                    for i in real:
                        ring_of[i].add(len(seen))
                    yield np.array(real), np.array(slots)


# ------------------------------------------------------------ reading

def _read_stars(gray, H):
    """Per star: core level, the 5 limb levels, and the 5 gap levels (gap k lies between limbs k and k+1)."""
    v = _sample(gray, _apply(H, _ALL_STAR_PTS)).reshape(12, -1)
    core = v[:, :_NC].mean(1)
    limb = v[:, _NC:_NC + 5 * _NL].reshape(12, 5, _NL).mean(2)
    gap = v[:, _NC + 5 * _NL:].reshape(12, 5, _NG).mean(2)
    return core, limb, gap


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _rot(k):
    a = math.radians(30 * k)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _orient(gray, H0, pol):
    """Pick which of the 12 ring rotations puts the top limbs on top. Returns (H, star score, logo score)."""
    best = None
    for k in range(12):
        H = H0 @ _rot(k)
        ss = -pol * _ncc(_sample(gray, _apply(H, _FIX_XY)), _FIX_T)
        ls = -pol * _ncc(_sample(gray, _apply(H, _LOGO_XY)), _LOGO_T)
        if best is None or ss + ls > best[0]:
            best = (ss + ls, H, ss, ls)
    return best[1], best[2], best[3]


def _refine(gray, H, pol, shapes):
    """Re-locate every star by template match, then refit the homography.

    shapes: the 12 symbols read so far, or None to use only the core and top limb.
    """
    pts = (_CENTERS[:, None, None, :] + _SEARCH[None, :, None, :] + _GRID[None, None, :, :])
    v = _sample(gray, _apply(H, pts.reshape(-1, 2))).reshape(12, len(_SEARCH), -1)
    w = _ALIGN_W[[16] * 12 if shapes is None else shapes]
    score = -pol * (v * w[:, None, :]).sum(2)
    best = score.argmax(1)
    found = np.empty((12, 2))
    for i in range(12):
        near = np.abs(_SEARCH - _SEARCH[best[i]]).max(1) <= 1.01
        sc = score[i, near] - score[i, near].min() + 1e-6
        found[i] = _CENTERS[i] + (sc[:, None] * _SEARCH[near]).sum(0) / sc.sum()
    return _homography(_CENTERS, _apply(H, found))


def _symbols(bits):
    return [int(r[0]) << 3 | int(r[1]) << 2 | int(r[2]) << 1 | int(r[3]) for r in bits]


def _bits(core, limb, gap, pol):
    """Limb presence per star and data limb, its confidence 0..1, and the top-limb strength.

    Each limb is compared with the background on its own two sides, as a ratio,
    so a shadow edge or light gradient across a star cancels out. The star's top
    limb, which is always present, sets the scale for "present".
    """
    eps = 8.0
    flank = 0.5 * (gap + np.roll(gap, 1, axis=1))
    q = pol * np.log((flank + eps) / (limb + eps))        # > 0 where the limb is inked
    top = np.maximum(q[:, 0], 1e-3)
    n = q[:, 1:] / top[:, None]
    q_core = pol * np.log((np.median(gap, 1) + eps) / (core + eps))
    return n > 0.5, np.clip(np.abs(n - 0.5) * 2, 0, 1), q[:, 0] / np.maximum(q_core, 1e-3)


# The two check stars catch every error confined to one or two stars, so a wrong
# number needs three or more stars misread at once. Refuse to answer when that
# many stars have a limb too close to the threshold to call.
WEAK_CONFIDENCE = 0.15
MAX_WEAK_STARS = 2


def _read_candidate(gray, centres, slots, pol, debug=None):
    ang = np.radians(30.0 * slots)
    model = np.stack([spec.R * np.cos(ang), spec.R * np.sin(ang)], 1)
    H0 = _homography(model, centres)
    # Blob centres sit within a few units of the star centres; a worse fit is not a ring.
    back = _apply(np.linalg.inv(H0), centres)
    if np.sqrt(((back - model) ** 2).sum(1)).max() > 9:
        return None
    H, star_score, logo_score = _orient(gray, H0, pol)
    if debug is not None:
        debug.append(dict(stage="orient", star_score=star_score, logo_score=logo_score))
    if star_score < 0.3 or logo_score < 0.2:
        return None
    H = _refine(gray, H, pol, None)
    for _ in range(2):
        bits = _bits(*_read_stars(gray, H), pol)[0]
        H = _refine(gray, H, pol, _symbols(bits))
    core, limb, gap = _read_stars(gray, H)
    bits, conf, top = _bits(core, limb, gap, pol)
    contrast = -pol * (core - np.median(gap, 1))
    logo_score = -pol * _ncc(_sample(gray, _apply(H, _LOGO_XY)), _LOGO_T)
    info = dict(stage="read", H=H, star_score=star_score, logo_score=logo_score, top_min=float(top.min()),
                contrast_min=float(contrast.min()), conf_min=float(conf.min()))
    if debug is not None:
        debug.append(info)
    if logo_score < 0.45 or top.min() < 0.3 or contrast.min() < 6:
        return None
    if (conf.min(1) < WEAK_CONFIDENCE).sum() > MAX_WEAK_STARS:
        return None
    number = spec.decode_symbols(_symbols(bits))
    if number is None:
        return None
    outline = _apply(H, np.array([[-spec.HALF, -spec.HALF], [spec.HALF, -spec.HALF],
                                  [spec.HALF, spec.HALF], [-spec.HALF, spec.HALF]]))
    return dict(number=number, text="%012d" % number, corners=outline,
                confidence=float(conf.min()), logo_score=logo_score,
                bit_confidence=conf, top_limb=float(top.min()))


def scan_gray(gray, debug=None):
    """Find and decode codes in a single-channel image. Returns a list of results."""
    gray = np.ascontiguousarray(gray, np.float32)
    h, w = gray.shape
    scale = min(1.0, 1280.0 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    max_area = (min(small.shape) / 5.0) ** 2
    passes = []
    for win in (31, 91):
        diff = small - cv2.blur(small, (win, win))
        for pol in (1, -1):                      # 1: dark ink on light, -1: light ink on dark
            binary = ((diff * pol) < -7).astype(np.uint8)
            passes.append((pol,) + _blobs(binary, max_area))
    results = []
    for limit in _MAX_NEIGHBOURS:
        for pol, pts, area in passes:
            for idx, slots in _find_rings(pts, area, limit):
                r = _read_candidate(gray, pts[idx] / scale, slots, pol, debug)
                if r is not None and all(r["number"] != q["number"] for q in results):
                    results.append(r)
        if results:
            break
    return results


def scan(image, debug=None):
    """Find and decode codes in a BGR or grey image. Falls back to single colour channels."""
    if image.ndim == 2:
        return scan_gray(image, debug)
    results = scan_gray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), debug)
    if not results:
        for ch in range(3):
            results = scan_gray(image[..., ch], debug)
            if results:
                break
    return results


if __name__ == "__main__":
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(path, "cannot read")
            continue
        found = scan(img)
        print(path, ", ".join(r["text"] for r in found) or "no code")
