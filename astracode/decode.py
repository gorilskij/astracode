"""Astracode scanner (format version 2).

Pipeline (the same shape as a QR reader):
  1. Binarise with a local-mean threshold, in both polarities.
  2. Connected components -> candidate star blobs.
  3. Walk rings: from three neighbouring blobs, predict where the next star
     must be and step round until the ring closes after 12 (a few may be missing).
  4. Homography from the ring slots to the blob centres.
  5. Resolve which of the 12 ring rotations is "up" from the stars themselves:
     every star is a complete upright pentagram, whatever its data. Whatever
     is drawn in the centre is ignored.
  6. Refine: re-locate every star against the grey image by template match and
     refit the homography.
  7. Sample the five limbs of each star against the background beside them and
     that star's solid centre. Split all 60 limbs of the code into a light and
     a solid group, which calibrates the two shades for this photo.
  8. Reed-Solomon: repair at most one star, refuse if more look wrong.
  9. If that fails, read the code as a mirror image (seen from behind glass);
     a mirrored reading is accepted only if it needs no repair.

Only steps 1-3 use the binary image; everything after samples grey levels.
"""
import math
import sys

import cv2
import numpy as np

from . import spec

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


# Model points whose ink value does not depend on the payload, with that value.
_FIX_XY = _ALL_STAR_PTS
_FIX_T = np.tile(np.concatenate([np.ones(_NC + 5 * _NL), np.zeros(5 * _NG)]).astype(np.float32), 12)

# Dense star template for alignment: a grid over the star's disc, weighted + inside
# the pentagram and - outside it. Every star has the same outline in this format.
_SEARCH = np.array([(dx, dy) for dx in np.arange(-4, 4.1, 1.0) for dy in np.arange(-4, 4.1, 1.0)], np.float32)
_GRID = np.array([(x, y) for x in np.arange(-21, 21.1, 1.5) for y in np.arange(-21, 21.1, 1.5)
                  if x * x + y * y <= 22 * 22], np.float32)


def _build_align_weights():
    poly = np.array(spec.star_polygon(31), np.float32)
    ink = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in _GRID])
    return (np.where(ink, 1.0 / ink.sum(), 0) - np.where(~ink, 1.0 / (~ink).sum(), 0)).astype(np.float32)


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
# The last pass has no limit: in a code with several rings every star is crowded.
_MAX_NEIGHBOURS = (12, 24, 10 ** 9)
_NEAREST = 8                                    # a star's ring neighbours are among its nearest similar blobs
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
    def nearest(mask, keep):
        out = []
        for i, row in enumerate(mask):
            idx = np.nonzero(row)[0]
            if len(idx) > keep:
                idx = idx[np.argsort(d[i, idx])[:keep]]
            out.append(idx.tolist())
        return out

    adj = (d > 0.8 * smax) & (d < 6.5 * smax) & (smax < 2.3 * smin)
    crowded = (adj.sum(1) > max_neighbours).tolist()
    nbrs = nearest(adj, _NEAREST)
    # Wider neighbourhoods, for stepping across one missing star.
    far = nearest((d > 0.8 * smax) & (d < 13 * smax) & (smax < 2.3 * smin), 3 * _NEAREST)
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

def _read_stars(gray, H, centers=_CENTERS):
    """Per star: core level, the 5 limb levels, and the 5 gap levels (gap k lies between limbs k and k+1)."""
    n = len(centers)
    pts = (centers[:, None, :] + _STAR_PTS[None, :, :]).reshape(-1, 2)
    v = _sample(gray, _apply(H, pts)).reshape(n, -1)
    core = v[:, :_NC].mean(1)
    limb = v[:, _NC:_NC + 5 * _NL].reshape(n, 5, _NL).mean(2)
    gap = v[:, _NC + 5 * _NL:].reshape(n, 5, _NG).mean(2)
    return core, limb, gap


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _rot(k):
    a = math.radians(30 * k)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ring_points(radius, count, offset=0.0):
    a = np.radians(offset + 360.0 * np.arange(count) / count)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], 1).astype(np.float32)


# The band between the centre picture and the stars is always background. In a code
# with several rings, loops of twelve stars turn up that are not ring 0; they have
# stars where this band should be.
_BAND_XY = _ring_points(0.5 * (spec.CENTER_R + spec.R - spec.STAR_RO), 72)
_BETWEEN_XY = _ring_points(spec.R, 12, 15.0)       # background, midway between ring 0 stars


def _band_clear(gray, H0, pol):
    core = _sample(gray, _apply(H0, _CENTERS)).mean()
    bg = np.median(_sample(gray, _apply(H0, _BETWEEN_XY)))
    ink = (bg - _sample(gray, _apply(H0, _BAND_XY))) / (bg - core + pol * 1e-3)
    return (ink > 0.5).mean() <= 0.1


_ORIENT_PTS = (_CENTERS[:, None, :] + _GRID[None, :, :]).reshape(-1, 2)
_MIRROR = np.diag([-1.0, 1.0, 1.0])


def _orient(gray, H0, pol):
    """Pick which of the 12 ring rotations has the stars upright. Returns (H, star score, margin).

    Turning the ring by one star position tilts every star by 30 degrees. Five or
    seven positions leave them only 6 degrees off upright, so this uses the dense
    star outline, which notices that, rather than a few points per limb.
    """
    scores = []
    for k in range(12):
        v = _sample(gray, _apply(H0 @ _rot(k), _ORIENT_PTS)).reshape(12, -1)
        scores.append(float(-pol * (v * _ALIGN_W).sum()))
    order = np.argsort(scores)
    H = H0 @ _rot(int(order[-1]))
    star_score = -pol * _ncc(_sample(gray, _apply(H, _FIX_XY)), _FIX_T)
    margin = (scores[order[-1]] - scores[order[-2]]) / (abs(scores[order[-1]]) + 1e-6)
    return H, star_score, margin


def _locate(gray, H, pol, centers):
    """Re-locate stars by template match around where H puts them. Returns their image positions."""
    n = len(centers)
    found = np.empty((n, 2))
    for at in range(0, n, 64):                       # in batches, to bound memory
        c = centers[at:at + 64]
        pts = (c[:, None, None, :] + _SEARCH[None, :, None, :] + _GRID[None, None, :, :])
        v = _sample(gray, _apply(H, pts.reshape(-1, 2))).reshape(len(c), len(_SEARCH), -1)
        score = -pol * (v * _ALIGN_W).sum(2)
        best = score.argmax(1)
        for i in range(len(c)):
            near = np.abs(_SEARCH - _SEARCH[best[i]]).max(1) <= 1.01
            sc = score[i, near] - score[i, near].min() + 1e-6
            found[at + i] = c[i] + (sc[:, None] * _SEARCH[near]).sum(0) / sc.sum()
    return _apply(H, found)


def _refine(gray, H, pol, centers=_CENTERS):
    """Re-locate every star by template match, then refit the homography."""
    return _homography(centers, _locate(gray, H, pol, centers))


def _limb_strength(core, limb, gap, pol):
    """How inked each limb is, on a scale where 0 is background and about 1 is solid.

    Each limb is compared with the background on its own two sides, so a shadow
    edge or light gradient across a star cancels out; the star's always-solid
    centre sets the scale.
    """
    eps = 8.0
    flank = 0.5 * (gap + np.roll(gap, 1, axis=1))
    bg = np.median(gap, 1)
    # Contrast of the limb against its flanks, relative to the core's contrast, with
    # the local light level (flank vs the star's typical background) divided out.
    num = pol * (flank - limb) / (flank + eps)
    den = pol * (bg - core) / (bg + eps)
    return num / np.maximum(den, 1e-3)[:, None]


def _two_groups(v):
    """Split values into a low and a high group (1-D 2-means). Returns (low centre, high centre)."""
    lo, hi = np.percentile(v, 10), np.percentile(v, 90)
    for _ in range(20):
        high = v > 0.5 * (lo + hi)
        if high.all() or not high.any():
            break
        new_lo, new_hi = v[~high].mean(), v[high].mean()
        if abs(new_lo - lo) + abs(new_hi - hi) < 1e-5:
            break
        lo, hi = new_lo, new_hi
    return float(lo), float(hi)


# Limb model:
#   log strength = a[limb direction] + plane(star position) + d[star] + light * log(ratio)
# The per-star offsets d are pulled towards zero (ridge). A star's own five limbs
# easily outweigh that, so a star dimmed by glare or shadow gets its own level; but
# for a star whose limbs are all one shade, it is what says "that shade, at the usual
# level" rather than "the other shade, at an unusual level".
_STAR_RIDGE = 4.0


def _fit_limb_model(v, usable, light, centers):
    """Least-squares fit of the limb model. Returns (v minus everything but the light term, log ratio).

    The per-star offsets are eliminated exactly, star by star, which leaves an
    8-column problem however many stars there are.
    """
    n = len(centers)
    G = np.zeros((n, 5, 8))
    G[:, np.arange(5), np.arange(5)] = 1
    G[:, :, 5] = centers[:, :1] / spec.R
    G[:, :, 6] = centers[:, 1:] / spec.R
    G[:, :, 7] = light.reshape(n, 5)
    V = v.reshape(n, 5)
    U = usable.reshape(n, 5).astype(float)
    count = U.sum(1)
    # Minimising over d[star] leaves r' (I - 11'/(count + ridge)) r per star; this is its square root.
    a = (1 - np.sqrt(_STAR_RIDGE / (count + _STAR_RIDGE))) / np.maximum(count, 1)
    Gt = (G - a[:, None, None] * (G * U[:, :, None]).sum(1, keepdims=True)) * U[:, :, None]
    Vt = (V - a[:, None] * (V * U).sum(1, keepdims=True)) * U
    coef = np.linalg.lstsq(Gt.reshape(-1, 8), Vt.ravel(), rcond=None)[0]
    r = V - G @ coef
    d = (r * U).sum(1) / (count + _STAR_RIDGE)
    resid = V - G[:, :, :7] @ coef[:7] - d[:, None]
    return resid.ravel(), float(coef[7])


def _classify(strength, centers=_CENTERS):
    """Decide light or solid for every limb. Returns (solid mask n x 5, confidence n x 5, ratio, separation).

    Blur, motion and lighting weaken limbs unevenly: by limb direction (motion
    blur), smoothly across the code (perspective, focus), and star by star
    (glare, reflections). So rather than one threshold, fit a model with those
    three effects plus the light/solid ratio, alternating with the assignment.
    """
    n = len(centers)
    flat = strength.ravel()
    # A limb washed out by glare or covered reads as hardly inked at all. Leave such
    # limbs out of the fit and report them with zero confidence.
    usable = flat > 0.3 * np.median(flat)
    v = np.log(np.maximum(flat, 1e-3))
    lo, hi = _two_groups(v[usable])
    light = v < 0.5 * (lo + hi)
    for _ in range(10):
        resid, log_ratio = _fit_limb_model(v, usable, light, centers)
        log_ratio = min(log_ratio, -0.05)
        new = resid < 0.5 * log_ratio
        if (new == light).all():
            break
        light = new
    light = resid < 0.5 * log_ratio
    conf = np.clip(np.abs(resid - 0.5 * log_ratio) / (0.5 * abs(log_ratio)), 0, 1)
    conf[~usable] = 0
    err = np.where(light, resid - log_ratio, resid)[usable]
    separation = abs(log_ratio) / (err.std() + 1e-6)
    return (~light).reshape(n, 5), conf.reshape(n, 5), float(np.exp(log_ratio)), float(separation)


def _symbols(bits):
    return [int(r[0]) << 4 | int(r[1]) << 3 | int(r[2]) << 2 | int(r[3]) << 1 | int(r[4]) for r in bits]


# The check stars repair one misread star anywhere, or else the two least certain
# stars. Refuse to answer when too many stars have a limb close to the light/solid
# threshold, or when the two shades are not clearly apart: that is when several
# stars can be wrong at once and a repair would be a guess.
WEAK_CONFIDENCE = 0.25      # distance from the threshold, as a fraction of half the gap between shades
MAX_WEAK_STARS = 3
CLEAR_CONFIDENCE = 0.4      # what every other star must reach before two stars may be rebuilt
MIN_SHADE_SEPARATION = 4.0  # gap between the two shades, in units of their spread


def _read_candidate(gray, centres, slots, pol, debug=None):
    ang = np.radians(30.0 * slots)
    model = np.stack([spec.R * np.cos(ang), spec.R * np.sin(ang)], 1)
    H0 = _homography(model, centres)
    # Blob centres sit on the star centres; a poor fit is not a ring.
    back = _apply(np.linalg.inv(H0), centres)
    if np.sqrt(((back - model) ** 2).sum(1)).max() > 9:
        return None
    if not _band_clear(gray, H0, pol):
        return None
    H, star_score, margin = _orient(gray, H0, pol)
    if debug is not None:
        debug.append(dict(stage="orient", star_score=star_score, margin=margin))
    if star_score < 0.3:
        return None
    for _ in range(3):
        H = _refine(gray, H, pol)
    # A mirror image also shows twelve upright stars; it is the same ring read
    # anticlockwise with each star's limbs swapped left to right.
    return _read_oriented(gray, H, pol, False, debug) or _read_oriented(gray, H @ _MIRROR, pol, True, debug)


def _read_oriented(gray, H, pol, mirrored, debug):
    core, limb, gap = _read_stars(gray, H)
    strength = _limb_strength(core, limb, gap, pol)
    contrast = -pol * (core - np.median(gap, 1))
    bits, conf, ratio, separation = _classify(strength)
    info = dict(stage="read", H=H, mirrored=mirrored, limb_min=float(strength.min()),
                contrast_min=float(contrast.min()), conf_min=float(conf.min()), ratio=ratio,
                separation=separation, strength=strength, bits=bits)
    if debug is not None:
        debug.append(info)
    if contrast.min() < 6 or not 0.3 < ratio < 0.97:
        return None
    if separation < MIN_SHADE_SEPARATION or (conf.min(1) < WEAK_CONFIDENCE).sum() > MAX_WEAK_STARS:
        return None
    # Rebuilding the two least certain stars is only safe when every other star is clear.
    order = np.argsort(conf.min(1))
    doubtful = (int(order[0]), int(order[1])) if conf[order[2]].min() >= CLEAR_CONFIDENCE else None
    if mirrored:
        doubtful = None
    symbols = _symbols(bits)
    decoded = spec.decode_symbols(symbols, doubtful)
    if decoded is None:
        header = spec.read_header(symbols, doubtful)
        if header is None or (mirrored and header[3]):
            return None
        return _read_rings(gray, H, pol, mirrored, header, debug)
    if mirrored and decoded[1]:
        return None
    number, repaired = decoded
    outline = _apply(H, np.array([[-spec.HALF, -spec.HALF], [spec.HALF, -spec.HALF],
                                  [spec.HALF, spec.HALF], [-spec.HALF, spec.HALF]]))
    metadata, payload = spec.unpack(number)
    return dict(key=number, rings=1, number=number, text="%012d" % number, metadata=metadata, payload=payload,
                member_id=spec.member_id(payload), corners=outline, repaired=repaired, mirrored=mirrored,
                confidence=float(conf.min()), separation=separation,
                ratio=ratio, bit_confidence=conf)


_RING_CENTERS = [np.array([spec.star_center(i, k) for i in range(spec.ring_stars(k))], np.float32)
                 for k in range(spec.MAX_RINGS)]


def _read_rings(gray, H, pol, mirrored, header, debug):
    """Read the outer rings of a code whose ring 0 announced them."""
    rings, level, value, _ = header
    # Work outwards: each ring is located with the fit from the rings inside it, then joins the fit.
    centers = _CENTERS
    found = _apply(H, _CENTERS)
    for k in range(1, rings):
        centers = np.concatenate([centers, _RING_CENTERS[k]])
        for attempt in range(2):
            here = _locate(gray, H, pol, _RING_CENTERS[k])
            H = _homography(centers, np.concatenate([found, here]))
        found = np.concatenate([found, here])
    core, limb, gap = _read_stars(gray, H, centers)
    strength = _limb_strength(core, limb, gap, pol)
    contrast = -pol * (core - np.median(gap, 1))
    bits, conf, ratio, separation = _classify(strength, centers)
    if debug is not None:
        debug.append(dict(stage="rings", H=H, rings=rings, level=level, contrast_min=float(contrast.min()),
                          ratio=ratio, separation=separation, strength=strength, bits=bits, conf=conf))
    if not 0.3 < ratio < 0.97 or separation < MIN_SHADE_SEPARATION:
        return None
    symbols, at = _symbols(bits), 12
    ring_symbols = [spec._encode_value(value)]       # ring 0 as its (possibly repaired) header says
    for k in range(1, rings):
        ring_symbols.append(symbols[at:at + spec.ring_stars(k)])
        at += spec.ring_stars(k)
    decoded = spec.decode_message(ring_symbols)
    if decoded is None:
        return None
    message, repaired = decoded
    half = spec.half_side(rings)
    outline = _apply(H, np.array([[-half, -half], [half, -half], [half, half], [-half, half]]))
    return dict(key=message, rings=rings, level=level, message=message, text=message.decode("utf-8", "replace"),
                corners=outline, repaired=repaired, mirrored=mirrored, confidence=float(conf.min()),
                separation=separation, ratio=ratio, bit_confidence=conf)


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
                if r is not None and all(r["key"] != q["key"] for q in results):
                    results.append(r)
                if results and limit > 1000:         # the slow pass: one code is enough
                    return results
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
        print(path, ", ".join("metadata %d, payload %010d (member ID %s)" % (r["metadata"], r["payload"], r["member_id"])
                              if r["rings"] == 1 else "%d rings, level %d: %r" % (r["rings"], r["level"], r["text"])
                              for r in found) or "no code")
