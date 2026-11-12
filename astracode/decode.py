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
     refit the homography. Each star is then read where it was actually found
     (smoothed over its neighbours), not where the flat fit puts it, so a
     curved or wavy surface is followed locally. Every star acts as what a QR
     code calls an alignment pattern.
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
    # Largest blobs first, so real stars are tried before fine texture. In the pass for
    # crowded blobs, which is the one that finds codes with several rings, start instead
    # from the middle of the crowd: that is where ring 0 is, and everything further out
    # only offers look-alike loops.
    order = np.argsort(-s)
    dense = adj.sum(1) > _MAX_NEIGHBOURS[0]
    if max_neighbours > 1000 and dense.sum() >= 12:
        middle = np.median(pts[dense], 0)
        order = np.argsort(((pts - middle) ** 2).sum(1))
    for b in order.tolist():
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

def _jacobians(H, centers):
    """Local linear map from model offsets to image offsets at each star: (n, 2, 2)."""
    c = np.asarray(centers, np.float64)
    p = _apply(H, c)
    return np.stack([_apply(H, c + [1.0, 0.0]) - p, _apply(H, c + [0.0, 1.0]) - p], 2)


def _local_fit(H, centers, found, dense=False):
    """Where and how to read each star on a surface that may not be flat: (positions, local maps).

    Around each star, how far its neighbours were found from the flat fit H is
    fitted with a small linear model: its constant term moves the star, its
    slope adjusts the star's local scale and rotation. One star's measured
    position is noisy; a neighbourhood's is not.

    With ring 0 alone a star has neighbours only along the ring, so the slope is
    held towards zero (ridge) and this is little more than a weighted average of
    the offsets. With more rings (dense) every star is surrounded, and a tight
    neighbourhood with a free slope follows strong bending.
    """
    sigma, slope_ridge = (_DENSE_SIGMA, _DENSE_RIDGE) if dense else (_LOCAL_SIGMA, _SLOPE_RIDGE)
    c = np.asarray(centers, np.float64)
    res = found - _apply(H, c)
    jac = _jacobians(H, c)
    pos = np.empty_like(res)
    ridge = np.diag([0.0, slope_ridge, slope_ridge])
    for at in range(0, len(c), 256):
        d = c[None, :, :] - c[at:at + 256, None, :]                    # m x n x 2, neighbour minus star
        w = np.exp(-(d ** 2).sum(2) / (2 * sigma ** 2))
        X = np.concatenate([np.ones(d.shape[:2] + (1,)), d], 2)        # m x n x 3
        A = np.einsum("mn,mni,mnj->mij", w, X, X) + ridge
        coef = np.linalg.solve(A, np.einsum("mn,mni,nj->mij", w, X, res))   # m x 3 x 2
        pos[at:at + 256] = coef[:, 0, :]
        jac[at:at + 256] += coef[:, 1:, :].transpose(0, 2, 1)
    return _apply(H, c) + pos, jac


_LOCAL_SIGMA = 45.0          # neighbourhood size, model units
_SLOPE_RIDGE = 3000.0
_DENSE_SIGMA = 28.0
_DENSE_RIDGE = 30.0


def _read_stars(gray, pos, jac):
    """Per star: core level, the 5 limb levels, and the 5 gap levels (gap k lies between limbs k and k+1).

    pos: image position of each star's centre; jac: its local model-to-image map.
    """
    n = len(pos)
    pts = pos[:, None, :] + np.einsum("nij,pj->npi", jac, _STAR_PTS.astype(np.float64))
    v = _sample(gray, pts.reshape(-1, 2)).reshape(n, -1)
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
    """The band inside ring 0, and the gaps between its stars, must be background."""
    core = _sample(gray, _apply(H0, _CENTERS)).mean()
    band = _sample(gray, _apply(H0, _BAND_XY))
    between = _sample(gray, _apply(H0, _BETWEEN_XY))
    bg = np.median(np.concatenate([band, between]))
    scale = bg - core + pol * 1e-3
    return ((bg - band) / scale > 0.5).mean() <= 0.1 and ((bg - between) / scale > 0.5).sum() <= 1


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


def _locate(gray, H, pol, centers, search=_SEARCH):
    """Re-locate stars by template match around where H puts them. Returns their image positions."""
    n = len(centers)
    found = np.empty((n, 2))
    step = max(1, 5000 // len(search))               # in batches, to bound memory
    for at in range(0, n, step):
        c = centers[at:at + step]
        pts = (c[:, None, None, :] + search[None, :, None, :] + _GRID[None, None, :, :])
        v = _sample(gray, _apply(H, pts.reshape(-1, 2))).reshape(len(c), len(search), -1)
        score = -pol * (v * _ALIGN_W).sum(2)
        best = score.argmax(1)
        for i in range(len(c)):
            near = np.abs(search - search[best[i]]).max(1) <= 1.01
            sc = score[i, near] - score[i, near].min() + 1e-6
            found[at + i] = c[i] + (sc[:, None] * search[near]).sum(0) / sc.sum()
    return _apply(H, found)


# For a surface that bends: stars can be this far from where a flat fit puts them.
_SEARCH_WIDE = np.array([(dx, dy) for dx in np.arange(-8, 8.1, 1.0) for dy in np.arange(-8, 8.1, 1.0)], np.float32)


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


def _classify(strength, centers=_CENTERS, robust=False):
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
    # With hundreds of stars, a few ruined ones (torn off, covered) must not decide whether
    # the two shades count as separable: that is for the error correction. Use a spread
    # that ignores outliers.
    spread = 1.4826 * np.median(np.abs(err - np.median(err))) if robust else err.std()
    separation = abs(log_ratio) / (spread + 1e-6)
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


def _read_candidate(gray, centres, slots, pol, debug=None, tried=None, count=None, areas=None):
    if debug is not None:
        debug.append(dict(stage="candidate"))
    ang = np.radians(30.0 * slots)
    model = np.stack([spec.R * np.cos(ang), spec.R * np.sin(ang)], 1)
    H0 = _homography(model, centres)
    # Blob centres sit on the star centres; a poor fit is not a ring.
    back = _apply(np.linalg.inv(H0), centres)
    if np.sqrt(((back - model) ** 2).sum(1)).max() > 9:
        return None
    if not _band_clear(gray, H0, pol):
        return None
    if count is not None:
        count["plausible"] += 1
    H, star_score, margin = _orient(gray, H0, pol)
    if debug is not None:
        debug.append(dict(stage="orient", star_score=star_score, margin=margin))
    if star_score < 0.3:
        return None
    # The same ring turns up many times: in each binarised image, and with different
    # stars missing. If this placement has been read already, do not read it again.
    marks = _apply(H, _MARKS)
    if tried is not None:
        tol = 3.0 * math.hypot(*(marks[1] - marks[0])) / 100.0           # 3 model units
        if any(np.abs(marks - m).max() < tol for m in tried):
            return None
        tried.append(marks)
    H = _refine(gray, H, pol)
    if not _stars_fit(gray, H, pol) or _centre_is_code(gray, H, pol):
        return None
    for _ in range(2):
        H = _refine(gray, H, pol)
    # A mirror image also shows twelve upright stars; it is the same ring read
    # anticlockwise with each star's limbs swapped left to right.
    bent = dict(H=H, blobs=None if areas is None else (centres, areas))
    return (_read_oriented(gray, H, pol, False, debug, count, bent)
            or _read_oriented(gray, H @ _MIRROR, pol, True, debug, count, bent))


_MARKS = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
_SIZE_PTS = [(_CENTERS[:, None, :] + k * _GRID[None, :, :]).reshape(-1, 2) for k in (0.85, 1.0, 1.18)]
_QUIET_XY = _ring_points(spec.R + spec.STAR_RO + 0.5 * spec.MARGIN, 72)


def _stars_fit(gray, H, pol):
    """Are the stars the right size for this ring?

    Deep inside a code with many rings, a star's twelve second-nearest
    neighbours form a loop that passes for ring 0, except that its stars are
    too big for it. Compare the star outline at three sizes.
    """
    small, right, big = (float(-pol * (_sample(gray, _apply(H, p)).reshape(12, -1) * _ALIGN_W).sum()) for p in _SIZE_PTS)
    return right > 0 and (big - small) / right <= 0.06


_INNER_SCALES = (1.0, 1.2, 1.4)
_INNER_NEAR = np.array([(x, y) for x in (-6, -3, 0, 3, 6) for y in (-6, -3, 0, 3, 6)], np.float64)
_INNER_ALL = np.array([(x, y) for x in np.arange(-66, 67, 3.0) for y in np.arange(-66, 67, 3.0)
                       if x * x + y * y <= 66 * 66], np.float64)


def _centre_is_code(gray, H, pol):
    """Is the middle of this loop itself part of a code: a star just like the loop's own, with more around it?

    That is what the look-alike loops deep inside a big code have. Any other
    picture is fine there, including a single star of any size.
    """
    def score(at, k):
        pts = (at[:, None, :] + k * _GRID[None, :, :]).reshape(-1, 2)
        return -pol * (_sample(gray, _apply(H, pts)).reshape(len(at), -1) * _ALIGN_W).sum(1)

    ring = [float(score(_CENTERS.astype(np.float64), k).mean()) for k in _INNER_SCALES]
    k = _INNER_SCALES[int(np.argmax(ring))]
    need = 0.7 * max(ring)
    if need <= 0 or score(_INNER_NEAR, k).max() < need:
        return False                                   # no star like the ring's at the centre
    sc = score(_INNER_ALL, k)
    stars = []                                         # distinct stars: strongest first, at least 25 units apart
    for i in np.argsort(-sc):
        if sc[i] < need:
            break
        if all(np.hypot(*(_INNER_ALL[i] - q)) >= 25 for q in stars):
            stars.append(_INNER_ALL[i])
    return len(stars) >= 5


def _quiet(gray, H, pol):
    """A one-ring code has clear margin outside its stars; ring 0 of a bigger code, or a look-alike loop, does not."""
    core = _sample(gray, _apply(H, _CENTERS)).mean()
    bg = np.median(_sample(gray, _apply(H, _BETWEEN_XY)))
    ink = (bg - _sample(gray, _apply(H, _QUIET_XY))) / (bg - core + pol * 1e-3)
    return (ink > 0.5).mean() <= 0.1


def _read_ring0(gray, pos, jac, pol, mirrored, debug, H):
    """Read ring 0's limbs at the given star positions. Returns (symbols, doubtful stars, details) or None."""
    core, limb, gap = _read_stars(gray, pos, jac)
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
    return _symbols(bits), None if mirrored else doubtful, dict(conf=conf, ratio=ratio, separation=separation)


def _fit_shapes(gray, pos, jac, pol):
    """Fit each star's own position, stretch and shear to the image. Returns (positions, local maps).

    For a surface bent so sharply that neighbouring stars say too little about
    each other. Each star's local map is adjusted, one term at a time, to where
    the star outline matches best, and its centre likewise; first in coarse steps,
    then finer ones.
    """
    n = len(pos)
    pos, jac = pos.copy(), jac.copy()
    thin = 3 if n > 200 else 1                       # with hundreds of stars, a third of the outline points will do
    grid, weight = _GRID[::thin].astype(np.float64), _ALIGN_W[::thin]

    def score(p, j):
        out = np.empty(n)
        for at in range(0, n, 256):
            pts = p[at:at + 256, None, :] + np.einsum("nij,pj->npi", j[at:at + 256], grid)
            out[at:at + 256] = -pol * (_sample(gray, pts.reshape(-1, 2)).reshape(-1, len(grid)) * weight).sum(1)
        return out

    best = score(pos, jac)
    for steps, shift in (((-0.12, -0.06, 0.06, 0.12), 1.0), ((-0.12, -0.06, 0.06, 0.12), 1.0),
                         ((-0.03, 0.03), 0.5), ((-0.015, 0.015), 0.25)):
        for row, col in ((0, 0), (1, 1), (0, 1), (1, 0)):          # stretch along x, along y, then the two shears
            for step in steps:
                change = np.eye(2)
                change[row, col] += step
                trial = jac @ change
                sc = score(pos, trial)
                better = sc > best
                jac[better], best[better] = trial[better], sc[better]
        for dx, dy in ((-shift, 0), (shift, 0), (0, -shift), (0, shift)):      # and the centre
            trial = pos + np.einsum("nij,j->ni", jac, np.array([dx, dy], np.float64))
            sc = score(trial, jac)
            better = sc > best
            pos[better], best[better] = trial[better], sc[better]
    return pos, jac


def _twist(pos, jac):
    """How far, in degrees, the stars' fitted "up" is turned from what their places on the ring imply.

    Read at the right rotation this is about zero. Read s ring positions round,
    every star is 30 s degrees off upright; a pentagram looks the same every 72
    degrees, so a shape fit quietly settles on the nearest upright-looking match,
    and each star's own map ends up turned against the direction to its ring
    neighbours by 30 s minus the nearest multiple of 72: 6 degrees for five
    positions, -12 for two, 18 for three and so on, a different multiple of 6
    for every s. So the star shapes alone say which rotation is right, even on
    a surface that twists the stars themselves by more than that.
    """
    c = _CENTERS.astype(np.float64)
    chord = np.roll(pos, -1, 0) - np.roll(pos, 1, 0)                 # from the star before to the star after
    pred = np.einsum("nij,nj->ni", jac, np.roll(c, -1, 0) - np.roll(c, 1, 0))
    ang = np.arctan2(pred[:, 0] * chord[:, 1] - pred[:, 1] * chord[:, 0], (pred * chord).sum(1))
    return float(np.degrees(np.median(ang)))                         # a few badly fitted stars must not sway it


def _turn(pos, jac, steps):
    """Star positions and maps renumbered for the ring read `steps` positions further round."""
    a = math.radians(30 * steps)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.roll(pos, -steps, 0), np.roll(jac, -steps, 0) @ rot


def _bent_ring0(gray, H, pol, blobs):
    """Ring 0 on a surface that is far from flat: (H, positions, local maps) from the stars themselves, or None.

    Starts each star from its blob (its centre, and its size relative to the
    others) where there is one, fits each star's shape, and then checks the
    rotation against the shapes: see _twist. H comes back turned if the rotation
    had to be corrected.
    """
    flat = _apply(H, _CENTERS.astype(np.float64))
    jac = _jacobians(H, _CENTERS)
    pos = flat.copy()
    has = np.zeros(12, bool)
    if blobs is not None:
        centres, areas = blobs
        d = np.sqrt(((flat[:, None, :] - centres[None, :, :]) ** 2).sum(2))
        mine = d.argmin(1)
        spacing = np.sqrt(((flat - np.roll(flat, 1, 0)) ** 2).sum(1))
        has = d[np.arange(12), mine] < 0.3 * spacing
        if has.sum() >= 6:
            size = np.sqrt(areas[mine] / np.abs(np.linalg.det(jac)))
            rel = np.clip(size / np.median(size[has]), 0.6, 1.7)
            pos[has] = centres[mine[has]]
            jac[has] *= rel[has, None, None]
        else:
            has[:] = False
    if not has.all():                              # stars without a blob: look around where the flat fit puts them
        pos[~has] = _locate(gray, H, pol, _CENTERS[~has], _SEARCH_WIDE)
    pos, jac = _fit_shapes(gray, pos, jac, pol)
    for _ in range(3):
        sixths = int(round(_twist(pos, jac) / 6.0))
        if sixths == 0:
            return H, pos, jac
        steps = (-5 * sixths) % 12                 # a twist of 6 q degrees means 5 q positions round
        H = H @ _rot(steps)
        pos, jac = _fit_shapes(gray, *_turn(pos, jac, steps), pol)
    return (H, pos, jac) if round(_twist(pos, jac) / 6.0) == 0 else None


_MAX_BENT = 4               # rings per frame that get the bent-surface treatment
_FLIP = np.diag([-1.0, 1.0])
_FLIP_ORDER = (-np.arange(12)) % 12


def _bent(gray, pol, mirrored, shared, count):
    """_bent_ring0 for this candidate, worked out once and shared by its plain and mirrored readings.

    A pentagram is its own mirror image, so the mirrored reading uses the same
    twelve fitted stars, numbered the other way round.
    """
    if "fit" not in shared:
        if count is not None:
            count["bent"] = count.get("bent", 0) + 1
        allowed = count is None or count["bent"] <= _MAX_BENT
        shared["fit"] = _bent_ring0(gray, shared["H"], pol, shared["blobs"]) if allowed else None
    fit = shared["fit"]
    if fit is None or not mirrored:
        return fit
    H, pos, jac = fit
    return H @ _MIRROR, pos[_FLIP_ORDER], jac[_FLIP_ORDER] @ _FLIP


def _with_ring1(gray, H, pol, found, jac=None):
    """If a ring 1 of stars surrounds ring 0, where and how to read ring 0 given both; else None.

    On a surface that bends, ring 0 alone cannot show how it is stretched across
    the ring, only along it. The stars of ring 1 can.
    """
    both = np.concatenate(_locate_rings(gray, H, pol, 2, found, jac))
    centers = np.concatenate(_RING_CENTERS[:2])
    pos, jac = _local_fit(_homography(centers, both), centers, both, dense=True)
    core, limb, gap = _read_stars(gray, pos, jac)
    contrast = -pol * (core - np.median(gap, 1))
    if (contrast[12:] > 0.5 * np.median(contrast[:12])).sum() < 15:
        return None
    return pos[:12], jac[:12]


def _read_oriented(gray, H, pol, mirrored, debug, count=None, shared=None):
    found = _locate(gray, H, pol, _CENTERS)
    decoded = header = shape = None
    read = _read_ring0(gray, *_local_fit(H, _CENTERS, found), pol, mirrored, debug, H)
    if read is not None:
        decoded = spec.decode_symbols(read[0], read[1])
        if decoded is None:
            header = spec.read_header(read[0], read[1])
    if decoded is None and header is None:
        # Not readable as a flat ring. Try it as a bent one: place and shape each star
        # on its own; then, if a ring 1 surrounds it, use that ring's stars as well.
        bent = _bent(gray, pol, mirrored, shared if shared is not None else dict(H=H, blobs=None), count)
        if debug is not None:
            debug.append(dict(stage="bent", mirrored=mirrored, H=H, fit=bent))
        if bent is None:
            return None
        H, found, shape = bent
        shaped = (found, shape)
        for attempt in (lambda: shaped, lambda: _with_ring1(gray, H, pol, *shaped)):
            where = attempt()
            read = None if where is None else _read_ring0(gray, *where, pol, mirrored, debug, H)
            if read is None:
                continue
            # No rebuilding of two doubtful stars here: this path gives a ring several
            # chances to be read, and that repair is the easiest for noise to pass.
            decoded = spec.decode_symbols(read[0], None)
            header = None if decoded is not None else spec.read_header(read[0], None)
            if decoded is not None or header is not None:
                break
        else:
            return None
    if decoded is None:
        if header is None or (mirrored and header[3]):
            return None
        if count is not None:
            count["headers"] += 1
        return _read_rings(gray, H, pol, mirrored, header, found, shape, debug)
    conf, ratio, separation = read[2]["conf"], read[2]["ratio"], read[2]["separation"]
    # A one-ring code has clear margin outside its stars and, above all, no ring 1. Ring 0
    # of a bigger code read with a star "repaired" can pass for a one-ring code otherwise.
    if (mirrored and decoded[1]) or not _quiet(gray, H, pol) or _with_ring1(gray, H, pol, found, shape) is not None:
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


_kernels = {}


def _star_kernel(ppu):
    """The dense star template as a correlation kernel at ppu pixels per model unit."""
    key = round(ppu, 2)
    if key not in _kernels:
        r = int(math.ceil(22 * ppu))
        fine = 4
        n = 2 * r + 1
        g = ((np.arange(n * fine) + 0.5) / fine - n / 2) / ppu           # model units, sub-pixel grid
        xx, yy = np.meshgrid(g, g)
        mask = np.zeros(xx.shape, np.uint8)
        poly = (np.array(spec.star_polygon(31)) * ppu * fine + n * fine / 2 - 0.5)
        cv2.fillPoly(mask, [np.round(poly * 64).astype(np.int32)], 1, shift=6)
        ink = mask.astype(np.float32)
        out = ((xx * xx + yy * yy <= 22 * 22) & (mask == 0)).astype(np.float32)
        ink = ink.reshape(n, fine, n, fine).sum((1, 3))
        out = out.reshape(n, fine, n, fine).sum((1, 3))
        _kernels[key] = ink / ink.sum() - out / out.sum()
    return _kernels[key]


_SEARCH_UNITS = 7.0                 # how far from its predicted place a star is looked for
_STAR_SIZES = (0.75, 1.0, 1.3)      # template sizes tried, relative to the flat fit
_CHUNKS = (3, 7, 15, spec.MAX_RINGS)   # re-straighten the image after this many rings are located


def _predict_ring(model_new, model_known, where_known, use_all):
    """Predict where new stars are from a local affine fit to their nearest located neighbours."""
    if use_all:
        idx = np.tile(np.arange(len(model_known)), (len(model_new), 1))
    else:
        d2 = ((model_new[:, None, :] - model_known[None, :, :]) ** 2).sum(2)
        idx = np.argpartition(d2, 8, axis=1)[:, :8]
    X = np.concatenate([model_known[idx], np.ones(idx.shape + (1,))], 2)      # n x q x 3
    Y = where_known[idx]                                                       # n x q x 2
    XtX = np.einsum("nqi,nqj->nij", X, X) + 1e-9 * np.eye(3)
    coef = np.linalg.solve(XtX, np.einsum("nqi,nqj->nij", X, Y))               # n x 3 x 2
    return np.einsum("ni,nij->nj", np.concatenate([model_new, np.ones((len(model_new), 1))], 1), coef)


def _locate_rings(gray, H, pol, rings, found0, jac0=None):
    """Image positions of every star of rings 0..rings-1, working outwards from ring 0.

    The image is straightened with the fit so far and correlated with the star
    template once; each new ring is then looked for where its already located
    neighbours say it should be, which follows a surface that is not flat.

    jac0: ring 0's local maps as fitted from the star shapes, if known. Ring 0
    by itself only shows how the surface runs along the ring; the shapes of its
    stars also show how it runs outwards, towards ring 1.
    """
    centers = [np.asarray(c, np.float64) for c in _RING_CENTERS[:rings]]
    found = [found0]
    native = math.sqrt(abs(np.linalg.det(_jacobians(H, np.zeros((1, 2)))[0])))
    ppu = float(np.clip(native, 0.8, 2.0))
    kernels = [_star_kernel(ppu * k) for k in _STAR_SIZES]
    done = 1
    for stop in _CHUNKS:
        stop = min(stop, rings)
        if stop <= done:
            continue
        known = np.concatenate(centers[:done])
        Hfit = _homography(known, np.concatenate(found))
        half = spec.ring_radius(stop - 1) + spec.STAR_RO + 12
        side = int(math.ceil(2 * half * ppu))
        to_image = Hfit @ np.array([[1 / ppu, 0, -half], [0, 1 / ppu, -half], [0, 0, 1]])   # straightened px -> image
        flat = cv2.warpPerspective(gray, to_image, (side, side), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        # A bent surface shows some stars larger and some smaller; take the best of several sizes.
        score = None
        for kernel in kernels:
            one = -pol * cv2.filter2D(flat, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
            score = one if score is None else np.maximum(score, one)
        from_image = np.linalg.inv(to_image)
        where = [_apply(from_image, f) for f in found]                 # located stars, in straightened px
        r = int(math.ceil(_SEARCH_UNITS * ppu))
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        for k in range(done, stop):
            recent = slice(max(0, k - 2), k)
            if k == 1 and jac0 is not None:
                d2 = ((centers[1][:, None, :] - centers[0][None, :, :]) ** 2).sum(2)
                two = np.argsort(d2, axis=1)[:, :2]                             # each star's two nearest in ring 0
                step = centers[1][:, None, :] - centers[0][two]                 # n x 2 x 2, model units
                image = (found0[two] + np.einsum("nqij,nqj->nqi", jac0[two], step)).mean(1)
                guess = _apply(from_image, image)
            else:
                guess = _predict_ring(centers[k], np.concatenate(centers[recent]), np.concatenate(where[recent]), k == 1)
            gx = np.clip(np.round(guess[:, 0]).astype(int), r + 1, side - r - 2)
            gy = np.clip(np.round(guess[:, 1]).astype(int), r + 1, side - r - 2)
            win = score[gy[:, None, None] + dy, gx[:, None, None] + dx].reshape(len(guess), -1)
            best = win.argmax(1)
            bx, by = gx + dx.ravel()[best], gy + dy.ravel()[best]
            # sub-pixel: centroid of the 3 x 3 scores around the peak
            sy, sx = np.mgrid[-1:2, -1:2]
            near = score[by[:, None, None] + sy, bx[:, None, None] + sx].reshape(len(guess), 9)
            wgt = near - near.min(1, keepdims=True) + 1e-6
            here = np.stack([bx + (wgt * sx.ravel()).sum(1) / wgt.sum(1), by + (wgt * sy.ravel()).sum(1) / wgt.sum(1)], 1)
            where.append(here)
            found.append(_apply(to_image, here))
        done = stop
    return found


def _read_rings(gray, H, pol, mirrored, header, found0, jac0, debug):
    """Read the outer rings of a code whose ring 0 announced them."""
    rings, level, value, _ = header
    found = np.concatenate(_locate_rings(gray, H, pol, rings, found0, jac0))
    centers = np.concatenate(_RING_CENTERS[:rings])
    H = _homography(centers, found)
    where = _local_fit(H, centers, found, dense=True)
    # First with each star placed by its neighbours, which is all a flat or gently curved
    # surface needs. If that does not decode, fit every star's own outline as well: on a
    # strongly bent surface the neighbours leave a star about a unit off centre, enough
    # to misread one limb in a hundred, which is one star pair in ten.
    for shaped in (False, True):
        if shaped:
            where = _fit_shapes(gray, *where, pol)
        core, limb, gap = _read_stars(gray, *where)
        strength = _limb_strength(core, limb, gap, pol)
        contrast = -pol * (core - np.median(gap, 1))
        bits, conf, ratio, separation = _classify(strength, centers, robust=True)
        if debug is not None:
            debug.append(dict(stage="rings", H=H, rings=rings, level=level, shaped=shaped, contrast_min=float(contrast.min()),
                              ratio=ratio, separation=separation, strength=strength, bits=bits, conf=conf))
        if not 0.3 < ratio < 0.97 or separation < MIN_SHADE_SEPARATION:
            continue
        symbols, at = _symbols(bits), 12
        ring_symbols = [spec._encode_value(value)]       # ring 0 as its (possibly repaired) header says
        for k in range(1, rings):
            ring_symbols.append(symbols[at:at + spec.ring_stars(k)])
            at += spec.ring_stars(k)
        decoded = spec.decode_message(ring_symbols)
        if decoded is not None:
            break
    else:
        return None
    message, repaired = decoded
    half = spec.half_side(rings)
    outline = _apply(H, np.array([[-half, -half], [half, -half], [half, half], [-half, half]]))
    return dict(key=message, rings=rings, level=level, message=message, text=message.decode("utf-8", "replace"),
                corners=outline, repaired=repaired, mirrored=mirrored, shaped=shaped, confidence=float(conf.min()),
                separation=separation, ratio=ratio, bit_confidence=conf)


# Stars are looked for at full resolution. Set this to a size in pixels (1280, say) to look
# for them in a shrunken copy instead: quicker on large frames without a code, but small
# stars are missed. Reading always uses the full image.
FIND_SIZE = None


_MAX_LOOPS = 600            # loops of twelve looked at per image
_MAX_PLAUSIBLE = 30         # of those, how many may pass the first checks


def scan_gray(gray, debug=None, count=None):
    """Find and decode codes in a single-channel image. Returns a list of results."""
    gray = np.ascontiguousarray(gray, np.float32)
    h, w = gray.shape
    scale = 1.0 if FIND_SIZE is None else min(1.0, FIND_SIZE / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    max_area = (min(small.shape) / 5.0) ** 2
    passes = []
    for win in (31, 91):
        diff = small - cv2.blur(small, (win, win))
        for pol in (1, -1):                      # 1: dark ink on light, -1: light ink on dark
            binary = ((diff * pol) < -7).astype(np.uint8)
            passes.append((pol,) + _blobs(binary, max_area))
    results, tried = [], []
    count = dict(plausible=0, headers=0, gave_up=False) if count is None else count
    loops, plausible = 0, count["plausible"]
    for limit in _MAX_NEIGHBOURS:
        for pol, pts, area in passes:
            for idx, slots in _find_rings(pts, area, limit):
                # A real ring turns up early (the pass for crowded blobs starts its search
                # where ring 0 would be). If it has not, give up on the frame rather than
                # work through every look-alike loop in a big code. And once a ring 0 has
                # announced more rings, no other loop in the frame can be the answer.
                loops += 1
                if loops > _MAX_LOOPS or count["plausible"] - plausible > _MAX_PLAUSIBLE:
                    count["gave_up"] = True
                if count["headers"] or count["gave_up"]:
                    return results
                r = _read_candidate(gray, pts[idx] / scale, slots, pol, debug, tried, count, area[idx] / scale ** 2)
                if r is not None and all(r["key"] != q["key"] for q in results):
                    results.append(r)
                if results and limit > 1000:         # the slow pass: one code is enough
                    return results
        if results or count["headers"]:
            break
    return results


def scan(image, debug=None):
    """Find and decode codes in a BGR or grey image. Falls back to single colour channels."""
    if image.ndim == 2:
        return scan_gray(image, debug)
    count = dict(plausible=0, headers=0, gave_up=False)
    results = scan_gray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), debug, count)
    if not results:
        for ch in range(3):
            # The single channels are for colour pairs that vanish in grey. If grey showed a
            # ring 0 whose outer rings could not be read, or a crowd of stars with no
            # readable ring among them, they will not do better.
            if count["headers"] or count["gave_up"]:
                break
            results = scan_gray(image[..., ch], debug, count)
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
