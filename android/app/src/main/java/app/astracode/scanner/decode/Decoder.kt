package app.astracode.scanner.decode

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/** A single-channel image, row-major, values 0..255. */
class Gray(val width: Int, val height: Int, val px: FloatArray) {
    companion object {
        /** From an 8-bit luma plane such as the Y plane of a camera frame. */
        fun fromLuma(data: ByteArray, width: Int, height: Int, rowStride: Int = width): Gray {
            val px = FloatArray(width * height)
            for (y in 0 until height) {
                val src = y * rowStride
                val dst = y * width
                for (x in 0 until width) px[dst + x] = (data[src + x].toInt() and 0xFF).toFloat()
            }
            return Gray(width, height, px)
        }
    }
}

/** A decoded code. [corners] is the code square in image pixels: x0, y0, ... clockwise from its top left. */
class ScanResult(val number: Long, val corners: FloatArray, val confidence: Float, val logoScore: Float) {
    val text: String get() = number.toString().padStart(12, '0')
}

/**
 * Astracode scanner (format version 2); a port of astracode/decode.py, which documents
 * the pipeline: local threshold -> blobs -> walk the ring of 12 stars -> homography ->
 * find "up" from the logo and star shapes -> refine against the grey image -> read
 * each limb as light or solid -> Reed-Solomon check and repair.
 */
object Decoder {
    // ---------------------------------------------------------------- sample patterns
    // Point sets are flat arrays x0, y0, x1, y1, ... in model units.

    private val BASE_W = 2 * Spec.STAR_RI * sin(Math.toRadians(36.0))
    private val BASE_R = Spec.STAR_RI * cos(Math.toRadians(36.0))
    private val LIMB_LEN = Spec.STAR_RO - BASE_R

    private val CORE: DoubleArray = run {
        val v = doubleArrayOf(-3.75, -2.25, -0.75, 0.75, 2.25, 3.75)
        val out = ArrayList<Double>()
        for (x in v) for (y in v) if (x * x + y * y <= 16.5) { out += x; out += y }
        out.toDoubleArray()
    }

    private fun limbPoints(k: Int): DoubleArray {
        val a = Math.toRadians(-90.0 + 72 * k)
        val ax = cos(a)
        val ay = sin(a)
        val out = ArrayList<Double>()
        for (r in doubleArrayOf(8.0, 9.5, 11.0, 12.5, 14.0)) {
            val w = BASE_W * (1 - (r - BASE_R) / LIMB_LEN)
            for (t in doubleArrayOf(-0.25, 0.0, 0.25)) { out += ax * r - ay * t * w; out += ay * r + ax * t * w }
        }
        return out.toDoubleArray()
    }

    private fun gapPoints(k: Int): DoubleArray {
        val a = Math.toRadians(-90.0 + 36 + 72 * k)
        val ax = cos(a)
        val ay = sin(a)
        val out = ArrayList<Double>()
        for (r in doubleArrayOf(15.0, 17.0, 19.0)) for (t in doubleArrayOf(-2.0, 0.0, 2.0)) {
            out += ax * r - ay * t; out += ay * r + ax * t
        }
        return out.toDoubleArray()
    }

    private val LIMB = Array(5) { limbPoints(it) }
    private val GAP = Array(5) { gapPoints(it) }   // gap k lies between limbs k and k+1
    private val NC = CORE.size / 2
    private val NL = LIMB[0].size / 2
    private val NG = GAP[0].size / 2
    private val CENTERS = DoubleArray(24) { if (it % 2 == 0) Spec.starCenterX(it / 2) else Spec.starCenterY(it / 2) }

    /** One star's sample points: core | 5 limbs | 5 gaps. */
    private val STAR_PTS: DoubleArray = CORE + LIMB.reduce { a, b -> a + b } + GAP.reduce { a, b -> a + b }
    private val STAR_N = STAR_PTS.size / 2
    private val ALL_STAR_PTS = offsetToStars(STAR_PTS)

    /** Points that are ink or background in every code: all of each star is ink, the gaps between limbs are not. */
    private val FIX_XY = ALL_STAR_PTS
    private val FIX_T = FloatArray(FIX_XY.size / 2) { if (it % STAR_N < NC + 5 * NL) 1f else 0f }
    private val LOGO_XY = DoubleArray(LogoTemplate.X.size * 2) {
        (if (it % 2 == 0) LogoTemplate.X[it / 2] else LogoTemplate.Y[it / 2]).toDouble()
    }

    private fun offsetToStars(pts: DoubleArray): DoubleArray {
        val out = DoubleArray(12 * pts.size)
        for (i in 0 until 12) for (j in pts.indices) out[i * pts.size + j] = pts[j] + CENTERS[2 * i + j % 2]
        return out
    }

    // Dense star template for alignment: a grid over the star's disc, weighted + inside
    // the pentagram and - outside it. Every star has the same outline in this format.
    private val SEARCH: DoubleArray = run {
        val out = ArrayList<Double>()
        for (dx in -4..4) for (dy in -4..4) { out += dx.toDouble(); out += dy.toDouble() }
        out.toDoubleArray()
    }
    private val NS = SEARCH.size / 2
    private val GRID: DoubleArray = run {
        val out = ArrayList<Double>()
        for (ix in 0..28) for (iy in 0..28) {
            val x = -21 + 1.5 * ix
            val y = -21 + 1.5 * iy
            if (x * x + y * y <= 22.0 * 22.0) { out += x; out += y }
        }
        out.toDoubleArray()
    }
    private val NGRID = GRID.size / 2
    private val ALIGN_W: FloatArray = run {
        val poly = Spec.starPolygon(31)
        val ink = BooleanArray(NGRID) { pointInPolygon(poly, GRID[2 * it], GRID[2 * it + 1]) }
        val ni = ink.count { it }
        val no = NGRID - ni
        FloatArray(NGRID) { if (ink[it]) 1f / ni else -1f / no }
    }

    private fun pointInPolygon(poly: DoubleArray, x: Double, y: Double): Boolean {
        var inside = false
        val n = poly.size / 2
        var j = n - 1
        for (i in 0 until n) {
            val xi = poly[2 * i]
            val yi = poly[2 * i + 1]
            val xj = poly[2 * j]
            val yj = poly[2 * j + 1]
            if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside
            j = i
        }
        return inside
    }

    // ---------------------------------------------------------------- geometry

    /** Map model points through homography [h] (row-major 3x3) and bilinearly sample the image there. */
    private fun sample(g: Gray, h: DoubleArray, pts: DoubleArray, out: FloatArray, dx: Double = 0.0, dy: Double = 0.0) {
        val w = g.width
        val px = g.px
        val maxX = w - 1.001
        val maxY = g.height - 1.001
        for (i in 0 until pts.size / 2) {
            val mx = pts[2 * i] + dx
            val my = pts[2 * i + 1] + dy
            val d = h[6] * mx + h[7] * my + h[8]
            var x = (h[0] * mx + h[1] * my + h[2]) / d
            var y = (h[3] * mx + h[4] * my + h[5]) / d
            if (!(x > 0.0)) x = 0.0 else if (x > maxX) x = maxX     // also catches NaN
            if (!(y > 0.0)) y = 0.0 else if (y > maxY) y = maxY
            val x0 = x.toInt()
            val y0 = y.toInt()
            val fx = (x - x0).toFloat()
            val fy = (y - y0).toFloat()
            val p = y0 * w + x0
            val a = px[p] * (1 - fx) + px[p + 1] * fx
            val b = px[p + w] * (1 - fx) + px[p + w + 1] * fx
            out[i] = a * (1 - fy) + b * fy
        }
    }

    private fun applyH(h: DoubleArray, x: Double, y: Double, out: DoubleArray, at: Int) {
        val d = h[6] * x + h[7] * y + h[8]
        out[at] = (h[0] * x + h[1] * y + h[2]) / d
        out[at + 1] = (h[3] * x + h[4] * y + h[5]) / d
    }

    private fun mul3(a: DoubleArray, b: DoubleArray) = DoubleArray(9) {
        val r = it / 3
        val c = it % 3
        a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c]
    }

    private fun inv3(m: DoubleArray): DoubleArray {
        val a = m[4] * m[8] - m[5] * m[7]
        val b = m[5] * m[6] - m[3] * m[8]
        val c = m[3] * m[7] - m[4] * m[6]
        val det = m[0] * a + m[1] * b + m[2] * c
        return doubleArrayOf(
            a / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
            b / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
            c / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det,
        )
    }

    /** Solve the n x n system a x = b in place by Gaussian elimination; false if singular. */
    private fun solve(a: Array<DoubleArray>, b: DoubleArray): Boolean {
        val n = b.size
        for (c in 0 until n) {
            var p = c
            for (r in c + 1 until n) if (abs(a[r][c]) > abs(a[p][c])) p = r
            if (abs(a[p][c]) < 1e-12) return false
            if (p != c) {
                val t = a[p]; a[p] = a[c]; a[c] = t
                val tb = b[p]; b[p] = b[c]; b[c] = tb
            }
            for (r in c + 1 until n) {
                val f = a[r][c] / a[c][c]
                if (f == 0.0) continue
                for (k in c until n) a[r][k] -= f * a[c][k]
                b[r] -= f * b[c]
            }
        }
        for (r in n - 1 downTo 0) {
            var s = b[r]
            for (k in r + 1 until n) s -= a[r][k] * b[k]
            b[r] = s / a[r][r]
        }
        return true
    }

    /** Least-squares homography mapping src -> dst (flat x, y arrays), or null if degenerate. */
    private fun homography(src: DoubleArray, dst: DoubleArray): DoubleArray? {
        val n = src.size / 2
        fun normalise(p: DoubleArray): DoubleArray {   // returns cx, cy, scale
            var cx = 0.0
            var cy = 0.0
            for (i in 0 until n) { cx += p[2 * i]; cy += p[2 * i + 1] }
            cx /= n; cy /= n
            var d = 0.0
            for (i in 0 until n) d += hypot(p[2 * i] - cx, p[2 * i + 1] - cy)
            return doubleArrayOf(cx, cy, sqrt(2.0) / (d / n + 1e-12))
        }
        val ns = normalise(src)
        val nd = normalise(dst)
        // Normal equations for the 8 unknowns with h33 = 1, in normalised coordinates.
        val ata = Array(8) { DoubleArray(8) }
        val atb = DoubleArray(8)
        val row = DoubleArray(8)
        for (i in 0 until n) {
            val x = (src[2 * i] - ns[0]) * ns[2]
            val y = (src[2 * i + 1] - ns[1]) * ns[2]
            val u = (dst[2 * i] - nd[0]) * nd[2]
            val v = (dst[2 * i + 1] - nd[1]) * nd[2]
            for (eq in 0..1) {
                val t = if (eq == 0) u else v
                row.fill(0.0)
                val o = 3 * eq
                row[o] = x; row[o + 1] = y; row[o + 2] = 1.0
                row[6] = -t * x; row[7] = -t * y
                for (r in 0 until 8) {
                    if (row[r] == 0.0) continue
                    for (c in 0 until 8) ata[r][c] += row[r] * row[c]
                    atb[r] += row[r] * t
                }
            }
        }
        if (!solve(ata, atb)) return null
        val hn = doubleArrayOf(atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], atb[6], atb[7], 1.0)
        val ts = doubleArrayOf(ns[2], 0.0, -ns[2] * ns[0], 0.0, ns[2], -ns[2] * ns[1], 0.0, 0.0, 1.0)
        val tdInv = doubleArrayOf(1 / nd[2], 0.0, nd[0], 0.0, 1 / nd[2], nd[1], 0.0, 0.0, 1.0)
        val h = mul3(mul3(tdInv, hn), ts)
        if (abs(h[8]) < 1e-12) return null
        for (i in 0 until 9) h[i] /= h[8]
        return h
    }

    // ---------------------------------------------------------------- image preparation

    /** Area-average downscale by [scale] < 1. */
    private fun shrink(g: Gray, scale: Double): Gray {
        val w = max(1, (g.width * scale).roundToInt())
        val h = max(1, (g.height * scale).roundToInt())
        val out = FloatArray(w * h)
        val xs = IntArray(w + 1) { min(g.width, (it / scale).roundToInt()) }
        val ys = IntArray(h + 1) { min(g.height, (it / scale).roundToInt()) }
        for (y in 0 until h) {
            val y0 = ys[y]
            val y1 = max(ys[y + 1], y0 + 1).coerceAtMost(g.height)
            for (x in 0 until w) {
                val x0 = xs[x]
                val x1 = max(xs[x + 1], x0 + 1).coerceAtMost(g.width)
                var s = 0f
                for (yy in y0 until y1) {
                    val r = yy * g.width
                    for (xx in x0 until x1) s += g.px[r + xx]
                }
                out[y * w + x] = s / ((y1 - y0) * (x1 - x0))
            }
        }
        return Gray(w, h, out)
    }

    /** Gaussian blur, sigma 1. */
    private fun smooth(g: Gray): FloatArray {
        val k = FloatArray(9) { exp(-0.5 * (it - 4) * (it - 4)).toFloat() }
        val ks = k.sum()
        for (i in k.indices) k[i] /= ks
        val w = g.width
        val h = g.height
        val tmp = FloatArray(w * h)
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val r = y * w
            for (x in 0 until w) {
                var s = 0f
                for (i in -4..4) s += k[i + 4] * g.px[r + (x + i).coerceIn(0, w - 1)]
                tmp[r + x] = s
            }
        }
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (i in -4..4) s += k[i + 4] * tmp[(y + i).coerceIn(0, h - 1) * w + x]
            out[y * w + x] = s
        }
        return out
    }

    /** Mean over a win x win window around each pixel, clipped at the image edge. */
    private fun boxMean(px: FloatArray, w: Int, h: Int, win: Int): FloatArray {
        val integral = DoubleArray((w + 1) * (h + 1))
        for (y in 0 until h) {
            var run = 0.0
            for (x in 0 until w) {
                run += px[y * w + x]
                integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + run
            }
        }
        val r = win / 2
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val y0 = max(0, y - r)
            val y1 = min(h, y + r + 1)
            for (x in 0 until w) {
                val x0 = max(0, x - r)
                val x1 = min(w, x + r + 1)
                val s = integral[y1 * (w + 1) + x1] - integral[y0 * (w + 1) + x1] -
                    integral[y1 * (w + 1) + x0] + integral[y0 * (w + 1) + x0]
                out[y * w + x] = (s / ((y1 - y0) * (x1 - x0))).toFloat()
            }
        }
        return out
    }

    private class Blobs(val x: FloatArray, val y: FloatArray, val area: FloatArray) {
        val size get() = x.size
    }

    /** 8-connected components of [mask] (consumed), filtered to plausible star shapes. */
    private fun blobs(mask: BooleanArray, w: Int, h: Int, maxArea: Float): Blobs {
        val xs = ArrayList<Float>()
        val ys = ArrayList<Float>()
        val areas = ArrayList<Float>()
        var stack = IntArray(1024)
        for (start in mask.indices) {
            if (!mask[start]) continue
            var sp = 0
            stack[sp++] = start
            mask[start] = false
            var area = 0
            var sx = 0L
            var sy = 0L
            var x0 = w
            var x1 = 0
            var y0 = h
            var y1 = 0
            while (sp > 0) {
                val p = stack[--sp]
                val px = p % w
                val py = p / w
                area++
                sx += px
                sy += py
                if (px < x0) x0 = px
                if (px > x1) x1 = px
                if (py < y0) y0 = py
                if (py > y1) y1 = py
                for (ny in max(0, py - 1)..min(h - 1, py + 1)) for (nx in max(0, px - 1)..min(w - 1, px + 1)) {
                    val q = ny * w + nx
                    if (mask[q]) {
                        mask[q] = false
                        if (sp == stack.size) stack = stack.copyOf(sp * 2)
                        stack[sp++] = q
                    }
                }
            }
            val bw = (x1 - x0 + 1).toFloat()
            val bh = (y1 - y0 + 1).toFloat()
            if (area >= 9 && area <= maxArea && bw < 3.5f * bh && bh < 3.5f * bw && area > 0.12f * bw * bh) {
                xs += sx.toFloat() / area
                ys += sy.toFloat() / area
                areas += area.toFloat()
            }
        }
        var idx = areas.indices.toList()
        if (idx.size > 3500) idx = idx.sortedByDescending { areas[it] }.take(3500)
        return Blobs(FloatArray(idx.size) { xs[idx[it]] }, FloatArray(idx.size) { ys[idx[it]] },
            FloatArray(idx.size) { areas[idx[it]] })
    }

    // ---------------------------------------------------------------- ring detection
    // Under an affine map, ring star j sits at C + A cos(30 j) + B sin(30 j), which is
    // linear in C, A, B. So the next star is a fixed weighted sum of the last few:
    // exact from three, least squares from four or five. Perspective bends this model
    // over long arcs, so no more than five are used.

    private val PREDICT: Array<DoubleArray> = Array(6) { m ->
        if (m < 3) return@Array DoubleArray(0)
        val a = Array(m + 1) { j ->
            val th = Math.toRadians(30.0 * j)
            doubleArrayOf(1.0, cos(th), sin(th))
        }
        val ata = DoubleArray(9)
        for (j in 0 until m) for (r in 0..2) for (c in 0..2) ata[3 * r + c] += a[j][r] * a[j][c]
        val inv = inv3(ata)
        // weights = a[m] . (A^T A)^-1 . A^T
        val t = DoubleArray(3) { c -> a[m][0] * inv[c] + a[m][1] * inv[3 + c] + a[m][2] * inv[6 + c] }
        DoubleArray(m) { j -> t[0] * a[j][0] + t[1] * a[j][1] + t[2] * a[j][2] }
    }
    private val PREDICT_TOL = doubleArrayOf(0.0, 0.0, 0.0, 0.9, 0.6, 0.55)   // allowed miss / last step
    private const val WALK_BUDGET = 300_000
    private const val MAX_MISSING = 3

    // A star has few similar blobs around it; blobs in dense regular texture (screen
    // pixel grids, halftone, fabric) have many and would support endless false rings.
    // Scan with a strict limit first, and only if that finds nothing with a lenient one.
    private val MAX_NEIGHBOURS = intArrayOf(12, 24)

    private class Ring(val blobs: IntArray, val slots: IntArray)

    private class RingFinder(private val b: Blobs, maxNeighbours: Int) {
        private val n = b.size
        private val s = FloatArray(n) { sqrt(b.area[it]) }
        private val nbrs: Array<IntArray>
        private val far: Array<IntArray>
        private val crowded: BooleanArray
        private val ringOf = Array(n) { HashSet<Int>() }
        private val seen = ArrayList<Set<Int>>()
        private val rings = ArrayList<Ring>()
        private var budget = WALK_BUDGET
        private val ids = IntArray(14)
        private val px = DoubleArray(14)
        private val py = DoubleArray(14)

        init {
            val near = Array(n) { ArrayList<Int>() }
            val wide = Array(n) { ArrayList<Int>() }
            for (i in 0 until n) for (j in i + 1 until n) {
                val smax = max(s[i], s[j])
                val smin = min(s[i], s[j])
                if (smax >= 2.3f * smin) continue
                val dx = b.x[i] - b.x[j]
                val dy = b.y[i] - b.y[j]
                val d2 = dx * dx + dy * dy
                if (d2 <= 0.64f * smax * smax || d2 >= 169f * smax * smax) continue
                wide[i] += j; wide[j] += i
                if (d2 < 42.25f * smax * smax) { near[i] += j; near[j] += i }
            }
            nbrs = Array(n) { near[it].toIntArray() }
            far = Array(n) { wide[it].toIntArray() }
            crowded = BooleanArray(n) { nbrs[it].size > maxNeighbours }
        }

        fun find(): List<Ring> {
            if (n < 12 - MAX_MISSING) return rings
            val order = (0 until n).sortedByDescending { s[it] }   // real stars before fine texture
            for (bi in order) {
                if (nbrs[bi].size < 2 || crowded[bi] || budget < 0) continue
                val bx = b.x[bi].toDouble()
                val by = b.y[bi].toDouble()
                for (ai in nbrs[bi]) {
                    if (crowded[ai]) continue
                    val ax = b.x[ai].toDouble()
                    val ay = b.y[ai].toDouble()
                    val l1 = hypot(bx - ax, by - ay)
                    for (ci in nbrs[bi]) {
                        if (ci == ai || crowded[ci]) continue
                        if (ringOf[ai].any { it in ringOf[bi] && it in ringOf[ci] }) continue
                        val cx = b.x[ci].toDouble()
                        val cy = b.y[ci].toDouble()
                        val ratio = hypot(cx - bx, cy - by) / l1
                        if (ratio <= 0.55 || ratio >= 1.8) continue
                        val cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
                        val dot = (bx - ax) * (cx - bx) + (by - ay) * (cy - by)
                        // Clockwise in image coordinates only, turning 5 to 85 degrees.
                        if (cross <= 0 || dot <= 0 || cross < 0.087 * dot) continue
                        ids[0] = ai; ids[1] = bi; ids[2] = ci
                        px[0] = ax; py[0] = ay; px[1] = bx; py[1] = by; px[2] = cx; py[2] = cy
                        walk(3, 0, (s[ai] + s[bi] + s[ci]) / 3)
                    }
                }
            }
            return rings
        }

        /** Extend the partial ring held in ids/px/py up to [step]; -1 marks a missing star. */
        private fun walk(step: Int, missing: Int, size: Float) {
            if (--budget < 0) return
            val m = min(step, 5)
            val w = PREDICT[m]
            var x = 0.0
            var y = 0.0
            for (i in 0 until m) { x += w[i] * px[step - m + i]; y += w[i] * py[step - m + i] }
            val tol = PREDICT_TOL[m] * hypot(px[step - 1] - px[step - 2], py[step - 1] - py[step - 2])
            if (step >= 12) {                       // must close onto the first two stars
                val q = step - 12
                if (hypot(px[q] - x, py[q] - y) <= tol) {
                    if (step == 13) emit()
                    else {
                        ids[step] = ids[0]; px[step] = px[q]; py[step] = py[q]
                        walk(step + 1, missing, size)
                    }
                }
                return
            }
            // Candidates come from the neighbours of the previous star, or after a gap
            // from the wider neighbourhood of the one before it. Closest first.
            val pool = if (ids[step - 1] >= 0) nbrs[ids[step - 1]] else far[ids[step - 2]]
            val limit = if (step <= 4) 3 else 1     // the early predictions are loose: try several
            val found = IntArray(limit) { -1 }
            val dist = DoubleArray(limit) { tol * tol }
            for (i in pool) {
                val dx = b.x[i] - x
                val dy = b.y[i] - y
                val dd = dx * dx + dy * dy
                if (dd >= dist[limit - 1] || s[i] <= size / 2.3f || s[i] >= size * 2.3f) continue
                var used = false
                for (k in 0 until step) if (ids[k] == i) { used = true; break }
                if (used) continue
                var at = limit - 1
                while (at > 0 && dist[at - 1] > dd) { dist[at] = dist[at - 1]; found[at] = found[at - 1]; at-- }
                dist[at] = dd; found[at] = i
            }
            for (i in found) {
                if (i < 0) break
                ids[step] = i; px[step] = b.x[i].toDouble(); py[step] = b.y[i].toDouble()
                walk(step + 1, missing, size)
            }
            // The 4th star must be real, and no two may be missing in a row.
            if (found[0] < 0 && step > 3 && ids[step - 1] >= 0 && missing < MAX_MISSING) {
                ids[step] = -1; px[step] = x; py[step] = y
                walk(step + 1, missing + 1, size)
            }
        }

        private fun emit() {
            val slots = (0 until 12).filter { ids[it] >= 0 }
            val real = slots.map { ids[it] }
            val key = real.toSet()
            if (seen.any { it.containsAll(key) }) return     // same ring, or part of one already found
            seen += key
            for (i in real) ringOf[i] += seen.size
            rings += Ring(real.toIntArray(), slots.toIntArray())
        }
    }

    // ---------------------------------------------------------------- reading

    private fun ncc(a: FloatArray, b: FloatArray): Double {
        var ma = 0.0
        var mb = 0.0
        for (i in a.indices) { ma += a[i]; mb += b[i] }
        ma /= a.size; mb /= a.size
        var sab = 0.0
        var saa = 0.0
        var sbb = 0.0
        for (i in a.indices) {
            val da = a[i] - ma
            val db = b[i] - mb
            sab += da * db; saa += da * da; sbb += db * db
        }
        return sab / (sqrt(saa * sbb) + 1e-9)
    }

    private val fixBuf = FloatArray(FIX_T.size)
    private val logoBuf = FloatArray(LogoTemplate.INK.size)
    private val starBuf = FloatArray(12 * STAR_N)
    private val gridBuf = FloatArray(NGRID)

    /** Re-locate every star by template match and refit the homography. */
    private fun refine(g: Gray, h: DoubleArray, pol: Int): DoubleArray? {
        val found = DoubleArray(24)
        val score = DoubleArray(NS)
        val centre = DoubleArray(2)
        for (i in 0 until 12) {
            var best = 0
            for (o in 0 until NS) {
                sample(g, h, GRID, gridBuf, CENTERS[2 * i] + SEARCH[2 * o], CENTERS[2 * i + 1] + SEARCH[2 * o + 1])
                var sc = 0.0
                for (k in 0 until NGRID) sc += gridBuf[k] * ALIGN_W[k]
                score[o] = -pol * sc
                if (score[o] > score[best]) best = o
            }
            // Sub-step position: score-weighted mean of the offsets around the peak.
            var lo = Double.MAX_VALUE
            for (o in 0 until NS) {
                if (abs(SEARCH[2 * o] - SEARCH[2 * best]) <= 1.01 && abs(SEARCH[2 * o + 1] - SEARCH[2 * best + 1]) <= 1.01) {
                    lo = min(lo, score[o])
                }
            }
            var sw = 0.0
            var ox = 0.0
            var oy = 0.0
            for (o in 0 until NS) {
                if (abs(SEARCH[2 * o] - SEARCH[2 * best]) <= 1.01 && abs(SEARCH[2 * o + 1] - SEARCH[2 * best + 1]) <= 1.01) {
                    val wt = score[o] - lo + 1e-6
                    sw += wt; ox += wt * SEARCH[2 * o]; oy += wt * SEARCH[2 * o + 1]
                }
            }
            applyH(h, CENTERS[2 * i] + ox / sw, CENTERS[2 * i + 1] + oy / sw, centre, 0)
            found[2 * i] = centre[0]
            found[2 * i + 1] = centre[1]
        }
        return homography(CENTERS, found)
    }

    /**
     * How inked each of the 60 limbs is (index star * 5 + limb): 0 is background, about 1
     * is solid. Each limb is compared with the background on its own two sides, so a
     * shadow edge or light gradient across a star cancels; the star's always-solid
     * centre sets the scale. Also returns the lowest core contrast in [minContrast].
     */
    private fun limbStrength(g: Gray, h: DoubleArray, pol: Int, minContrast: FloatArray): DoubleArray {
        sample(g, h, ALL_STAR_PTS, starBuf)
        val out = DoubleArray(60)
        val limb = DoubleArray(5)
        val gap = DoubleArray(5)
        val eps = 8.0
        minContrast[0] = Float.MAX_VALUE
        for (i in 0 until 12) {
            val o = i * STAR_N
            var core = 0.0
            for (k in 0 until NC) core += starBuf[o + k]
            core /= NC
            for (j in 0 until 5) {
                var a = 0.0
                for (k in 0 until NL) a += starBuf[o + NC + j * NL + k]
                limb[j] = a / NL
                var c = 0.0
                for (k in 0 until NG) c += starBuf[o + NC + 5 * NL + j * NG + k]
                gap[j] = c / NG
            }
            val bg = gap.sorted()[2]
            val den = max(pol * (bg - core) / (bg + eps), 1e-3)
            for (j in 0 until 5) {
                val flank = 0.5 * (gap[j] + gap[(j + 4) % 5])
                out[i * 5 + j] = pol * (flank - limb[j]) / (flank + eps) / den
            }
            minContrast[0] = min(minContrast[0], (-pol * (core - bg)).toFloat())
        }
        return out
    }

    private fun percentile(sorted: DoubleArray, p: Double): Double {
        val pos = p / 100 * (sorted.size - 1)
        val i = pos.toInt().coerceAtMost(sorted.size - 2).coerceAtLeast(0)
        return if (sorted.size == 1) sorted[0] else sorted[i] + (sorted[i + 1] - sorted[i]) * (pos - i)
    }

    private class Limbs(val solid: BooleanArray, val conf: DoubleArray, val ratio: Double, val separation: Double)

    private const val STAR_RIDGE = 4.0

    /**
     * Decide light or solid for all 60 limbs. Blur, motion and lighting weaken limbs
     * unevenly: by limb direction (motion blur), smoothly across the code (perspective,
     * focus), and star by star (glare, reflections). So rather than one threshold, fit
     *   log strength = a[limb direction] + plane(star position) + d[star] + light * log(ratio)
     * alternating with the light/solid assignment. The per-star offsets d are pulled
     * towards zero, which is what decides a star whose limbs are all one shade.
     */
    private fun classify(strength: DoubleArray): Limbs? {
        val med = strength.sorted().let { 0.5 * (it[29] + it[30]) }
        // A limb washed out by glare or covered reads as hardly inked: leave it out of the fit.
        val usable = BooleanArray(60) { strength[it] > 0.3 * med }
        val v = DoubleArray(60) { ln(max(strength[it], 1e-3)) }
        val vs = (0 until 60).filter { usable[it] }.map { v[it] }.sorted().toDoubleArray()
        if (vs.size < 30) return null
        var lo = percentile(vs, 10.0)
        var hi = percentile(vs, 90.0)
        for (iter in 0 until 20) {                  // 1-D 2-means
            val thr = 0.5 * (lo + hi)
            var sl = 0.0; var nl = 0; var sh = 0.0; var nh = 0
            for (x in vs) if (x > thr) { sh += x; nh++ } else { sl += x; nl++ }
            if (nl == 0 || nh == 0) break
            val nlo = sl / nl
            val nhi = sh / nh
            val done = abs(nlo - lo) + abs(nhi - hi) < 1e-5
            lo = nlo; hi = nhi
            if (done) break
        }
        val light = BooleanArray(60) { v[it] < 0.5 * (lo + hi) }
        // Columns: 5 limb directions, 2 plane slopes, 12 star offsets, 1 light flag.
        val n = 20
        val row = DoubleArray(n)
        val resid = DoubleArray(60)
        var logRatio = -0.05
        for (iter in 0 until 10) {
            val ata = Array(n) { DoubleArray(n) }
            val atb = DoubleArray(n)
            for (r in 0 until 60) {
                if (!usable[r]) continue
                fillRow(row, r, light[r])
                for (a in 0 until n) {
                    if (row[a] == 0.0) continue
                    for (b in 0 until n) ata[a][b] += row[a] * row[b]
                    atb[a] += row[a] * v[r]
                }
            }
            for (i in 0 until 12) ata[7 + i][7 + i] += STAR_RIDGE
            for (a in 0 until n) ata[a][a] += 1e-9
            if (!solve(ata, atb)) return null
            logRatio = min(atb[19], -0.05)
            var changed = false
            for (r in 0 until 60) {
                fillRow(row, r, false)
                var fit = 0.0
                for (a in 0 until 19) fit += row[a] * atb[a]
                resid[r] = v[r] - fit
                val nowLight = resid[r] < 0.5 * logRatio
                if (nowLight != light[r]) { light[r] = nowLight; changed = true }
            }
            if (!changed) break
        }
        val conf = DoubleArray(60) {
            if (usable[it]) (abs(resid[it] - 0.5 * logRatio) / (0.5 * abs(logRatio))).coerceIn(0.0, 1.0) else 0.0
        }
        var se = 0.0; var se2 = 0.0; var ne = 0
        for (r in 0 until 60) {
            if (!usable[r]) continue
            val e = if (light[r]) resid[r] - logRatio else resid[r]
            se += e; se2 += e * e; ne++
        }
        val sd = sqrt(max(se2 / ne - (se / ne) * (se / ne), 0.0))
        return Limbs(BooleanArray(60) { !light[it] }, conf, exp(logRatio), abs(logRatio) / (sd + 1e-6))
    }

    private fun fillRow(row: DoubleArray, r: Int, light: Boolean) {
        row.fill(0.0)
        val star = r / 5
        row[r % 5] = 1.0
        row[5] = CENTERS[2 * star] / Spec.R
        row[6] = CENTERS[2 * star + 1] / Spec.R
        row[7 + star] = 1.0
        if (light) row[19] = 1.0
    }

    // The check stars repair one misread star anywhere, or else the two least certain
    // stars. Refuse to answer when too many stars have a limb close to the light/solid
    // threshold, or when the two shades are not clearly apart: that is when several
    // stars can be wrong at once and a repair would be a guess.
    private const val WEAK_CONFIDENCE = 0.25
    private const val MAX_WEAK_STARS = 3
    private const val CLEAR_CONFIDENCE = 0.4
    private const val MIN_SHADE_SEPARATION = 4.0

    private fun readCandidate(g: Gray, b: Blobs, ring: Ring, scale: Double, pol: Int): ScanResult? {
        val n = ring.blobs.size
        val model = DoubleArray(2 * n)
        val centres = DoubleArray(2 * n)
        for (i in 0 until n) {
            val a = Math.toRadians(30.0 * ring.slots[i])
            model[2 * i] = Spec.R * cos(a)
            model[2 * i + 1] = Spec.R * sin(a)
            centres[2 * i] = b.x[ring.blobs[i]] / scale
            centres[2 * i + 1] = b.y[ring.blobs[i]] / scale
        }
        val h0 = homography(model, centres) ?: return null
        // Blob centres sit on the star centres; a poor fit is not a ring.
        val inv = inv3(h0)
        val back = DoubleArray(2)
        for (i in 0 until n) {
            applyH(inv, centres[2 * i], centres[2 * i + 1], back, 0)
            if (!(hypot(back[0] - model[2 * i], back[1] - model[2 * i + 1]) <= 9.0)) return null
        }
        // Which of the 12 ring rotations is upright?
        var h: DoubleArray? = null
        var bestScore = -Double.MAX_VALUE
        var starScore = 0.0
        var logoScore = 0.0
        for (k in 0 until 12) {
            val a = Math.toRadians(30.0 * k)
            val hk = mul3(h0, doubleArrayOf(cos(a), -sin(a), 0.0, sin(a), cos(a), 0.0, 0.0, 0.0, 1.0))
            sample(g, hk, FIX_XY, fixBuf)
            sample(g, hk, LOGO_XY, logoBuf)
            val ss = -pol * ncc(fixBuf, FIX_T)
            val ls = -pol * ncc(logoBuf, LogoTemplate.INK)
            if (ss + ls > bestScore) { bestScore = ss + ls; h = hk; starScore = ss; logoScore = ls }
        }
        if (h == null || starScore < 0.3 || logoScore < 0.2) return null
        repeat(3) { h = refine(g, h!!, pol) ?: return null }
        val hf = h!!
        val minContrast = FloatArray(1)
        val limbs = classify(limbStrength(g, hf, pol, minContrast)) ?: return null
        sample(g, hf, LOGO_XY, logoBuf)
        logoScore = -pol * ncc(logoBuf, LogoTemplate.INK)
        if (logoScore < 0.45 || minContrast[0] < 6f || limbs.ratio <= 0.3 || limbs.ratio >= 0.97) return null
        if (limbs.separation < MIN_SHADE_SEPARATION) return null
        val starConf = DoubleArray(12) { i -> (0 until 5).minOf { limbs.conf[i * 5 + it] } }
        if (starConf.count { it < WEAK_CONFIDENCE } > MAX_WEAK_STARS) return null
        val symbols = IntArray(12) { i -> (0 until 5).fold(0) { acc, k -> (acc shl 1) or (if (limbs.solid[i * 5 + k]) 1 else 0) } }
        // Rebuilding the two least certain stars is only safe when every other star is clear.
        val order = (0 until 12).sortedBy { starConf[it] }
        val canFill = starConf[order[2]] >= CLEAR_CONFIDENCE
        val number = Spec.decodeSymbols(symbols, if (canFill) order[0] else -1, if (canFill) order[1] else -1)
        if (number < 0) return null
        val corners = DoubleArray(8)
        val hs = Spec.HALF
        applyH(hf, -hs, -hs, corners, 0)
        applyH(hf, hs, -hs, corners, 2)
        applyH(hf, hs, hs, corners, 4)
        applyH(hf, -hs, hs, corners, 6)
        return ScanResult(number, FloatArray(8) { corners[it].toFloat() }, starConf.min().toFloat(), logoScore.toFloat())
    }

    /** Find and decode every code in the image. Not thread-safe: call from one thread at a time. */
    @Synchronized
    fun scan(gray: Gray): List<ScanResult> {
        val scale = min(1.0, 1280.0 / max(gray.width, gray.height))
        val small = if (scale < 1) shrink(gray, scale) else gray
        val w = small.width
        val h = small.height
        val smoothed = smooth(small)
        val maxArea = (min(w, h) / 5f).let { it * it }
        class Pass(val pol: Int, val blobs: Blobs)
        val passes = ArrayList<Pass>()
        for (win in intArrayOf(31, 91)) {
            val mean = boxMean(smoothed, w, h, win)
            for (pol in intArrayOf(1, -1)) {      // 1: dark ink on light, -1: light ink on dark
                val mask = BooleanArray(w * h) { (smoothed[it] - mean[it]) * pol < -7f }
                passes += Pass(pol, blobs(mask, w, h, maxArea))
            }
        }
        val results = ArrayList<ScanResult>()
        for (limit in MAX_NEIGHBOURS) {
            for (p in passes) {
                for (ring in RingFinder(p.blobs, limit).find()) {
                    val r = readCandidate(gray, p.blobs, ring, scale, p.pol) ?: continue
                    if (results.none { it.number == r.number }) results += r
                }
            }
            if (results.isNotEmpty()) break
        }
        return results
    }
}
