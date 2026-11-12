package app.astracode.scanner.decode

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Astracode format, version 2. Mirrors astracode/spec.py; see that file for the full description.
 *
 * Lengths are in model units: the ring of star centres has radius 100 around the
 * origin, x to the right, y down. Star 0 is at 12 o'clock, then clockwise. Every
 * star is a complete pentagram with a solid centre; each limb is solid (bit 1) or
 * a lighter shade (bit 0). The top limb is bit 4 of the star's symbol, then
 * clockwise bits 3, 2, 1, 0.
 */
object Spec {
    const val R = 100.0
    const val STAR_RO = 19.5
    val STAR_RATIO = sin(PI / 10) / sin(3 * PI / 10)   // regular pentagram
    val STAR_RI = STAR_RO * STAR_RATIO
    const val MARGIN = 16.0
    const val HALF = R + STAR_RO + MARGIN
    const val N_STARS = 12
    const val N_DATA = 8
    const val N_CHECK = N_STARS - N_DATA
    const val MAX_NUMBER = 1_000_000_000_000L

    private val MASK = intArrayOf(0x15, 0x0A, 0x13, 0x1C, 0x09, 0x06, 0x1E, 0x01, 0x17, 0x08, 0x1B, 0x04)

    // GF(32), x^5 + x^2 + 1
    private val EXP = IntArray(62)
    private val LOG = IntArray(32)
    private val GEN: IntArray     // generator polynomial, roots a^0..a^3, highest degree first

    init {
        var x = 1
        for (i in 0 until 31) {
            EXP[i] = x
            EXP[i + 31] = x
            LOG[x] = i
            x = x shl 1
            if (x and 32 != 0) x = x xor 0x25
        }
        var g = intArrayOf(1)
        for (j in 0 until N_CHECK) {
            val next = IntArray(g.size + 1)
            for (i in next.indices) {
                next[i] = (if (i < g.size) g[i] else 0) xor (if (i > 0) gfMul(g[i - 1], EXP[j]) else 0)
            }
            g = next
        }
        GEN = g
    }

    private fun gfMul(a: Int, b: Int) = if (a == 0 || b == 0) 0 else EXP[LOG[a] + LOG[b]]
    private fun gfDiv(a: Int, b: Int) = if (a == 0) 0 else EXP[LOG[a] - LOG[b] + 31]

    /** Number -> the 12 symbols as drawn (8 data, 4 Reed-Solomon check, masked). */
    fun encode(number: Long): IntArray {
        require(number in 0 until MAX_NUMBER) { "number must have at most 12 digits" }
        val cw = IntArray(N_STARS)
        for (i in 0 until N_DATA) cw[i] = ((number shr (5 * (N_DATA - 1 - i))) and 31).toInt()
        val rem = IntArray(N_CHECK)
        for (i in 0 until N_DATA) {
            val f = cw[i] xor rem[0]
            for (j in 0 until N_CHECK) rem[j] = (if (j + 1 < N_CHECK) rem[j + 1] else 0) xor gfMul(f, GEN[j + 1])
        }
        for (j in 0 until N_CHECK) cw[N_DATA + j] = rem[j]
        return IntArray(N_STARS) { cw[it] xor MASK[it] }
    }

    private fun syndromes(cw: IntArray) = IntArray(N_CHECK) { j ->
        var s = 0
        for (i in 0 until N_STARS) s = s xor gfMul(cw[i], EXP[(j * (N_STARS - 1 - i)) % 31])
        s
    }

    /** Repair at most one symbol in place. Returns stars repaired (0 or 1), or -1 if that is not enough. */
    private fun correctOne(cw: IntArray): Int {
        val s = syndromes(cw)
        if (s.all { it == 0 }) return 0
        if (s.any { it == 0 }) return -1
        val x = gfDiv(s[1], s[0])                     // a^position of a single error
        if (gfMul(s[1], x) != s[2] || gfMul(s[2], x) != s[3]) return -1
        val pos = LOG[x]
        if (pos >= N_STARS) return -1
        cw[N_STARS - 1 - pos] = cw[N_STARS - 1 - pos] xor s[0]
        return 1
    }

    /** Treat symbols i1 and i2 as unknown and solve for them in place; false unless the result checks out. */
    private fun fillTwo(cw: IntArray, i1: Int, i2: Int): Boolean {
        val s = syndromes(cw)
        val x1 = EXP[N_STARS - 1 - i1]
        val x2 = EXP[N_STARS - 1 - i2]
        val e2 = gfDiv(s[1] xor gfMul(s[0], x1), x1 xor x2)
        val e1 = s[0] xor e2
        val x1s = gfMul(x1, x1)
        val x2s = gfMul(x2, x2)
        if (gfMul(e1, x1s) xor gfMul(e2, x2s) != s[2]) return false
        if (gfMul(e1, gfMul(x1s, x1)) xor gfMul(e2, gfMul(x2s, x2)) != s[3]) return false
        cw[i1] = cw[i1] xor e1
        cw[i2] = cw[i2] xor e2
        return true
    }

    /**
     * The 12 symbols as drawn -> number, or -1 if the check fails. Repairs one
     * misread star anywhere; if that fails and [doubtful1], [doubtful2] name two
     * stars the reader is unsure of, repairs exactly those two instead.
     */
    fun decodeSymbols(symbols: IntArray, doubtful1: Int = -1, doubtful2: Int = -1): Long {
        val cw = IntArray(N_STARS) { symbols[it] xor MASK[it] }
        if (correctOne(cw) < 0) {
            if (doubtful1 < 0 || !fillTwo(cw, doubtful1, doubtful2)) return -1
        }
        var number = 0L
        for (i in 0 until N_DATA) number = (number shl 5) or cw[i].toLong()
        return if (number < MAX_NUMBER) number else -1
    }

    fun starCenterX(i: Int) = R * cos(Math.toRadians(-90.0 + 30 * i))
    fun starCenterY(i: Int) = R * sin(Math.toRadians(-90.0 + 30 * i))

    fun limbSolid(symbol: Int, k: Int) = (symbol shr (4 - k)) and 1 != 0

    /** Outline of the solid part of a star (centre plus solid limbs) centred on the origin, as x0, y0, x1, y1, ... */
    fun starPolygon(symbol: Int): DoubleArray {
        val pts = ArrayList<Double>(20)
        for (k in 0 until 5) {
            var a = Math.toRadians(-90.0 - 36 + 72 * k)
            pts += STAR_RI * cos(a)
            pts += STAR_RI * sin(a)
            if (limbSolid(symbol, k)) {
                a = Math.toRadians(-90.0 + 72 * k)
                pts += STAR_RO * cos(a)
                pts += STAR_RO * sin(a)
            }
        }
        return pts.toDoubleArray()
    }
}
