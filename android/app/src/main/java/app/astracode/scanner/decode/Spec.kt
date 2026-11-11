package app.astracode.scanner.decode

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Astracode format. Mirrors astracode/spec.py; see that file for the full description.
 *
 * Lengths are in model units: the ring of star centres has radius 100 around the
 * origin, x to the right, y down. Star 0 is at 12 o'clock, then clockwise. Each
 * star's top limb is always present; the other four, clockwise from the upper
 * right, carry bits 3,2,1,0 of its symbol.
 */
object Spec {
    const val R = 100.0
    const val STAR_RO = 19.5
    val STAR_RATIO = sin(PI / 10) / sin(3 * PI / 10)   // regular pentagram
    val STAR_RI = STAR_RO * STAR_RATIO
    const val MARGIN = 16.0
    const val HALF = R + STAR_RO + MARGIN
    const val N_STARS = 12
    const val N_DATA = 10
    const val MAX_NUMBER = 1_000_000_000_000L

    private val MASK = intArrayOf(0x5, 0xA, 0x3, 0xC, 0x9, 0x6, 0xE, 0x1, 0x7, 0x8, 0xB, 0x4)

    private val EXP = IntArray(30)
    private val LOG = IntArray(16)

    init {
        var x = 1
        for (i in 0 until 15) {
            EXP[i] = x
            EXP[i + 15] = x
            LOG[x] = i
            x = x shl 1
            if (x and 16 != 0) x = x xor 0x13   // x^4 + x + 1
        }
    }

    private fun gfMul(a: Int, b: Int) = if (a == 0 || b == 0) 0 else EXP[LOG[a] + LOG[b]]

    /** Number -> the 12 symbols as drawn (10 data, 2 Reed-Solomon check, masked). */
    fun encode(number: Long): IntArray {
        require(number in 0 until MAX_NUMBER) { "number must have at most 12 digits" }
        val cw = IntArray(N_STARS)
        for (i in 0 until N_DATA) cw[i] = ((number shr (4 * (N_DATA - 1 - i))) and 15).toInt()
        var r1 = 0
        var r0 = 0
        for (i in 0 until N_DATA) {
            val f = cw[i] xor r1
            r1 = r0 xor gfMul(f, 3)
            r0 = gfMul(f, 2)
        }
        cw[10] = r1
        cw[11] = r0
        return IntArray(N_STARS) { cw[it] xor MASK[it] }
    }

    /** The 12 symbols as drawn -> number, or -1 if the check fails. */
    fun decodeSymbols(symbols: IntArray): Long {
        var s0 = 0
        var s1 = 0
        var number = 0L
        for (i in 0 until N_STARS) {
            val c = symbols[i] xor MASK[i]
            s0 = s0 xor c
            s1 = s1 xor gfMul(c, EXP[(N_STARS - 1 - i) % 15])
            if (i < N_DATA) number = (number shl 4) or c.toLong()
        }
        return if (s0 == 0 && s1 == 0 && number < MAX_NUMBER) number else -1
    }

    fun starCenterX(i: Int) = R * cos(Math.toRadians(-90.0 + 30 * i))
    fun starCenterY(i: Int) = R * sin(Math.toRadians(-90.0 + 30 * i))

    fun limbPresent(symbol: Int, k: Int) = k == 0 || (symbol shr (4 - k)) and 1 != 0

    /** Outline of a star centred on the origin, as x0, y0, x1, y1, ... */
    fun starPolygon(symbol: Int): DoubleArray {
        val pts = ArrayList<Double>(20)
        for (k in 0 until 5) {
            var a = Math.toRadians(-90.0 - 36 + 72 * k)
            pts += STAR_RI * cos(a)
            pts += STAR_RI * sin(a)
            if (limbPresent(symbol, k)) {
                a = Math.toRadians(-90.0 + 72 * k)
                pts += STAR_RO * cos(a)
                pts += STAR_RO * sin(a)
            }
        }
        return pts.toDoubleArray()
    }
}
