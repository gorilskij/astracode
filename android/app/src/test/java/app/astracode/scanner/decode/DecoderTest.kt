package app.astracode.scanner.decode

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.imageio.ImageIO

class DecoderTest {
    private fun load(file: File): Gray {
        val img = ImageIO.read(file)
        val w = img.width
        val h = img.height
        val rgb = img.getRGB(0, 0, w, h, null, 0, w)
        return Gray(w, h, FloatArray(w * h) {
            val p = rgb[it]
            0.299f * ((p shr 16) and 255) + 0.587f * ((p shr 8) and 255) + 0.114f * (p and 255)
        })
    }

    private class Tally(var frames: Int = 0, var ok: Int = 0, var wrong: Int = 0, var reference: Int = 0,
                        var empty: Int = 0, var falsePositives: Int = 0, var millis: Long = 0)

    /** Scan every frame listed in dir/expected.csv (file,number,reference-result). */
    private fun run(dir: File): Tally {
        val t = Tally()
        for (line in File(dir, "expected.csv").readLines().drop(1)) {
            val (name, number, reference) = line.split(",")
            val gray = load(File(dir, name))
            val start = System.nanoTime()
            val found = Decoder.scan(gray)
            t.millis += (System.nanoTime() - start) / 1_000_000
            if (number.isEmpty()) {
                t.empty++
                if (found.isNotEmpty()) t.falsePositives++
            } else {
                t.frames++
                if (reference == number) t.reference++
                when {
                    found.isEmpty() -> println("missed $name")
                    found.size == 1 && found[0].text == number -> t.ok++
                    else -> { t.wrong++; println("WRONG $name: ${found.map { it.text }} expected $number") }
                }
            }
        }
        println("${dir.name}: decoded ${t.ok}/${t.frames} (reference ${t.reference}), wrong ${t.wrong}, " +
            "false positives ${t.falsePositives}/${t.empty}, ${t.millis / (t.frames + t.empty)} ms per frame")
        return t
    }

    @Test
    fun encodeRoundTrips() {
        for (n in longArrayOf(0, 1, 42, 123456789012, 555555555555, 999999999999)) {
            assertEquals(n, Spec.decodeSymbols(Spec.encode(n)))
        }
        // Same symbols as the Python and JavaScript encoders.
        assertEquals(listOf(4, 6, 8, 2, 0, 15, 15, 11, 6, 12, 5, 12), Spec.encode(123456789012).toList())
        val damaged = Spec.encode(123456789012).also { it[3] = it[3] xor 4 }
        assertEquals(-1, Spec.decodeSymbols(damaged))
    }

    @Test
    fun decodesBundledFrames() {
        val dir = File(javaClass.getResource("/frames/expected.csv")!!.toURI()).parentFile
        val t = run(dir)
        assertEquals(0, t.wrong)
        assertEquals(0, t.falsePositives)
        assertEquals(t.frames, t.ok)
    }

    /** Runs only when ASTRA_FIXTURES points at a set made by tools/export_android.py. */
    @Test
    fun matchesReferenceOnLargeFixtureSet() {
        val path = System.getenv("ASTRA_FIXTURES").orEmpty()
        if (path.isEmpty()) return
        val t = run(File(path))
        assertEquals(0, t.wrong)
        assertEquals(0, t.falsePositives)
        assertTrue("decoded ${t.ok}, reference ${t.reference}", t.ok >= t.reference - t.frames / 50)
    }
}
