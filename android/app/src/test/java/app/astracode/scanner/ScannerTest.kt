package app.astracode.scanner

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** [parseCodes] against bytes laid out the way the native library documents them. */
class ScannerTest {
    private fun header(buffer: ByteBuffer, kind: Int, rings: Int, level: Int, flags: Int, repaired: Int) {
        buffer.put(kind.toByte()).put(rings.toByte()).put(level.toByte()).put(flags.toByte())
        buffer.putShort(repaired.toShort()).putFloat(0.6f)
        for (i in 1..8) buffer.putFloat(i * 10f)
    }

    @Test
    fun unpacksBothKindsOfCode() {
        val buffer = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(3)
        header(buffer, kind = 0, rings = 1, level = 0, flags = 1, repaired = 1)
        buffer.put(5).putLong(4_320_880_580L).put("432088058060".toByteArray())
        header(buffer, kind = 1, rings = 4, level = 2, flags = 2, repaired = 300)
        val text = "https://astra-europa.example/ é".toByteArray()
        buffer.putInt(text.size).put(text)
        header(buffer, kind = 1, rings = 2, level = 0, flags = 0, repaired = 0)
        buffer.putInt(2).put(byteArrayOf(0xFF.toByte(), 0xFE.toByte()))
        val codes = parseCodes(buffer.array().copyOf(buffer.position()))

        assertEquals(3, codes.size)
        val number = codes[0]
        assertEquals(Payload.Number(5, 4_320_880_580L, "432088058060"), number.payload)
        assertEquals(1, number.rings)
        assertTrue(number.mirrored && !number.bent)
        assertEquals(1, number.repaired)
        assertEquals(0.6f, number.lightLevel)
        assertArrayEquals(floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f), number.corners, 0f)

        val message = codes[1].payload as Payload.Message
        assertEquals("https://astra-europa.example/ é", message.text)
        assertEquals(2, message.level)
        assertEquals(4, codes[1].rings)
        assertTrue(codes[1].bent && !codes[1].mirrored)
        assertEquals(300, codes[1].repaired)

        // Bytes that are not text stay bytes.
        val raw = codes[2].payload as Payload.Message
        assertNull(raw.text)
        assertArrayEquals(byteArrayOf(0xFF.toByte(), 0xFE.toByte()), raw.bytes)
    }

    @Test
    fun nothingFoundIsAnEmptyList() {
        assertTrue(parseCodes(null).isEmpty())
        assertTrue(parseCodes(byteArrayOf(0)).isEmpty())
    }
}
