package app.astracode.scanner

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** [parseCodes] against bytes laid out the way the native library documents them. */
class ScannerTest {
    private fun header(buffer: ByteBuffer, rings: Int, level: Int, flags: Int, repaired: Int) {
        buffer.put(rings.toByte()).put(level.toByte()).put(flags.toByte())
        buffer.putShort(repaired.toShort()).putFloat(0.6f)
        for (i in 1..8) buffer.putFloat(i * 10f)
    }

    @Test
    fun unpacksEveryKindOfPayload() {
        val buffer = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(4)
        // An Astra number: 38 bits holding 5 * 10^10 + 4320880580, then what they stand for.
        header(buffer, rings = 1, level = 0, flags = 1, repaired = 1)
        val value = 54_320_880_580L shl 2
        buffer.putInt(38).put(ByteArray(5) { i -> (value shr (8 * (4 - i))).toByte() })
        buffer.put(1).put(5).putLong(4_320_880_580L).put("432088058060".toByteArray())
        // Text in several rings.
        header(buffer, rings = 4, level = 2, flags = 2, repaired = 300)
        val text = "https://astra-europa.example/ é".toByteArray()
        buffer.putInt(8 * text.size).put(text).put(0)
        // Bytes that are not text.
        header(buffer, rings = 1, level = 0, flags = 0, repaired = 0)
        buffer.putInt(16).put(byteArrayOf(0xFF.toByte(), 0xFE.toByte())).put(0)
        // Bits that are not bytes.
        header(buffer, rings = 1, level = 0, flags = 0, repaired = 0)
        buffer.putInt(11).put(byteArrayOf(0b1010_0000.toByte(), 0b0110_0000)).put(0)
        val codes = parseCodes(buffer.array().copyOf(buffer.position()))

        assertEquals(4, codes.size)
        val number = codes[0]
        assertEquals(AstraNumber(5, 4_320_880_580L, "432088058060"), number.payload.astra)
        assertEquals(38, number.payload.bitLength)
        assertNull(number.payload.bytes)
        assertEquals(1, number.rings)
        assertTrue(number.mirrored && !number.bent)
        assertEquals(1, number.repaired)
        assertEquals(0.6f, number.lightLevel)
        assertArrayEquals(floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f), number.corners, 0f)

        val message = codes[1]
        assertEquals("https://astra-europa.example/ é", message.payload.text)
        assertNull(message.payload.astra)
        assertEquals(2, message.level)
        assertEquals(4, message.rings)
        assertTrue(message.bent && !message.mirrored)
        assertEquals(300, message.repaired)

        // Bytes that are not text stay bytes.
        assertNull(codes[2].payload.text)
        assertArrayEquals(byteArrayOf(0xFF.toByte(), 0xFE.toByte()), codes[2].payload.bytes)

        // Bits that are not bytes stay bits.
        assertNull(codes[3].payload.bytes)
        assertNull(codes[3].payload.text)
        assertEquals("10100000011", codes[3].payload.bits)

        assertEquals(codes[2].payload, codes[2].payload)
        assertNotEquals(codes[2].payload, codes[3].payload)
    }

    @Test
    fun nothingFoundIsAnEmptyList() {
        assertTrue(parseCodes(null).isEmpty())
        assertTrue(parseCodes(byteArrayOf(0)).isEmpty())
    }
}
