package app.astracode.scanner

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction

/**
 * An Astra number: a ten-digit [payload] and a [metadata] value, 0 to 26, saying what it is.
 * [memberId] is the payload followed by its two check digits (payload modulo 97).
 *
 * This is Astra's own use of the code, not part of the format: a payload of exactly 38 bits,
 * which always fills one ring. The native library decides whether a payload is one.
 */
data class AstraNumber(val metadata: Int, val payload: Long, val memberId: String)

/**
 * What a code carries: a string of [bitLength] bits, usually whole bytes. The format gives
 * them no meaning; [text], [bytes] and [astra] are the readings this app knows.
 */
class Payload(
    val bitLength: Int,
    /** The bits packed into bytes, first bit highest, the last byte padded with zero bits. */
    private val packed: ByteArray,
    /** The payload read as an Astra number, if it is one. */
    val astra: AstraNumber?,
) {
    /** The payload as bytes, or null if it is not a whole number of them. */
    val bytes: ByteArray? get() = if (bitLength % 8 == 0) packed else null

    /** The payload as text, or null if it is not whole bytes of valid UTF-8. */
    val text: String? by lazy {
        val bytes = bytes ?: return@lazy null
        try {
            Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (_: CharacterCodingException) {
            null
        }
    }

    /** The bits written out as 0s and 1s. */
    val bits: String
        get() = String(CharArray(bitLength) { i -> if (packed[i / 8].toInt() and (0x80 shr (i % 8)) != 0) '1' else '0' })

    override fun equals(other: Any?) = other is Payload && bitLength == other.bitLength && packed.contentEquals(other.packed)
    override fun hashCode() = 31 * bitLength + packed.contentHashCode()
}

/** A code found in a camera frame. */
class ScannedCode(
    val payload: Payload,
    /** Number of rings, 1 to 24. */
    val rings: Int,
    /** Error-correction level of the rings beyond the first, 0 to 3; 0 for a one-ring code, which has none. */
    val level: Int,
    /** Whether the code was seen as a mirror image: through glass, from behind. */
    val mirrored: Boolean,
    /** Whether reading needed each star's own shape: the surface was far from flat. */
    val bent: Boolean,
    /** What the error correction repaired: stars of a one-ring code, star pairs of a bigger one. */
    val repaired: Int,
    /** How inked the light limbs are relative to solid ones, 0 to 1, as measured in this frame. */
    val lightLevel: Float,
    /** The corners of the code's square in frame pixels: x0, y0, x1, y1, ... clockwise from its top left. */
    val corners: FloatArray,
)

/**
 * Unpack what [NativeScanner.scan] returns. The layout is documented with the function that
 * writes it, `pack` in `rust/astracode/src/packed.rs`: little-endian, a count, then per
 * code a fixed header, the corners, the payload, and the Astra number if the payload is one.
 */
fun parseCodes(packed: ByteArray?): List<ScannedCode> {
    if (packed == null || packed.isEmpty()) return emptyList()
    val buffer = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN)
    return List(buffer.get().toInt() and 0xFF) {
        val rings = buffer.get().toInt() and 0xFF
        val level = buffer.get().toInt()
        val flags = buffer.get().toInt()
        val repaired = buffer.short.toInt() and 0xFFFF
        val lightLevel = buffer.float
        val corners = FloatArray(8) { buffer.float }
        val bitLength = buffer.int
        val bits = ByteArray((bitLength + 7) / 8).also(buffer::get)
        val astra = if (buffer.get().toInt() == 1) {
            val metadata = buffer.get().toInt() and 0xFF
            val payload = buffer.long
            AstraNumber(metadata, payload, ByteArray(12).also(buffer::get).toString(Charsets.US_ASCII))
        } else {
            null
        }
        ScannedCode(Payload(bitLength, bits, astra), rings, level, flags and 1 != 0, flags and 2 != 0, repaired, lightLevel, corners)
    }
}

/** Scan a camera frame's brightness plane. See [NativeScanner.scan]. */
fun scanFrame(luma: ByteBuffer, width: Int, height: Int, rowStride: Int, findSize: Int = 0): List<ScannedCode> =
    parseCodes(NativeScanner.scan(luma, width, height, rowStride, findSize))
