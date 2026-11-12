package app.astracode.scanner

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction

/** What a code carries. */
sealed interface Payload {
    /**
     * A one-ring code: a ten-digit [payload] and a [metadata] value, 0 to 108, saying what it is.
     * [memberId] is the payload followed by its two check digits (payload modulo 97).
     */
    data class Number(val metadata: Int, val payload: Long, val memberId: String) : Payload

    /** A code with several rings: a byte string, written at error-correction [level] 0 to 3. */
    class Message(val level: Int, val bytes: ByteArray) : Payload {
        /** The message as text, or null if it is not valid UTF-8. */
        val text: String? by lazy {
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

        override fun equals(other: Any?) = other is Message && level == other.level && bytes.contentEquals(other.bytes)
        override fun hashCode() = 31 * level + bytes.contentHashCode()
    }
}

/** A code found in a camera frame. */
class ScannedCode(
    val payload: Payload,
    /** Number of rings, 1 to 24. */
    val rings: Int,
    /** Whether the code was seen as a mirror image: through glass, from behind. */
    val mirrored: Boolean,
    /** Whether reading needed each star's own shape: the surface was far from flat. */
    val bent: Boolean,
    /** Stars (one-ring code) or star pairs (message) that the error correction repaired. */
    val repaired: Int,
    /** How inked the light limbs are relative to solid ones, 0 to 1, as measured in this frame. */
    val lightLevel: Float,
    /** The corners of the code's square in frame pixels: x0, y0, x1, y1, ... clockwise from its top left. */
    val corners: FloatArray,
)

/**
 * Unpack what [NativeScanner.scan] returns. The layout is documented with the function that
 * writes it, `encode` in `rust/astracode-android/src/lib.rs`: little-endian, a count, then per
 * code a fixed header, the corners, and the payload.
 */
fun parseCodes(packed: ByteArray?): List<ScannedCode> {
    if (packed == null || packed.isEmpty()) return emptyList()
    val buffer = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN)
    return List(buffer.get().toInt() and 0xFF) {
        val kind = buffer.get().toInt()
        val rings = buffer.get().toInt() and 0xFF
        val level = buffer.get().toInt()
        val flags = buffer.get().toInt()
        val repaired = buffer.short.toInt() and 0xFFFF
        val lightLevel = buffer.float
        val corners = FloatArray(8) { buffer.float }
        val payload = if (kind == 0) {
            val metadata = buffer.get().toInt() and 0xFF
            val payload = buffer.long
            Payload.Number(metadata, payload, ByteArray(12).also(buffer::get).toString(Charsets.US_ASCII))
        } else {
            Payload.Message(level, ByteArray(buffer.int).also(buffer::get))
        }
        ScannedCode(payload, rings, flags and 1 != 0, flags and 2 != 0, repaired, lightLevel, corners)
    }
}

/** Scan a camera frame's brightness plane. See [NativeScanner.scan]. */
fun scanFrame(luma: ByteBuffer, width: Int, height: Int, rowStride: Int, findSize: Int = 0): List<ScannedCode> =
    parseCodes(NativeScanner.scan(luma, width, height, rowStride, findSize))
