package app.astracode.scanner

import java.nio.ByteBuffer

/**
 * The scanner itself, which is the Rust library in `rust/astracode`, built for Android by
 * `rust/astracode-android` (see `build-rust.sh`).
 */
internal object NativeScanner {
    init {
        System.loadLibrary("astracode_android")
    }

    /**
     * Find and read the codes in a camera frame.
     *
     * @param luma the frame's brightness (Y) plane, as a direct buffer
     * @param rowStride bytes per row of [luma], of which the first [width] are pixels
     * @param findSize look for stars in a copy of the frame shrunk to this many pixels on its
     *   longer side, which is quicker but misses small stars; 0 looks at full resolution
     * @return the codes found, packed as [parseCodes] expects, or null if nothing could be scanned
     */
    @JvmStatic
    external fun scan(luma: ByteBuffer, width: Int, height: Int, rowStride: Int, findSize: Int): ByteArray?
}
