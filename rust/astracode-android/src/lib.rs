//! The Astracode scanner for the Android app: one native function, called
//! through JNI from `app.astracode.scanner.NativeScanner`.
//!
//! The app hands over the camera frame's brightness plane and gets back a
//! byte array describing the codes found, laid out as [`pack`] documents.
//! Keeping the interface to one array of bytes means no Java objects are
//! built or looked up from native code; `Scanner.kt` unpacks it.

use std::panic::{catch_unwind, AssertUnwindSafe};

use astracode::image::Gray;
use astracode::packed::pack;
use astracode::scan::{scan_gray, Code, Options};
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jbyteArray, jint};
use jni::JNIEnv;

/// Scan a brightness plane given as `height` rows of `row_stride` bytes, of
/// which the first `width` in each row are pixels.
fn scan(luma: &[u8], width: usize, height: usize, row_stride: usize, find_size: usize) -> Vec<Code> {
    let gray = Gray::from_luma8_rows(width, height, row_stride, luma);
    scan_gray(&gray, &Options { find_size: (find_size > 0).then_some(find_size) })
}

/// `NativeScanner.scan(ByteBuffer luma, int width, int height, int rowStride, int findSize)`.
///
/// `luma` must be a direct buffer holding the frame's Y plane. `findSize` is
/// [`Options::find_size`], or 0 to look for stars at full resolution. Returns
/// the codes found as described at [`pack`], or `null` if the arguments do
/// not describe a frame or something went wrong inside the scanner; the app
/// treats both as "nothing found" and moves on to the next frame.
#[no_mangle]
pub extern "system" fn Java_app_astracode_scanner_NativeScanner_scan<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    luma: JByteBuffer<'local>,
    width: jint,
    height: jint,
    row_stride: jint,
    find_size: jint,
) -> jbyteArray {
    let null = std::ptr::null_mut();
    if width <= 0 || height <= 0 || row_stride < width || find_size < 0 {
        return null;
    }
    let (width, height, row_stride) = (width as usize, height as usize, row_stride as usize);
    let (Ok(address), Ok(capacity)) = (env.get_direct_buffer_address(&luma), env.get_direct_buffer_capacity(&luma)) else {
        return null;
    };
    // The last row need not be padded out to the full stride.
    if address.is_null() || capacity < row_stride * (height - 1) + width {
        return null;
    }
    // SAFETY: `address` points to `capacity` readable bytes of a direct buffer that the
    // caller keeps alive and does not write to for the duration of this call.
    let pixels = unsafe { std::slice::from_raw_parts(address, capacity) };
    // A panic must not cross into the Java runtime.
    let Ok(codes) = catch_unwind(AssertUnwindSafe(|| scan(pixels, width, height, row_stride, find_size as usize))) else {
        return null;
    };
    env.byte_array_from_slice(&pack(&codes)).map(|array| array.into_raw()).unwrap_or(null)
}
