//! The Astracode scanner for the Android app: one native function, called
//! through JNI from `app.astracode.scanner.NativeScanner`.
//!
//! The app hands over the camera frame's brightness plane and gets back a
//! byte array describing the codes found, laid out as [`encode`] documents.
//! Keeping the interface to one array of bytes means no Java objects are
//! built or looked up from native code; `Scanner.kt` unpacks it.

use std::panic::{catch_unwind, AssertUnwindSafe};

use astracode::format::ring0;
use astracode::image::Gray;
use astracode::scan::{scan_gray, Code, Options, Payload};
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jbyteArray, jint};
use jni::JNIEnv;

/// The codes found, as bytes (integers and floats little-endian):
///
/// ```text
/// u8   number of codes
/// then, for each code:
///   u8   kind: 0 = one-ring code, 1 = message
///   u8   rings
///   u8   error-correction level (0 for a one-ring code)
///   u8   flags: bit 0 = seen as a mirror image, bit 1 = read from the star shapes (bent surface)
///   u16  stars or star pairs repaired
///   f32  how inked the light limbs are relative to solid, 0 to 1
///   f32  x 8: the corners of the code's square, x then y, in frame pixels:
///             top left, top right, bottom right, bottom left
///   kind 0:  u8 metadata, u64 payload, 12 ASCII digits of member ID
///   kind 1:  u32 length, then the message bytes
/// ```
fn encode(codes: &[Code]) -> Vec<u8> {
    let mut out = vec![codes.len().min(255) as u8];
    for code in codes.iter().take(255) {
        let (kind, level) = match &code.payload {
            Payload::Number(_) => (0u8, 0u8),
            Payload::Message { level, .. } => (1, *level),
        };
        out.extend([kind, code.rings as u8, level, code.mirrored as u8 | (code.shaped as u8) << 1]);
        out.extend((code.repaired.min(u16::MAX as usize) as u16).to_le_bytes());
        out.extend((code.light_level as f32).to_le_bytes());
        for corner in code.corners {
            out.extend((corner[0] as f32).to_le_bytes());
            out.extend((corner[1] as f32).to_le_bytes());
        }
        match &code.payload {
            Payload::Number(value) => {
                let (metadata, payload) = ring0::unpack(*value);
                out.push(metadata as u8);
                out.extend(payload.to_le_bytes());
                out.extend(ring0::member_id(payload).as_bytes());
            }
            Payload::Message { bytes, .. } => {
                out.extend((bytes.len() as u32).to_le_bytes());
                out.extend(bytes);
            }
        }
    }
    out
}

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
/// the codes found as described at [`encode`], or `null` if the arguments do
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
    env.byte_array_from_slice(&encode(&codes)).map(|array| array.into_raw()).unwrap_or(null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_are_laid_out_as_documented() {
        let corners = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let base = Code {
            payload: Payload::Number(ring0::pack(5, 4_320_880_580).unwrap()),
            rings: 1,
            corners,
            mirrored: true,
            repaired: 1,
            shaped: false,
            confidence: 1.0,
            separation: 9.0,
            light_level: 0.5,
        };
        let message =
            Code { payload: Payload::Message { level: 2, bytes: b"hi".to_vec() }, rings: 3, mirrored: false, shaped: true, ..base.clone() };
        let bytes = encode(&[base, message]);
        assert_eq!(bytes[0], 2);
        // First code: kind, rings, level, flags, repaired.
        assert_eq!(&bytes[1..7], &[0, 1, 0, 1, 1, 0]);
        assert_eq!(f32::from_le_bytes(bytes[7..11].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(bytes[11..15].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[39..43].try_into().unwrap()), 8.0);
        assert_eq!(bytes[43], 5);
        assert_eq!(u64::from_le_bytes(bytes[44..52].try_into().unwrap()), 4_320_880_580);
        assert_eq!(&bytes[52..64], b"432088058060");
        // Second code.
        assert_eq!(&bytes[64..70], &[1, 3, 2, 2, 1, 0]);
        assert_eq!(u32::from_le_bytes(bytes[106..110].try_into().unwrap()), 2);
        assert_eq!(&bytes[110..], b"hi");
    }
}
