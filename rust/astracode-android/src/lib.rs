//! The Astracode scanner for the Android app: one native function, called
//! through JNI from `app.astracode.scanner.NativeScanner`.
//!
//! The app hands over the camera frame's brightness plane and gets back a
//! byte array describing the codes found, laid out as [`encode`] documents.
//! Keeping the interface to one array of bytes means no Java objects are
//! built or looked up from native code; `Scanner.kt` unpacks it.

use std::panic::{catch_unwind, AssertUnwindSafe};

use astracode::astra::Number;
use astracode::image::Gray;
use astracode::scan::{scan_gray, Code, Options};
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jbyteArray, jint};
use jni::JNIEnv;

/// The codes found, as bytes (integers and floats little-endian):
///
/// ```text
/// u8   number of codes
/// then, for each code:
///   u8   rings
///   u8   error-correction level (0 for a one-ring code)
///   u8   flags: bit 0 = seen as a mirror image, bit 1 = read from the star shapes (bent surface)
///   u16  stars or star pairs repaired
///   f32  how inked the light limbs are relative to solid, 0 to 1
///   f32  x 8: the corners of the code's square, x then y, in frame pixels:
///             top left, top right, bottom right, bottom left
///   u32  length of the payload in bits
///   then the payload: that many bits packed into bytes, first bit highest,
///             the last byte padded with zero bits
///   u8   1 if the payload is an Astra number, else 0; if 1, then:
///     u8 metadata, u64 payload, 12 ASCII digits of member ID
/// ```
///
/// Whether a payload is an Astra number is decided here, by
/// [`astracode::astra`], so that the app does not carry its own copy of that rule.
fn encode(codes: &[Code]) -> Vec<u8> {
    let mut out = vec![codes.len().min(255) as u8];
    for code in codes.iter().take(255) {
        out.extend([code.rings as u8, code.level, code.mirrored as u8 | (code.shaped as u8) << 1]);
        out.extend((code.repaired.min(u16::MAX as usize) as u16).to_le_bytes());
        out.extend((code.light_level as f32).to_le_bytes());
        for corner in code.corners {
            out.extend((corner[0] as f32).to_le_bytes());
            out.extend((corner[1] as f32).to_le_bytes());
        }
        out.extend((code.payload.len() as u32).to_le_bytes());
        out.extend(code.payload.padded_bytes());
        match Number::from_payload(&code.payload) {
            Some(number) => {
                out.extend([1, number.metadata]);
                out.extend(number.payload.to_le_bytes());
                out.extend(number.member_id().as_bytes());
            }
            None => out.push(0),
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
    use astracode::scan::Payload;

    #[test]
    fn results_are_laid_out_as_documented() {
        let corners = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let base = Code {
            payload: Number::new(5, 4_320_880_580).unwrap().to_payload(),
            rings: 1,
            level: 0,
            corners,
            mirrored: true,
            repaired: 1,
            shaped: false,
            confidence: 1.0,
            separation: 9.0,
            light_level: 0.5,
        };
        let text = Code { payload: Payload::from_bytes("hi"), rings: 3, level: 2, mirrored: false, shaped: true, ..base.clone() };
        let bytes = encode(&[base, text]);
        assert_eq!(bytes[0], 2);
        // First code: rings, level, flags, repaired.
        assert_eq!(&bytes[1..6], &[1, 0, 1, 1, 0]);
        assert_eq!(f32::from_le_bytes(bytes[6..10].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(bytes[10..14].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[38..42].try_into().unwrap()), 8.0);
        // Its payload: 38 bits in five bytes, then the Astra number they stand for.
        assert_eq!(u32::from_le_bytes(bytes[42..46].try_into().unwrap()), 38);
        assert_eq!(u64::from_be_bytes([0, 0, 0, bytes[46], bytes[47], bytes[48], bytes[49], bytes[50]]) >> 2, 54_320_880_580);
        assert_eq!(&bytes[51..53], &[1, 5]);
        assert_eq!(u64::from_le_bytes(bytes[53..61].try_into().unwrap()), 4_320_880_580);
        assert_eq!(&bytes[61..73], b"432088058060");
        // Second code: plain bytes, not an Astra number.
        assert_eq!(&bytes[73..78], &[3, 2, 2, 1, 0]);
        assert_eq!(u32::from_le_bytes(bytes[114..118].try_into().unwrap()), 16);
        assert_eq!(&bytes[118..], b"hi\0");
    }
}
