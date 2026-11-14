//! Scan results as one array of bytes, for callers on the far side of a
//! language boundary: the Android app through JNI, the web pages (the
//! `astracode-web` repo) through WebAssembly. One array to hand over, and
//! nothing for the native side to build or look up in the caller's world; the
//! caller unpacks it.

use crate::astra::Number;
use crate::scan::Code;

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
/// Whether a payload is an Astra number is decided here, by [`crate::astra`],
/// so that no app carries its own copy of that rule.
pub fn pack(codes: &[Code]) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::Payload;

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
        let bytes = pack(&[base, text]);
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
