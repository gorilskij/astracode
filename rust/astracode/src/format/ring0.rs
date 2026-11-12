//! Ring 0: twelve stars holding a 40-bit value.
//!
//! The value is scrambled ([`super::mix`]), split into 8 five-bit symbols, most
//! significant first, followed by 4 Reed-Solomon check symbols over GF(32),
//! and the twelve are XORed with [`MASK`].
//!
//! What the value means depends on its range:
//!
//! * below [`MAX_NUMBER`]: a **one-ring code**, `metadata * 10^10 + payload`,
//!   see [`pack`]. The use this was designed for is a 12-digit member ID whose
//!   last two digits are the first ten modulo 97; only the ten are stored
//!   ([`member_id`]).
//! * from [`MAX_NUMBER`] up: the **header** of a code with more rings, see
//!   [`Header`].
//!
//! Reading is deliberately cautious: at most one star is repaired, or two that
//! the reader has already singled out as doubtful.

use super::gf::gf32;
use super::mix::{mix, unmix};
use super::reed_solomon;
use super::{FormatError, MASK, MAX_NUMBER, MAX_RINGS, METADATA_LIMIT, N_DATA, N_STARS, PAYLOAD_LIMIT};

/// The twelve symbols of ring 0 as drawn. Each is `0..32`.
pub type Symbols = [u8; N_STARS];

/// Metadata `0..=108` and a payload of up to ten digits, as one stored value.
pub fn pack(metadata: u64, payload: u64) -> Result<u64, FormatError> {
    if metadata >= METADATA_LIMIT {
        return Err(FormatError::Metadata);
    }
    if payload >= PAYLOAD_LIMIT {
        return Err(FormatError::Payload);
    }
    Ok(metadata * PAYLOAD_LIMIT + payload)
}

/// A one-ring value split into `(metadata, payload)`.
pub fn unpack(value: u64) -> (u64, u64) {
    (value / PAYLOAD_LIMIT, value % PAYLOAD_LIMIT)
}

/// The 12-digit member ID for a 10-digit payload: the payload, then the payload modulo 97.
pub fn member_id(payload: u64) -> String {
    format!("{:010}{:02}", payload, payload % 97)
}

/// The payload of a 12-digit member ID. Anything but digits is ignored;
/// fails if there are not twelve digits or the last two do not check out.
pub fn payload_of_member_id(text: &str) -> Result<u64, FormatError> {
    let digits: String = text.chars().filter(char::is_ascii_digit).collect();
    if digits.len() != 12 {
        return Err(FormatError::MemberId);
    }
    let payload: u64 = digits[..10].parse().map_err(|_| FormatError::MemberId)?;
    let check: u64 = digits[10..].parse().map_err(|_| FormatError::MemberId)?;
    if payload % 97 != check {
        return Err(FormatError::MemberId);
    }
    Ok(payload)
}

/// Any 40-bit value as ring 0's twelve symbols.
pub(crate) fn encode_value(value: u64) -> Symbols {
    let mixed = mix(value);
    let mut cw = [0u16; N_STARS];
    for (i, c) in cw.iter_mut().take(N_DATA).enumerate() {
        *c = ((mixed >> (5 * (N_DATA - 1 - i))) & 31) as u16;
    }
    let checks = reed_solomon::parity(gf32(), &cw[..N_DATA], N_STARS - N_DATA);
    cw[N_DATA..].copy_from_slice(&checks);
    std::array::from_fn(|i| cw[i] as u8 ^ MASK[i])
}

/// The twelve symbols of a one-ring code for a value from [`pack`].
pub fn encode(value: u64) -> Result<Symbols, FormatError> {
    if value >= MAX_NUMBER {
        return Err(FormatError::Value);
    }
    Ok(encode_value(value))
}

/// Ring 0's symbols back to its 40-bit value, with the number of stars repaired.
///
/// Repairs one misread star anywhere. If that fails and `doubtful` names two
/// stars the reader is unsure of, rebuilds exactly those two instead.
pub(crate) fn decode_value(symbols: &Symbols, doubtful: Option<(usize, usize)>) -> Option<(u64, usize)> {
    let mut cw: [u16; N_STARS] = std::array::from_fn(|i| (symbols[i] ^ MASK[i]) as u16);
    let repaired = match reed_solomon::correct_one(gf32(), &mut cw) {
        Some(count) => count,
        None => {
            let (i1, i2) = doubtful?;
            if !reed_solomon::fill_two(gf32(), &mut cw, i1, i2) {
                return None;
            }
            2
        }
    };
    let mixed = cw[..N_DATA].iter().fold(0u64, |acc, &d| acc << 5 | d as u64);
    Some((unmix(mixed), repaired))
}

/// Read a one-ring code: `(value, stars repaired)`, or `None` if the check
/// fails or this is the header of a bigger code.
pub fn decode(symbols: &Symbols, doubtful: Option<(usize, usize)>) -> Option<(u64, usize)> {
    decode_value(symbols, doubtful).filter(|&(value, _)| value < MAX_NUMBER)
}

/// What ring 0 of a code with more rings says about the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Total number of rings, `2..=24`.
    pub rings: usize,
    /// Error-correction level of the outer rings, `0..4`.
    pub level: u8,
    /// Ring 0's whole 40-bit value. Its low 26 bits are the first bits of the
    /// message, and the outer rings' check symbols cover all of it.
    pub value: u64,
    /// Stars of ring 0 that had to be repaired to read this.
    pub repaired: usize,
}

/// Message bits carried in ring 0 of a code with more rings.
pub(crate) const HEAD_BITS: usize = 26;

/// The header value for a code with `rings` rings at `level`, with the first
/// [`HEAD_BITS`] bits of the message in `head`.
pub(crate) fn header_value(rings: usize, level: u8, head: u32) -> u64 {
    MAX_NUMBER + ((rings as u64) << 28 | (level as u64) << 26 | head as u64)
}

/// Read ring 0 as a header, or `None` if the check fails or this is a one-ring code.
pub fn read_header(symbols: &Symbols, doubtful: Option<(usize, usize)>) -> Option<Header> {
    let (value, repaired) = decode_value(symbols, doubtful)?;
    if value < MAX_NUMBER {
        return None;
    }
    let bits = value - MAX_NUMBER;
    let rings = (bits >> 28) as usize;
    let level = ((bits >> 26) & 3) as u8;
    (2..=MAX_RINGS).contains(&rings).then_some(Header { rings, level, value, repaired })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn member_ids_round_trip() {
        assert_eq!(member_id(4_320_880_580), "432088058060");
        assert_eq!(payload_of_member_id("4320 8805 8060"), Ok(4_320_880_580));
        assert_eq!(payload_of_member_id("432088058061"), Err(FormatError::MemberId));
    }

    #[test]
    fn a_misread_star_is_repaired() {
        let value = pack(5, 4_320_880_580).unwrap();
        let good = encode(value).unwrap();
        assert_eq!(decode(&good, None), Some((value, 0)));
        for star in 0..12 {
            let mut bad = good;
            bad[star] ^= 0b10110;
            assert_eq!(decode(&bad, None), Some((value, 1)));
        }
        let mut bad = good;
        bad[1] ^= 3;
        bad[7] ^= 17;
        assert_eq!(decode(&bad, None), None);
        assert_eq!(decode(&bad, Some((7, 1))), Some((value, 2)));
    }

    #[test]
    fn headers_and_one_ring_codes_do_not_mix() {
        let symbols = encode_value(header_value(5, 2, 0x155_5555));
        assert_eq!(decode(&symbols, None), None);
        let header = read_header(&symbols, None).unwrap();
        assert_eq!((header.rings, header.level, header.repaired), (5, 2, 0));
        assert_eq!(read_header(&encode(pack(0, 1).unwrap()).unwrap(), None), None);
        assert_eq!(encode(MAX_NUMBER), Err(FormatError::Value));
    }
}
