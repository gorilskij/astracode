//! Ring 0: twelve stars holding a 40-bit value.
//!
//! The value is scrambled ([`super::mix`]), split into 8 five-bit symbols, most
//! significant first, followed by 4 Reed-Solomon check symbols over GF(32),
//! and the twelve are XORed with [`MASK`].
//!
//! The value's top bit says whether the code has more rings:
//!
//! * `0`: ring 0 is the whole code, and the other [`ONE_RING_BITS`] = 39 bits
//!   are the stream (the payload, a 1 bit, then zeros).
//! * `1`: then 5 bits giving the number of rings minus one (1 to 23), 2 bits
//!   giving the error-correction level of the outer rings, and the first
//!   [`HEAD_BITS`] = 32 bits of the stream.
//!
//! Either way ring 0 is a [`Header`]: it tells a reader how many rings to
//! expect before anything else is read. See [`super::code`] for the stream.
//!
//! Reading is deliberately cautious: at most one star is repaired, or two that
//! the reader has already singled out as doubtful.

use super::gf::gf32;
use super::mix::{mix, unmix};
use super::reed_solomon;
use super::{MASK, MAX_RINGS, N_DATA, N_STARS};

/// The twelve symbols of ring 0 as drawn. Each is `0..32`.
pub type Symbols = [u8; N_STARS];

/// Stream bits in ring 0 when it is the whole code.
pub const ONE_RING_BITS: usize = 39;
/// Stream bits in ring 0 when more rings follow.
pub const HEAD_BITS: usize = 32;

/// What ring 0 says about its code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Total number of rings, `1..=24`.
    pub rings: usize,
    /// Error-correction level of the outer rings, `0..4`. Always 0 for a
    /// one-ring code, which has no outer rings.
    pub level: u8,
    /// Ring 0's whole 40-bit value. The outer rings' check symbols cover all of it.
    pub value: u64,
    /// Stars of ring 0 that had to be repaired to read this.
    pub repaired: usize,
}

impl Header {
    /// How many bits of the stream ring 0 carries.
    pub fn head_bits(&self) -> usize {
        head_bits(self.rings)
    }

    /// Those bits, the first of them highest.
    pub fn head(&self) -> u64 {
        self.value & ((1 << self.head_bits()) - 1)
    }
}

/// How many bits of the stream ring 0 carries in a code with this many rings.
pub(crate) fn head_bits(rings: usize) -> usize {
    if rings == 1 {
        ONE_RING_BITS
    } else {
        HEAD_BITS
    }
}

/// Ring 0's value for a code with `rings` rings at `level`, with the first
/// bits of the stream in `head`.
pub(crate) fn value(rings: usize, level: u8, head: u64) -> u64 {
    debug_assert!(head >> head_bits(rings) == 0);
    if rings == 1 {
        head
    } else {
        1 << 39 | (rings as u64 - 1) << 34 | (level as u64) << 32 | head
    }
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

/// Read ring 0. `None` if its check fails, or if what it says cannot be: a
/// number of rings above 24, or a one-ring stream with no end.
///
/// `doubtful` may name two stars the reader is unsure of; see [`decode_value`].
pub fn read(symbols: &Symbols, doubtful: Option<(usize, usize)>) -> Option<Header> {
    let (value, repaired) = decode_value(symbols, doubtful)?;
    if value >> 39 == 0 {
        // The stream ends with a 1 bit, so it is never all zeros.
        return (value != 0).then_some(Header { rings: 1, level: 0, value, repaired });
    }
    let rings = (value >> 34 & 31) as usize + 1;
    let level = (value >> 32 & 3) as u8;
    (rings <= MAX_RINGS).then_some(Header { rings, level, value, repaired })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_misread_star_is_repaired() {
        let value = value(1, 0, 0x12_3456_789B);
        let good = encode_value(value);
        let header = Header { rings: 1, level: 0, value, repaired: 0 };
        assert_eq!(read(&good, None), Some(header));
        for star in 0..12 {
            let mut bad = good;
            bad[star] ^= 0b10110;
            assert_eq!(read(&bad, None), Some(Header { repaired: 1, ..header }));
        }
        let mut bad = good;
        bad[1] ^= 3;
        bad[7] ^= 17;
        assert_eq!(read(&bad, None), None);
        assert_eq!(read(&bad, Some((7, 1))), Some(Header { repaired: 2, ..header }));
    }

    #[test]
    fn ring_0_says_how_many_rings_there_are() {
        for (rings, level) in [(2, 0), (5, 2), (24, 3)] {
            let header = read(&encode_value(value(rings, level, 0x5555_5555)), None).unwrap();
            assert_eq!((header.rings, header.level, header.head(), header.head_bits()), (rings, level, 0x5555_5555, 32));
        }
        let header = read(&encode_value(value(1, 0, 0x55_5555_5555)), None).unwrap();
        assert_eq!((header.rings, header.head(), header.head_bits()), (1, 0x55_5555_5555, 39));
    }

    #[test]
    fn impossible_values_are_refused() {
        // A one-ring stream without its closing 1 bit.
        assert_eq!(read(&encode_value(0), None), None);
        // 25 rings.
        assert_eq!(read(&encode_value(1 << 39 | 24 << 34), None), None);
    }
}
