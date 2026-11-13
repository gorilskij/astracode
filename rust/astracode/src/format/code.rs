//! A whole code: a payload laid out over ring 0 and, if it needs them, more rings.
//!
//! The payload ([`Payload`], a string of bits) becomes a *stream*: the
//! payload's bits, then a single 1 bit, then zeros to fill the code. The first
//! bits of the stream ride in ring 0 ([`ring0`]): 39 of them if ring 0 is the
//! whole code, 32 if more rings follow.
//!
//! The stars of rings 1 and up, in order, are taken in pairs as 10-bit symbols
//! (first star = high five bits) forming a single Reed-Solomon codeword over
//! GF(1024): the rest of the stream, then check symbols. How many are check
//! symbols is set by the *error-correction level*: level 0, 1, 2 or 3 makes
//! 1/8, 2/8, 3/8 or 4/8 of the outer symbols check symbols.
//!
//! The check symbols also cover ring 0's 40-bit value, as four leading symbols
//! that are not drawn a second time. That ties the rings together: outer rings
//! only validate against the ring 0 they were made with, and ring 0 says how
//! many rings there must be. A code with a ring removed is not a smaller valid
//! code. Outer symbols are XORed with a fixed pseudo-random sequence before
//! drawing so that the two shades stay balanced.
//!
//! A code has as many rings as its payload needs ([`rings_for`]); [`encode`]
//! works that out. One ring holds 38 bits.

use super::gf::gf1024;
use super::payload::Payload;
use super::reed_solomon;
use super::ring0::{self, Header, Symbols};
use super::{mix, ring_stars, FormatError, LEVELS, MAX_RINGS};

/// The symbols of every ring as drawn: ring 0's twelve, ring 1's eighteen, and so on.
pub type RingSymbols = Vec<Vec<u8>>;

/// Number of 10-bit symbols in rings 1 and up.
pub fn outer_symbols(rings: usize) -> usize {
    (1..rings).map(ring_stars).sum::<usize>() / 2
}

/// How many of the outer symbols are check symbols at this level. Never fewer
/// than two, except in a one-ring code, which has no outer symbols at all.
pub fn check_symbols(rings: usize, level: u8) -> usize {
    if rings == 1 {
        0
    } else {
        (outer_symbols(rings) * (level as usize + 1)).div_ceil(8).max(2)
    }
}

/// Length of the stream: ring 0's share plus the outer rings' data symbols.
fn stream_bits(rings: usize, level: u8) -> usize {
    ring0::head_bits(rings) + 10 * (outer_symbols(rings) - check_symbols(rings, level))
}

/// Longest payload, in bits, that a code with this many rings holds at this level.
pub fn capacity_bits(rings: usize, level: u8) -> usize {
    stream_bits(rings, level) - 1
}

/// Longest payload, in whole bytes, that a code with this many rings holds at this level.
pub fn capacity(rings: usize, level: u8) -> usize {
    capacity_bits(rings, level) / 8
}

/// The fewest rings that hold a payload of `bits` bits at this level, or
/// `None` if even 24 rings are too few.
pub fn rings_for(bits: usize, level: u8) -> Option<usize> {
    (1..=MAX_RINGS).find(|&rings| capacity_bits(rings, level) >= bits)
}

/// How many wrong outer symbols a reader repairs at this level.
///
/// Two check symbols are always kept back for detection, so that a badly
/// misread code is refused rather than "repaired" into a different payload.
pub fn max_repairs(rings: usize, level: u8) -> usize {
    check_symbols(rings, level).saturating_sub(2) / 2
}

/// The whitening sequence XORed onto outer symbol `j`.
fn whiten(j: usize) -> u16 {
    (mix::round(j as u32, 0x2545F) & 1023) as u16
}

/// Ring 0's value as the four leading symbols the outer codeword also covers.
fn covered(value: u64) -> [u16; 4] {
    [30, 20, 10, 0].map(|shift| ((value >> shift) & 1023) as u16)
}

/// Read `count` bits, most significant first, starting at bit `from`.
fn take(bits: &[u8], from: usize, count: usize) -> u64 {
    bits[from..from + count].iter().fold(0, |acc, &b| acc << 1 | b as u64)
}

/// Turn a payload into the symbols of every ring, using the fewest rings that hold it.
///
/// `level` is the error-correction level of the outer rings, `0..4`; it makes
/// no difference to a payload that fits in one ring.
pub fn encode(payload: &Payload, level: u8) -> Result<RingSymbols, FormatError> {
    if level >= LEVELS {
        return Err(FormatError::Level);
    }
    let too_long = FormatError::TooLong { bits: payload.len(), max: capacity_bits(MAX_RINGS, level) };
    encode_with_rings(payload, level, rings_for(payload.len(), level).ok_or(too_long)?)
}

/// As [`encode`], but with a given number of rings, `1..=24`, which must be
/// enough for the payload. More rings than needed make a bigger code holding
/// the same thing.
pub fn encode_with_rings(payload: &Payload, level: u8, rings: usize) -> Result<RingSymbols, FormatError> {
    if !(1..=MAX_RINGS).contains(&rings) {
        return Err(FormatError::Rings);
    }
    if level >= LEVELS {
        return Err(FormatError::Level);
    }
    let total = stream_bits(rings, level);
    if payload.len() + 1 > total {
        return Err(FormatError::TooLong { bits: payload.len(), max: total - 1 });
    }

    // The stream: the payload's bits, a 1, then zeros.
    let mut bits: Vec<u8> = payload.bits().map(u8::from).collect();
    bits.push(1);
    bits.resize(total, 0);

    let head_bits = ring0::head_bits(rings);
    let value = ring0::value(rings, level, take(&bits, 0, head_bits));
    let mut out = vec![ring0::encode_value(value).to_vec()];
    if rings == 1 {
        return Ok(out);
    }

    let checks = check_symbols(rings, level);
    let data_count = outer_symbols(rings) - checks;
    let mut codeword: Vec<u16> = covered(value).to_vec();
    codeword.extend((0..data_count).map(|j| take(&bits, head_bits + 10 * j, 10) as u16));
    let parity = reed_solomon::parity(gf1024(), &codeword, checks);
    codeword.extend(parity);

    // Everything after the four covered symbols is drawn, two stars per symbol.
    let stars: Vec<u8> = codeword[4..]
        .iter()
        .enumerate()
        .flat_map(|(j, &symbol)| {
            let drawn = symbol ^ whiten(j);
            [(drawn >> 5) as u8, (drawn & 31) as u8]
        })
        .collect();
    let mut at = 0;
    for k in 1..rings {
        out.push(stars[at..at + ring_stars(k)].to_vec());
        at += ring_stars(k);
    }
    Ok(out)
}

/// Read the symbols of every ring back into the payload, with the number of
/// outer symbols that had to be repaired.
///
/// `None` if ring 0 does not check out, if the number or size of the rings is
/// not what ring 0 says, or if the outer rings cannot be repaired within
/// [`max_repairs`].
pub fn decode(rings: &[Vec<u8>]) -> Option<(Payload, usize)> {
    let ring0: Symbols = rings.first()?.as_slice().try_into().ok()?;
    let header = ring0::read(&ring0, None)?;
    decode_outer(&header, &rings[1..])
}

/// As [`decode`], for a reader that has already made sense of ring 0.
/// `outer_rings` are rings 1 and up; none for a one-ring code.
pub fn decode_outer(header: &Header, outer_rings: &[Vec<u8>]) -> Option<(Payload, usize)> {
    let (rings, level) = (header.rings, header.level);
    if outer_rings.len() != rings - 1 || outer_rings.iter().enumerate().any(|(k, r)| r.len() != ring_stars(k + 1)) {
        return None;
    }
    let head_bits = header.head_bits();
    let mut bits: Vec<u8> = (0..head_bits).rev().map(|i| (header.head() >> i & 1) as u8).collect();
    let mut repaired = 0;
    if rings > 1 {
        let stars: Vec<u8> = outer_rings.iter().flatten().copied().collect();
        let mut codeword: Vec<u16> = covered(header.value).to_vec();
        codeword.extend(stars.chunks_exact(2).enumerate().map(|(j, pair)| ((pair[0] as u16) << 5 | pair[1] as u16) ^ whiten(j)));
        let checks = check_symbols(rings, level);
        let fixed;
        (fixed, repaired) = reed_solomon::correct_errors(gf1024(), &codeword, checks, max_repairs(rings, level))?;
        // Ring 0's value is known, not read here; a "repair" to it means the outer rings belong to another code.
        if fixed[..4] != covered(header.value) {
            return None;
        }
        let data_count = codeword.len() - 4 - checks;
        for &symbol in &fixed[4..4 + data_count] {
            bits.extend((0..10).rev().map(|i| (symbol >> i & 1) as u8));
        }
    }
    // The payload ends just before the last 1 bit.
    let end = bits.iter().rposition(|&b| b == 1)?;
    Some((Payload::from_bits(bits[..end].iter().map(|&b| b == 1)), repaired))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(len: usize) -> Payload {
        Payload::from_bytes((0..len).map(|i| (i * 31 + 7) as u8).collect::<Vec<u8>>())
    }

    #[test]
    fn a_payload_round_trips_at_every_size() {
        for (rings, level) in [(1, 0), (2, 0), (2, 3), (3, 1), (7, 2), (24, 0), (24, 3)] {
            for len in [0, 1, capacity(rings, level)] {
                let payload = bytes(len);
                let symbols = encode_with_rings(&payload, level, rings).unwrap();
                assert_eq!(symbols.len(), rings);
                assert_eq!(decode(&symbols), Some((payload, 0)));
            }
            let full = Payload::from_bits((0..capacity_bits(rings, level)).map(|i| i % 3 == 0));
            assert_eq!(decode(&encode_with_rings(&full, level, rings).unwrap()), Some((full.clone(), 0)));
            let too_long = Payload::from_bits(full.bits().chain([true]));
            assert!(matches!(encode_with_rings(&too_long, level, rings), Err(FormatError::TooLong { .. })));
        }
    }

    #[test]
    fn a_code_has_as_many_rings_as_its_payload_needs() {
        assert_eq!(capacity_bits(1, 0), 38);
        assert_eq!(encode(&bytes(4), 1).unwrap().len(), 1);
        assert_eq!(encode(&Payload::from_number((1 << 38) - 1, 38).unwrap(), 3).unwrap().len(), 1);
        assert_eq!(encode(&bytes(5), 1).unwrap().len(), 2);
        for level in 0..LEVELS {
            for len in [0usize, 9, 40, 200, 600] {
                let symbols = encode(&bytes(len), level).unwrap();
                let rings = symbols.len();
                assert!(capacity_bits(rings, level) >= 8 * len);
                assert!(rings == 1 || capacity_bits(rings - 1, level) < 8 * len);
                assert_eq!(decode(&symbols), Some((bytes(len), 0)));
            }
            let most = capacity(MAX_RINGS, level);
            assert_eq!(encode(&bytes(most), level).unwrap().len(), MAX_RINGS);
            assert!(matches!(encode(&bytes(most + 1), level), Err(FormatError::TooLong { .. })));
        }
        assert_eq!(encode(&bytes(1), LEVELS), Err(FormatError::Level));
        assert_eq!(encode_with_rings(&bytes(1), 0, 25), Err(FormatError::Rings));
    }

    #[test]
    fn damage_is_repaired_up_to_the_limit_and_refused_beyond_it() {
        let payload = Payload::from_bytes(b"astracode".to_vec());
        let good = encode_with_rings(&payload, 2, 6).unwrap();
        let limit = max_repairs(6, 2);
        let mut bad = good.clone();
        for i in 0..limit {
            bad[1 + i % 5][2 * (i / 5)] ^= 9; // one star of a pair spoils the whole symbol
        }
        assert_eq!(decode(&bad), Some((payload.clone(), limit)));
        bad[5][40] ^= 1;
        assert!(decode(&bad).is_none_or(|(p, _)| p == payload));
        assert_eq!(max_repairs(1, 0), 0);
    }

    #[test]
    fn rings_cannot_be_removed_or_swapped() {
        let a = encode_with_rings(&Payload::from_bytes(b"hello world, this is".to_vec()), 1, 3).unwrap();
        let b = encode_with_rings(&Payload::from_bytes(b"jello world, this is".to_vec()), 1, 3).unwrap();
        assert_eq!(decode(&a[..2]), None);
        // Ring 0 of a bigger code is not a code by itself.
        assert_eq!(decode(&a[..1]), None);
        let mixed = vec![b[0].clone(), a[1].clone(), a[2].clone()];
        assert_eq!(decode(&mixed), None);
        // Nor does a one-ring code accept rings added around it.
        let one = encode(&bytes(3), 0).unwrap();
        assert_eq!(decode(&[one[0].clone(), a[1].clone()]), None);
    }
}
