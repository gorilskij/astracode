//! Codes with more than one ring: a byte string in the outer rings.
//!
//! The stars of rings 1 and up, in order, are taken in pairs as 10-bit symbols
//! (first star = high five bits) forming a single Reed-Solomon codeword over
//! GF(1024). A share of those symbols are check symbols, set by the
//! *error-correction level*: level 0, 1, 2 or 3 makes 1/8, 2/8, 3/8 or 4/8 of
//! them check symbols.
//!
//! The check symbols also cover ring 0's 40-bit value, as four leading symbols
//! that are not drawn a second time. That ties the rings together: outer rings
//! only validate against the ring 0 they were made with, and ring 0 says how
//! many rings there must be. A code with a ring removed is not a smaller valid
//! code.
//!
//! The message itself is laid out as a bit string: the message bits, then a
//! single 1 bit, then zeros to fill. Its first 26 bits ride in ring 0's header
//! and the rest fill the data symbols. Outer symbols are XORed with a fixed
//! pseudo-random sequence before drawing so that the two shades stay balanced.

use super::gf::gf1024;
use super::reed_solomon;
use super::ring0::{self, Symbols, HEAD_BITS};
use super::{mix, ring_stars, FormatError, LEVELS, MAX_NUMBER, MAX_RINGS};

/// The symbols of every ring as drawn: ring 0's twelve, ring 1's eighteen, and so on.
pub type RingSymbols = Vec<Vec<u8>>;

/// Number of 10-bit symbols in rings 1 and up.
pub fn outer_symbols(rings: usize) -> usize {
    (1..rings).map(ring_stars).sum::<usize>() / 2
}

/// How many of the outer symbols are check symbols at this level. Never fewer than two.
pub fn check_symbols(rings: usize, level: u8) -> usize {
    (outer_symbols(rings) * (level as usize + 1)).div_ceil(8).max(2)
}

/// Bits available for the padded message: [`HEAD_BITS`] in ring 0 plus the data symbols.
fn capacity_bits(rings: usize, level: u8) -> usize {
    HEAD_BITS + 10 * (outer_symbols(rings) - check_symbols(rings, level))
}

/// Longest message, in bytes, that a code with this many rings holds at this level.
pub fn capacity(rings: usize, level: u8) -> usize {
    (capacity_bits(rings, level) - 1) / 8
}

/// How many wrong outer symbols a reader repairs at this level.
///
/// Two check symbols are always kept back for detection, so that a badly
/// misread code is refused rather than "repaired" into a different message.
pub fn max_repairs(rings: usize, level: u8) -> usize {
    (check_symbols(rings, level) - 2) / 2
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
fn take(bits: &[u8], from: usize, count: usize) -> u32 {
    bits[from..from + count].iter().fold(0, |acc, &b| acc << 1 | b as u32)
}

/// Turn a byte string into the symbols of every ring.
///
/// `rings` is `2..=24` and `level` is `0..4`; see [`capacity`] for how long
/// the message may be.
pub fn encode(message: &[u8], rings: usize, level: u8) -> Result<RingSymbols, FormatError> {
    if !(2..=MAX_RINGS).contains(&rings) {
        return Err(FormatError::Rings);
    }
    if level >= LEVELS {
        return Err(FormatError::Level);
    }
    let total = capacity_bits(rings, level);
    if 8 * message.len() + 1 > total {
        return Err(FormatError::MessageTooLong { max: capacity(rings, level) });
    }
    let checks = check_symbols(rings, level);
    let data_count = outer_symbols(rings) - checks;

    // The message bits, a 1, then zeros.
    let mut bits = Vec::with_capacity(total);
    for &byte in message {
        bits.extend((0..8).rev().map(|i| byte >> i & 1));
    }
    bits.push(1);
    bits.resize(total, 0);

    let value = ring0::header_value(rings, level, take(&bits, 0, HEAD_BITS));
    let mut codeword: Vec<u16> = covered(value).to_vec();
    codeword.extend((0..data_count).map(|j| take(&bits, HEAD_BITS + 10 * j, 10) as u16));
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
    let mut out = vec![ring0::encode_value(value).to_vec()];
    let mut at = 0;
    for k in 1..rings {
        out.push(stars[at..at + ring_stars(k)].to_vec());
        at += ring_stars(k);
    }
    Ok(out)
}

/// Read the symbols of every ring back into the message, with the number of
/// outer symbols that had to be repaired.
///
/// `None` if ring 0 is not a header, if the number or size of the rings is not
/// what the header says, or if the outer rings cannot be repaired within
/// [`max_repairs`].
pub fn decode(rings: &[Vec<u8>]) -> Option<(Vec<u8>, usize)> {
    let ring0: Symbols = rings.first()?.as_slice().try_into().ok()?;
    let header = ring0::read_header(&ring0, None)?;
    decode_outer(&header, &rings[1..])
}

/// As [`decode`], for a reader that has already made sense of ring 0.
pub fn decode_outer(header: &ring0::Header, outer_rings: &[Vec<u8>]) -> Option<(Vec<u8>, usize)> {
    let (rings, level) = (header.rings, header.level);
    if outer_rings.len() != rings - 1 || outer_rings.iter().enumerate().any(|(k, r)| r.len() != ring_stars(k + 1)) {
        return None;
    }
    let stars: Vec<u8> = outer_rings.iter().flatten().copied().collect();
    let mut codeword: Vec<u16> = covered(header.value).to_vec();
    codeword.extend(stars.chunks_exact(2).enumerate().map(|(j, pair)| ((pair[0] as u16) << 5 | pair[1] as u16) ^ whiten(j)));
    let checks = check_symbols(rings, level);
    let (fixed, repaired) = reed_solomon::correct_errors(gf1024(), &codeword, checks, max_repairs(rings, level))?;
    // Ring 0's value is known, not read here; a "repair" to it means the outer rings belong to another code.
    if fixed[..4] != covered(header.value) {
        return None;
    }

    let data_count = codeword.len() - 4 - checks;
    let mut bits: Vec<u8> = Vec::with_capacity(HEAD_BITS + 10 * data_count);
    let head = (header.value - MAX_NUMBER) as u32;
    bits.extend((0..HEAD_BITS).rev().map(|i| (head >> i & 1) as u8));
    for &symbol in &fixed[4..4 + data_count] {
        bits.extend((0..10).rev().map(|i| (symbol >> i & 1) as u8));
    }
    // The message ends just before the last 1 bit.
    let end = bits.iter().rposition(|&b| b == 1)?;
    if end % 8 != 0 {
        return None;
    }
    let message = bits[..end].chunks_exact(8).map(|byte| byte.iter().fold(0u8, |acc, &b| acc << 1 | b)).collect();
    Some((message, repaired))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_round_trips_at_every_size() {
        for (rings, level) in [(2, 0), (2, 3), (3, 1), (7, 2), (24, 0), (24, 3)] {
            for len in [0, 1, capacity(rings, level)] {
                let message: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
                let symbols = encode(&message, rings, level).unwrap();
                assert_eq!(symbols.len(), rings);
                assert_eq!(decode(&symbols), Some((message, 0)));
            }
            let too_long = vec![0u8; capacity(rings, level) + 1];
            assert!(matches!(encode(&too_long, rings, level), Err(FormatError::MessageTooLong { .. })));
        }
    }

    #[test]
    fn damage_is_repaired_up_to_the_limit_and_refused_beyond_it() {
        let message = b"astracode".to_vec();
        let good = encode(&message, 6, 2).unwrap();
        let limit = max_repairs(6, 2);
        let mut bad = good.clone();
        for i in 0..limit {
            bad[1 + i % 5][2 * (i / 5)] ^= 9; // one star of a pair spoils the whole symbol
        }
        assert_eq!(decode(&bad), Some((message.clone(), limit)));
        bad[5][40] ^= 1;
        assert_ne!(decode(&bad).map(|r| r.0), Some(vec![0u8; 0]));
        assert!(decode(&bad).is_none_or(|(m, _)| m == message));
    }

    #[test]
    fn rings_cannot_be_removed_or_swapped() {
        let a = encode(b"hello world, this is", 3, 1).unwrap();
        let b = encode(b"jello world, this is", 3, 1).unwrap();
        assert_eq!(decode(&a[..2]), None);
        assert_eq!(decode(&a[..1]), None);
        let mixed = vec![b[0].clone(), a[1].clone(), a[2].clone()];
        assert_eq!(decode(&mixed), None);
        // Ring 0 of a message code is not a one-ring code.
        let ring0: Symbols = a[0].as_slice().try_into().unwrap();
        assert_eq!(ring0::decode(&ring0, None), None);
    }
}
