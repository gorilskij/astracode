//! A fixed, public, reversible scramble.
//!
//! Without it, codes for neighbouring values would differ in a star or two and
//! small values would be mostly one shade. [`mix`] makes any change in the value
//! change about half of the stored bits. It is not encryption: there is no
//! key, and anyone can run [`unmix`].

const KEYS: [u32; 4] = [0x3C6EF, 0x1B873, 0x5A827, 0x6ED9E];

/// 20 bits in, 20 bits out, in 32-bit integer arithmetic so that every
/// implementation of the format agrees exactly.
pub(crate) fn round(x: u32, key: u32) -> u32 {
    let mut t = x.wrapping_add(key).wrapping_mul(0x9E37_79B1);
    t ^= t >> 15;
    t = t.wrapping_mul(0x85EB_CA77);
    t ^= t >> 13;
    t & 0xFFFFF
}

/// Scramble a 40-bit value: a four-round Feistel network on its two 20-bit halves.
pub fn mix(value: u64) -> u64 {
    let (mut left, mut right) = ((value >> 20) as u32, (value & 0xFFFFF) as u32);
    for key in KEYS {
        (left, right) = (right, left ^ round(right, key));
    }
    (left as u64) << 20 | right as u64
}

/// Undo [`mix`].
pub fn unmix(value: u64) -> u64 {
    let (mut left, mut right) = ((value >> 20) as u32, (value & 0xFFFFF) as u32);
    for key in KEYS.into_iter().rev() {
        (left, right) = (right ^ round(left, key), left);
    }
    (left as u64) << 20 | right as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmix_undoes_mix() {
        for value in [0u64, 1, 54_320_880_580, (1 << 40) - 1, 0x12345_6789A] {
            assert!(mix(value) < 1 << 40);
            assert_eq!(unmix(mix(value)), value);
        }
    }
}
