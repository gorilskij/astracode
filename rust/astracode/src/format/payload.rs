//! What a code carries: a string of bits.
//!
//! The format gives the bits no meaning. Most payloads are whole bytes (text,
//! a link, binary data), and [`Payload::from_bytes`] and [`Payload::as_bytes`]
//! cover that. Lengths that are not a multiple of eight are just as valid, and
//! are how a small code makes the most of its room: one ring holds 38 bits,
//! which is four bytes or a number up to 274 877 906 943.

/// A string of bits, most often whole bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Payload {
    /// The bits, eight to a byte, the first bit in the top of the first byte.
    /// Unused low bits of the last byte are zero.
    bytes: Vec<u8>,
    /// Number of bits.
    len: usize,
}

impl Payload {
    /// A payload of whole bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Payload {
        let bytes = bytes.into();
        Payload { len: 8 * bytes.len(), bytes }
    }

    /// A payload from its bits, first bit first.
    pub fn from_bits(bits: impl IntoIterator<Item = bool>) -> Payload {
        let mut payload = Payload::default();
        for bit in bits {
            if payload.len % 8 == 0 {
                payload.bytes.push(0);
            }
            if bit {
                *payload.bytes.last_mut().expect("a byte was just pushed") |= 0x80 >> (payload.len % 8);
            }
            payload.len += 1;
        }
        payload
    }

    /// A payload of `len` bits holding `value`, most significant bit first.
    /// `None` if `value` needs more than `len` bits or `len` is more than 64.
    pub fn from_number(value: u64, len: usize) -> Option<Payload> {
        if len > 64 || (len < 64 && value >> len != 0) {
            return None;
        }
        Some(Payload::from_bits((0..len).rev().map(|i| value >> i & 1 == 1)))
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no bits at all.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bits, first bit first.
    pub fn bits(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(|i| self.bytes[i / 8] & (0x80 >> (i % 8)) != 0)
    }

    /// The payload as bytes, if it is a whole number of them.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        (self.len % 8 == 0).then_some(&self.bytes)
    }

    /// The payload as text, if it is whole bytes of valid UTF-8.
    pub fn as_text(&self) -> Option<&str> {
        std::str::from_utf8(self.as_bytes()?).ok()
    }

    /// The payload read as one number, most significant bit first, if it has at most 64 bits.
    pub fn as_number(&self) -> Option<u64> {
        (self.len <= 64).then(|| self.bits().fold(0u64, |acc, bit| acc << 1 | bit as u64))
    }

    /// The bits packed into bytes whatever the length: the last byte is filled
    /// out with zero bits. Use together with [`Payload::len`].
    pub fn padded_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_and_bits_are_the_same_thing() {
        let payload = Payload::from_bytes(b"hi".to_vec());
        assert_eq!(payload.len(), 16);
        assert_eq!(payload.as_text(), Some("hi"));
        assert_eq!(Payload::from_bits(payload.bits()), payload);
        assert_eq!(payload.as_number(), Some(0x6869));
    }

    #[test]
    fn odd_lengths_are_not_bytes() {
        let payload = Payload::from_number(0b101, 3).unwrap();
        assert_eq!(payload.len(), 3);
        assert_eq!(payload.as_bytes(), None);
        assert_eq!(payload.padded_bytes(), &[0b1010_0000]);
        assert_eq!(payload.as_number(), Some(5));
        assert_eq!(Payload::from_number(8, 3), None);
        assert_eq!(Payload::from_number(u64::MAX, 64).unwrap().as_number(), Some(u64::MAX));
        assert!(Payload::default().is_empty());
        assert_eq!(Payload::default().as_bytes(), Some(&[][..]));
    }
}
