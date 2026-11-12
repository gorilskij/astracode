//! The Astracode format: what a code looks like and what its stars mean.
//!
//! Nothing here touches images. This module turns values into star symbols and
//! back ([`ring0`], [`message`]), describes where the stars sit ([`geometry`]),
//! and draws a code as SVG ([`svg`]). The scanner in [`crate::scan`] is built
//! on top of it.
//!
//! # Layout
//!
//! All lengths are in *model units*: ring 0's star centres lie on a circle of
//! radius [`R`] = 100 around the origin, x to the right, y down.
//!
//! * Ring `k` has [`ring_stars`]`(k)` = 12 + 6k stars at radius
//!   [`ring_radius`]`(k)` = 100 + 40k. Star 0 of every ring is at 12 o'clock
//!   and the rest follow clockwise, so every ring has a star on each of the
//!   six spokes at multiples of 60 degrees.
//! * Every star is the same size and upright, never rotated: a regular
//!   pentagram of outer radius [`STAR_RO`]. Its centre pentagon is always
//!   solid ink. Each of its five limbs is either solid (bit 1) or a lighter
//!   shade (bit 0). The top limb is bit 4 of the star's 5-bit *symbol*, then
//!   clockwise bits 3, 2, 1, 0.
//! * The centre may hold any picture, or nothing, inside a circle of radius
//!   [`CENTER_R`]. Readers do not look at it. The band between that circle and
//!   ring 0's stars must stay background colour.
//!
//! # What the stars carry
//!
//! * **Ring 0** always holds a 40-bit value in 8 symbols, followed by 4
//!   Reed-Solomon check symbols over GF(32), XORed with [`MASK`]. See
//!   [`ring0`].
//!   * Values below [`MAX_NUMBER`] make a *one-ring code*:
//!     `metadata * 10^10 + payload`.
//!   * Values from [`MAX_NUMBER`] up are the *header* of a code with more
//!     rings: how many rings, the error-correction level, and the first bits
//!     of the message. So ring 0 cut out of a bigger code never reads as a
//!     one-ring code.
//! * **Rings 1 and up**, taken together, are one Reed-Solomon codeword over
//!   GF(1024), two stars per symbol, carrying a byte string. See [`message`].

pub mod geometry;
pub mod gf;
pub mod message;
pub mod mix;
pub mod reed_solomon;
pub mod ring0;
pub mod svg;

/// Radius of ring 0, which defines the model unit.
pub const R: f64 = 100.0;
/// Outer radius of a star (centre to limb tip).
pub const STAR_RO: f64 = 19.5;
/// Inner radius of a star (centre to the notch between two limbs). A regular
/// pentagram has inner/outer ratio sin 18 / sin 54.
pub const STAR_RI: f64 = STAR_RO * 0.381_966_011_250_105_1;
/// The centre picture must stay within this radius.
pub const CENTER_R: f64 = 73.0;
/// Clear margin around the outermost stars.
pub const MARGIN: f64 = 16.0;
/// Ink fraction of a light limb when drawing. Readers work the level out for
/// themselves, so this is only the generator's default.
pub const SHADE: f64 = 0.7;

/// Stars in ring 0, and symbols in its codeword.
pub const N_STARS: usize = 12;
/// Data symbols in ring 0.
pub const N_DATA: usize = 8;
/// Check symbols in ring 0.
pub const N_CHECK: usize = N_STARS - N_DATA;

/// One-ring payloads are `0 .. PAYLOAD_LIMIT`: ten decimal digits.
pub const PAYLOAD_LIMIT: u64 = 10_000_000_000;
/// One-ring metadata is `0 .. METADATA_LIMIT`.
pub const METADATA_LIMIT: u64 = (1 << 40) / PAYLOAD_LIMIT;
/// Every one-ring value is below this. From here up to 2^40 are headers.
pub const MAX_NUMBER: u64 = METADATA_LIMIT * PAYLOAD_LIMIT;

/// XORed onto ring 0's twelve symbols before drawing, so no code is one flat shade.
pub const MASK: [u8; N_STARS] = [0x15, 0x0A, 0x13, 0x1C, 0x09, 0x06, 0x1E, 0x01, 0x17, 0x08, 0x1B, 0x04];

/// Most rings a code may have. 24 rings have 1932 outer stars, 966 symbols,
/// which fits GF(1024)'s limit of 1023.
pub const MAX_RINGS: usize = 24;
/// Radius added per ring.
pub const RING_STEP: f64 = 40.0;
/// Error-correction levels are `0 .. LEVELS`.
pub const LEVELS: u8 = 4;

/// Number of stars in ring `k`.
pub const fn ring_stars(k: usize) -> usize {
    12 + 6 * k
}

/// Radius of ring `k`.
pub fn ring_radius(k: usize) -> f64 {
    R + RING_STEP * k as f64
}

/// Half the side of the square a code with this many rings occupies, margin included.
pub fn half_side(rings: usize) -> f64 {
    ring_radius(rings - 1) + STAR_RO + MARGIN
}

/// Total number of stars in a code with this many rings.
pub fn total_stars(rings: usize) -> usize {
    (0..rings).map(ring_stars).sum()
}

/// Why a value could not be turned into a code.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FormatError {
    #[error("metadata must be 0..{}", METADATA_LIMIT - 1)]
    Metadata,
    #[error("payload must have at most 10 digits")]
    Payload,
    #[error("value is not a one-ring value")]
    Value,
    #[error("a message code has 2..{MAX_RINGS} rings")]
    Rings,
    #[error("the error-correction level is 0..{}", LEVELS - 1)]
    Level,
    #[error("message too long: this code holds at most {max} bytes")]
    MessageTooLong { max: usize },
    #[error("not a valid member ID")]
    MemberId,
}
