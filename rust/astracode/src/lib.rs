//! Astracode: a scannable code made of rings of stars around a picture.
//!
//! * [`mod@format`] is the code itself: a payload of bits to stars and back, geometry, SVG.
//! * [`scan`] finds and reads codes in an image.
//! * [`astra`] is Astra's own use of the code: a member number in one ring.
//! * [`image`] and [`math`] are the small image and geometry toolkits the scanner stands on.
//! * [`packed`] lays scan results out as bytes, for the apps that call in from other languages.
//!
//! See the `README.md` next to this crate for an overview of how the scanner works.

pub mod astra;
pub mod format;
pub mod image;
pub mod math;
pub mod packed;
pub mod scan;
