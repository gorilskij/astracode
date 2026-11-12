//! Astracode: a scannable code made of rings of stars around a picture.
//!
//! * [`mod@format`] is the code itself: values to stars and back, geometry, SVG.
//! * [`scan`] finds and reads codes in an image.
//! * [`image`] and [`math`] are the small image and geometry toolkits the scanner stands on.
//!
//! See the `README.md` next to this crate for an overview of how the scanner works.

pub mod format;
pub mod image;
pub mod math;
pub mod scan;
