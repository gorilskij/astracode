//! Finding and reading codes in an image.
//!
//! [`scan_rgb8`] and [`scan_gray`] are the entry points. The work is laid out
//! like a QR reader's:
//!
//! 1. `blobs`: binarise against the local mean, in both polarities, and
//!    collect connected regions that could be stars.
//! 2. `rings`: walk rings of twelve evenly spaced, similar blobs.
//! 3. `candidate`: fit a homography to a ring, throw out look-alikes
//!    (`checks`), find which way is up from the star shapes (`align`),
//!    and read ring 0's limbs (`limbs`). Ring 0 says how many rings the
//!    code has, and may be all of it.
//! 4. `outer`: for a bigger code, locate the remaining rings outwards from
//!    ring 0 and read them.
//! 5. `bent`: when a ring cannot be read as flat, each star's own outline
//!    gives its local stretch, shear and rotation, and reading is retried
//!    with those.
//!
//! Only steps 1 and 2 use the binarised image; everything after samples grey
//! levels. Whatever is drawn in the centre of a code is never looked at.
//!
//! The scanner is tuned never to return a wrong value: when in doubt it
//! returns nothing, and the caller scans the next frame.
//!
//! A result's payload is a string of bits. What they mean is the caller's
//! business; see [`crate::astra`] for one convention.

mod align;
mod bent;
mod blobs;
mod candidate;
mod checks;
mod limbs;
mod outer;
mod par;
mod patterns;
mod rings;

use crate::format::geometry::inside;
pub use crate::format::Payload;
use crate::image::Gray;
use crate::math::{distance, Homography, LocalMap, Point};

/// Least gap between the light and solid shades, in units of their spread,
/// for limbs to be read at all.
pub(crate) const MIN_SHADE_SEPARATION: f64 = 4.0;

/// A code found in an image.
#[derive(Debug, Clone)]
pub struct Code {
    /// What the code carries.
    pub payload: Payload,
    /// Number of rings, 1 to 24.
    pub rings: usize,
    /// Error-correction level of the outer rings, 0 to 3. Always 0 for a
    /// one-ring code, which has none.
    pub level: u8,
    /// The corners of the code's square in the image: top left, top right,
    /// bottom right, bottom left, as the code itself is oriented.
    pub corners: [Point; 4],
    /// Whether the code was seen as a mirror image (through glass, from behind).
    pub mirrored: bool,
    /// What the error correction repaired: stars of a one-ring code, star pairs of a bigger one.
    pub repaired: usize,
    /// Whether reading needed each star's own shape: the surface was far from flat.
    pub shaped: bool,
    /// Confidence of the least certain limb, 0 to 1.
    pub confidence: f64,
    /// The gap between the light and solid shades, in units of their spread.
    pub separation: f64,
    /// How inked the light limbs are relative to solid ones, as measured in this image.
    pub light_level: f64,
}

/// Settings for a scan.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Look for stars in a copy of the image shrunk so that its longer side is
    /// this many pixels. Quicker on large frames without a code, which suits a
    /// live preview, but small stars are missed. Reading always uses the full
    /// image. `None`, the default, looks at full resolution.
    pub find_size: Option<usize>,
}

/// One image plane being scanned, with the polarity being tried.
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    pub image: &'a Gray,
    /// `1.0` for dark ink on a light background, `-1.0` for light ink on dark.
    pub polarity: f64,
}

/// Where each star of a set is in the image, and its local map: how model
/// offsets around its centre appear in the image.
#[derive(Clone)]
pub(crate) struct Placement {
    pub at: Vec<Point>,
    pub maps: Vec<LocalMap>,
}

impl Placement {
    /// Stars exactly where a flat fit puts them.
    pub fn flat(h: &Homography, centers: &[Point]) -> Placement {
        Placement { at: centers.iter().map(|&c| h.apply(c)).collect(), maps: centers.iter().map(|&c| h.local_map(c)).collect() }
    }

    /// The first `n` stars.
    pub fn first(&self, n: usize) -> Placement {
        Placement { at: self.at[..n].to_vec(), maps: self.maps[..n].to_vec() }
    }
}

/// What one frame has cost so far, across its colour planes.
///
/// A real ring turns up early. In a code with many rings there are hundreds of
/// look-alike loops, and in a frame whose code cannot be read the scanner
/// would otherwise work through all of them.
#[derive(Default)]
pub(crate) struct Budget {
    /// Rings that passed the first checks.
    plausible: usize,
    /// Ring 0s that announced more rings. Once one has, no other loop in the
    /// frame can be the answer.
    headers: usize,
    /// Some part of the frame offered more plausible rings than a code would.
    gave_up: bool,
    /// The scanner has given up on this frame.
    stop: bool,
    /// Parts of the frame that need no more looking at, as centre and radius in pixels: a
    /// code whose ring 0 was found but whose rings could not be read, or a crowd of stars
    /// with no readable ring among them. Another code in the frame may still be read.
    spent: Vec<(Point, f64)>,
    /// Rings given the bent-surface treatment.
    bent: usize,
    /// The frame offered so many loops of twelve that it must be full of texture or noise.
    textured: bool,
}

/// Loops of twelve looked at in one pass (one binarised image, at one level of
/// crowding) before moving on to the next. Each pass has its own allowance: a
/// texture that fills one pass with chance loops must not keep the scanner
/// from the pass that has the code's stars.
const MAX_LOOPS: usize = 100;
/// Loops that may pass the first checks, which is where the work starts,
/// before giving up: on the crowd of stars they are in, if they are in one
/// (a code that cannot be read; there may be another in the frame), or else on the frame.
const MAX_PLAUSIBLE: usize = 30;
/// A plane that offers more loops than this in all is texture or noise, and
/// its colour channels are not worth trying one by one.
const TEXTURED_LOOPS: usize = 600;

/// Scan one grey plane.
fn scan_plane(gray: &Gray, options: &Options, budget: &mut Budget) -> Vec<Code> {
    if gray.width < 32 || gray.height < 32 {
        return Vec::new();
    }
    // Stars are looked for in `small`; `scale` takes its pixels back to the full image.
    let scale = match options.find_size {
        Some(size) => (size as f64 / gray.width.max(gray.height) as f64).min(1.0),
        None => 1.0,
    };
    let shrunk;
    let small = if scale < 1.0 {
        shrunk = gray.shrink(scale);
        &shrunk
    } else {
        gray
    };
    let passes: Vec<(f64, Vec<blobs::Blob>, rings::Neighbourhood)> = blobs::binarise(small)
        .into_iter()
        .map(|(polarity, blobs)| {
            let neighbourhood = rings::Neighbourhood::new(&blobs);
            (polarity, blobs, neighbourhood)
        })
        .collect();

    let mut results: Vec<Code> = Vec::new();
    let mut tried: Vec<candidate::Marks> = Vec::new();
    let mut loops_in_all = 0usize;
    let middle_of_frame = [gray.width as f64 / 2.0, gray.height as f64 / 2.0];
    // First only blobs with few similar neighbours, which is quick and finds a code on
    // its own; then more crowded ones; last everything, which a code with several
    // rings needs, since all its stars are crowded.
    'stages: for max_crowd in [rings::CROWD, 2 * rings::CROWD, usize::MAX] {
        for (polarity, blobs, neighbourhood) in &passes {
            let view = View { image: gray, polarity: *polarity };
            // The allowances are counted afresh for each crowd of stars, and for the blobs in none.
            let (mut loops, mut plausible_before, mut finished) = (0usize, budget.plausible, false);
            let mut crowd: Option<(Point, f64)> = None;
            rings::find_rings(neighbourhood, max_crowd, [small.width as f64 / 2.0, small.height as f64 / 2.0], |found| {
                let ring = match found {
                    rings::Found::Group(group) => {
                        crowd = group.map(|c| ([c.middle[0] / scale, c.middle[1] / scale], c.reach / scale));
                        (loops, plausible_before) = (0, budget.plausible);
                        // A crowd already given up on, as another pass saw it.
                        let spent = crowd.is_some_and(|(middle, _)| budget.spent.iter().any(|&(at, reach)| distance(at, middle) < reach));
                        return if spent { rings::Next::Leave } else { rings::Next::Go };
                    }
                    rings::Found::Ring(ring) => ring,
                };
                loops += 1;
                loops_in_all += 1;
                budget.textured |= loops_in_all > TEXTURED_LOOPS;
                if budget.stop {
                    finished = true;
                    return rings::Next::Stop;
                }
                let plausible = budget.plausible - plausible_before;
                if loops > MAX_LOOPS || plausible > MAX_PLAUSIBLE {
                    // Texture or noise, or a code that cannot be read. Leave it: the code's
                    // stars may be in another crowd, or in another pass.
                    budget.gave_up |= plausible > MAX_PLAUSIBLE;
                    match crowd.take() {
                        Some(zone) => budget.spent.push(zone),
                        None if plausible > MAX_PLAUSIBLE => {
                            budget.stop = true;
                            finished = true;
                            return rings::Next::Stop;
                        }
                        None => {}
                    }
                    return rings::Next::Leave;
                }
                let centres: Vec<Point> = ring.blobs.iter().map(|&i| [blobs[i as usize].x / scale, blobs[i as usize].y / scale]).collect();
                let areas: Vec<f64> = ring.blobs.iter().map(|&i| blobs[i as usize].area / (scale * scale)).collect();
                let count = centres.len() as f64;
                let middle = [centres.iter().map(|c| c[0]).sum::<f64>() / count, centres.iter().map(|c| c[1]).sum::<f64>() / count];
                if budget.spent.iter().any(|&(at, reach)| distance(at, middle) < reach) {
                    return rings::Next::Go;
                }
                let headers_before = budget.headers;
                match candidate::read_candidate(view, &centres, &ring.slots, &areas, &mut tried, budget) {
                    Some(code) => {
                        if results.iter().all(|r| r.payload != code.payload) {
                            results.push(code);
                        }
                    }
                    // Ring 0 of a code whose other rings could not be read. No other loop
                    // of that code can be the answer; a loop elsewhere in the frame can.
                    None if budget.headers > headers_before => {
                        let in_crowd = crowd.is_some();
                        let radius = centres.iter().map(|&c| distance(c, middle)).fold(0.0, f64::max);
                        budget.spent.push(crowd.take().unwrap_or((middle, 3.0 * radius)));
                        return if in_crowd { rings::Next::Leave } else { rings::Next::Go };
                    }
                    None => {}
                }
                // In the slow pass one code is enough.
                if !results.is_empty() && max_crowd == usize::MAX {
                    finished = true;
                    return rings::Next::Stop;
                }
                rings::Next::Go
            });
            if finished {
                break 'stages;
            }
        }
        // A code over the middle of the frame is the one being pointed at. With only codes
        // off to the side so far, a bigger one in the middle may still be waiting for the
        // slower stages.
        if results.iter().any(|code| inside(&code.corners, middle_of_frame)) || budget.stop {
            break;
        }
    }
    // The code nearest the middle of the frame first.
    let from_middle = |code: &Code| {
        let centre = [0, 1].map(|k| code.corners.iter().map(|c| c[k]).sum::<f64>() / 4.0);
        distance(centre, middle_of_frame)
    };
    results.sort_by(|a, b| from_middle(a).total_cmp(&from_middle(b)));
    results
}

/// Find and read the codes in a grey image.
///
/// With several codes in the image, the one nearest its middle comes first: that
/// is the one a camera is being pointed at. A code with several rings ends the
/// search, so others further from the middle may go unreported.
pub fn scan_gray(gray: &Gray, options: &Options) -> Vec<Code> {
    scan_plane(gray, options, &mut Budget::default())
}

/// Find and read the codes in an 8-bit RGB image (`r, g, b, r, g, b, ...`, row by row).
///
/// Brightness is tried first. If that finds nothing, each colour channel is
/// tried on its own, for ink and background colours that are alike in brightness.
/// The order of the codes is that of [`scan_gray`].
pub fn scan_rgb8(width: usize, height: usize, pixels: &[u8], options: &Options) -> Vec<Code> {
    let mut budget = Budget::default();
    let results = scan_plane(&Gray::from_rgb8(width, height, pixels), options, &mut budget);
    if !results.is_empty() {
        return results;
    }
    for channel in [2, 1, 0] {
        // If brightness showed a ring 0 whose outer rings could not be read, or a crowd of
        // stars with no readable ring among them, the single channels will not do better.
        if budget.headers > 0 || budget.gave_up || budget.stop || budget.textured {
            break;
        }
        let results = scan_plane(&Gray::from_rgb8_channel(width, height, pixels, channel), options, &mut budget);
        if !results.is_empty() {
            return results;
        }
    }
    Vec::new()
}
