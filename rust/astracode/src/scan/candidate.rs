//! One candidate ring, from twelve blobs to a result.
//!
//! This is where the scanner decides what a ring is: not a code at all, a
//! one-ring code, or ring 0 of a bigger code (in which case [`super::outer`]
//! takes over). The order of work is cheapest first, and flat before bent.

use super::align::{local_fit, locate, orient, refine, Neighbours};
use super::bent::{bent_ring0, mirrored as mirror_placement, with_ring1, RingBlobs};
use super::checks::{band_clear, centre_is_code, quiet, stars_fit};
use super::limbs::{classify, contrast, limb_strength, read_levels, Reading};
use super::outer::read_rings;
use super::patterns::patterns;
use super::{Budget, Code, Payload, Placement, View, MIN_SHADE_SEPARATION};
use crate::format::ring0::{self, Symbols};
use crate::format::{half_side, R};
use crate::math::{distance, Homography, LocalMap, Point};

// The check stars repair one misread star anywhere, or else the two least certain
// stars. Refuse to answer when too many stars have a limb close to the light/solid
// threshold, or when the two shades are not clearly apart: that is when several
// stars can be wrong at once and a repair would be a guess.

/// A limb closer to the light/solid threshold than this, as a fraction of half
/// the gap between the shades, makes its star weak.
const WEAK_CONFIDENCE: f64 = 0.25;
/// More weak stars than this and ring 0 is not read at all.
const MAX_WEAK_STARS: usize = 3;
/// What every other star must reach before the two least certain may be rebuilt.
const CLEAR_CONFIDENCE: f64 = 0.4;
/// Least contrast, in grey levels, between a star's centre and its background.
const MIN_CONTRAST: f64 = 6.0;
/// Rings per frame that get the bent-surface treatment.
const MAX_BENT: usize = 4;

/// Three model points whose image positions identify a placement of ring 0.
const MARKS: [Point; 3] = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]];

/// Where a candidate's marks landed; see [`read_candidate`].
pub(crate) type Marks = [Point; 3];

/// Ring 0 as read: its twelve symbols and what is known about their reliability.
struct Ring0 {
    symbols: Symbols,
    /// The two least certain stars, if every other star is clear enough for
    /// those two to be rebuilt by the check stars.
    doubtful: Option<(usize, usize)>,
    reading: Reading,
}

/// Read ring 0's limbs with its stars placed as given. `None` if the stars do
/// not stand out, or the two shades cannot be told apart well enough.
fn read_ring0(view: View, stars: &Placement, mirrored: bool) -> Option<Ring0> {
    let levels = read_levels(view.image, stars);
    let reading = classify(&limb_strength(&levels, view.polarity), patterns().ring0(), false);
    let faintest = contrast(&levels, view.polarity).into_iter().fold(f64::INFINITY, f64::min);
    if faintest < MIN_CONTRAST || !(0.3 < reading.ratio && reading.ratio < 0.97) {
        return None;
    }
    let confidence = reading.star_confidence();
    if reading.separation < MIN_SHADE_SEPARATION || confidence.iter().filter(|&&c| c < WEAK_CONFIDENCE).count() > MAX_WEAK_STARS {
        return None;
    }
    // Rebuilding the two least certain stars is only safe when every other star is clear,
    // and never for a mirrored reading, which must need no repair at all.
    let mut order: Vec<usize> = (0..12).collect();
    order.sort_by(|&a, &b| confidence[a].total_cmp(&confidence[b]));
    let doubtful = (!mirrored && confidence[order[2]] >= CLEAR_CONFIDENCE).then_some((order[0], order[1]));
    let symbols: Symbols = reading.symbols().try_into().expect("ring 0 has twelve stars");
    Some(Ring0 { symbols, doubtful, reading })
}

/// The bent-surface fit of a candidate, worked out at most once and shared by
/// its plain and mirrored readings.
struct Bent<'a> {
    /// The candidate's flat fit, not mirrored.
    h: Homography,
    blobs: RingBlobs<'a>,
    fit: Option<Option<(Homography, Placement)>>,
}

impl Bent<'_> {
    fn get(&mut self, view: View, mirrored: bool, budget: &mut Budget) -> Option<(Homography, Placement)> {
        if self.fit.is_none() {
            budget.bent += 1;
            self.fit = Some(if budget.bent <= MAX_BENT { bent_ring0(view, &self.h, self.blobs) } else { None });
        }
        let (h, stars) = self.fit.as_ref()?.as_ref()?;
        Some(if mirrored { (h.after(&Homography::mirror()), mirror_placement(stars)) } else { (*h, stars.clone()) })
    }
}

/// Read ring 0 under the fit `h` (already mirrored if `mirrored`) and, if it
/// is the header of a bigger code, the rest of the code.
fn read_oriented(view: View, h: &Homography, mirrored: bool, budget: &mut Budget, bent: &mut Bent) -> Option<Code> {
    let p = patterns();
    let mut h = *h;
    let mut found = locate(view, &h, p.ring0(), &p.search);
    let mut shapes: Option<Vec<LocalMap>> = None;

    // As a flat ring first.
    let mut read = read_ring0(view, &local_fit(&h, p.ring0(), &found, Neighbours::Ring), mirrored);
    let mut number = read.as_ref().and_then(|r| ring0::decode(&r.symbols, r.doubtful));
    let mut header = if number.is_none() { read.as_ref().and_then(|r| ring0::read_header(&r.symbols, r.doubtful)) } else { None };

    if number.is_none() && header.is_none() {
        // Not readable as a flat ring. Try it as a bent one: place and shape each star on
        // its own; then, if a ring 1 surrounds it, use that ring's stars as well.
        let (turned, stars) = bent.get(view, mirrored, budget)?;
        h = turned;
        found = stars.at.clone();
        shapes = Some(stars.maps.clone());
        for attempt in 0..2 {
            let placed = if attempt == 0 { Some(stars.clone()) } else { with_ring1(view, &h, &found, shapes.as_deref()) };
            let Some(ring) = placed.and_then(|stars| read_ring0(view, &stars, mirrored)) else { continue };
            // No rebuilding of two doubtful stars here: this path gives a ring several
            // chances to be read, and that repair is the easiest for noise to pass.
            number = ring0::decode(&ring.symbols, None);
            header = if number.is_none() { ring0::read_header(&ring.symbols, None) } else { None };
            read = Some(ring);
            if number.is_some() || header.is_some() {
                break;
            }
        }
    }

    let Some((value, repaired)) = number else {
        let header = header?;
        if mirrored && header.repaired > 0 {
            return None;
        }
        budget.headers += 1;
        return read_rings(view, &h, mirrored, &header, &found, shapes.as_deref());
    };
    // A one-ring code has clear margin outside its stars and, above all, no ring 1. Ring 0
    // of a bigger code read with a star "repaired" can pass for a one-ring code otherwise.
    if (mirrored && repaired > 0) || !quiet(view, &h) || with_ring1(view, &h, &found, shapes.as_deref()).is_some() {
        return None;
    }
    let reading = read?.reading;
    let half = half_side(1);
    Some(Code {
        payload: Payload::Number(value),
        rings: 1,
        corners: [[-half, -half], [half, -half], [half, half], [-half, half]].map(|c| h.apply(c)),
        mirrored,
        repaired,
        shaped: shapes.is_some(),
        confidence: reading.star_confidence().into_iter().fold(f64::INFINITY, f64::min),
        separation: reading.separation,
        light_level: reading.ratio,
    })
}

/// Take a ring of blobs through the checks and, if it survives, read it.
///
/// * `centres`, `areas`: the blobs found, in image pixels.
/// * `slots`: which of the ring's twelve places each blob fills.
/// * `tried`: placements already read in this image. The same ring turns up
///   many times, in each binarised image and with different stars missing;
///   it is read once.
pub(crate) fn read_candidate(
    view: View,
    centres: &[Point],
    slots: &[u8],
    areas: &[f64],
    tried: &mut Vec<Marks>,
    budget: &mut Budget,
) -> Option<Code> {
    // Fit the twelve places of a ring to the blobs. Blob centres sit on the star
    // centres, so a poor fit is not a ring.
    let model: Vec<Point> = slots.iter().map(|&s| (30.0 * s as f64).to_radians()).map(|a| [R * a.cos(), R * a.sin()]).collect();
    let h0 = Homography::fit(&model, centres);
    let back = h0.inverse()?;
    if centres.iter().zip(&model).any(|(&c, &m)| distance(back.apply(c), m) > 9.0) {
        return None;
    }
    if !band_clear(view, &h0) {
        return None;
    }
    budget.plausible += 1;

    let (h, star_score) = orient(view, &h0);
    if star_score < 0.3 {
        return None;
    }
    let marks = MARKS.map(|m| h.apply(m));
    let tolerance = 3.0 * distance(marks[1], marks[0]) / 100.0; // three model units
    if tried.iter().any(|seen| (0..3).all(|i| (marks[i][0] - seen[i][0]).abs().max((marks[i][1] - seen[i][1]).abs()) < tolerance)) {
        return None;
    }
    tried.push(marks);

    let h = refine(view, &h);
    if !stars_fit(view, &h) || centre_is_code(view, &h) {
        return None;
    }
    let h = refine(view, &refine(view, &h));

    // A mirror image also shows twelve upright stars; it is the same ring read
    // anticlockwise with each star's limbs swapped left to right.
    let mut bent = Bent { h, blobs: RingBlobs { centres, areas }, fit: None };
    read_oriented(view, &h, false, budget, &mut bent)
        .or_else(|| read_oriented(view, &h.after(&Homography::mirror()), true, budget, &mut bent))
}
