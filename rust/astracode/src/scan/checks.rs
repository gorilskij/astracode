//! Cheap tests that tell ring 0 of a real code from its look-alikes.
//!
//! The ring finder accepts any twelve similar blobs on an ellipse. Most false
//! rings are thrown out by these before any reading is attempted.

use super::patterns::{outline_flat, patterns};
use super::View;
use crate::math::{distance, median, Homography, Point};

/// Share of the standard ink contrast above which a sample counts as inked.
const INKED: f64 = 0.5;

/// Levels at model points under the flat fit `h`.
fn levels(view: View, h: &Homography, points: &[Point]) -> Vec<f64> {
    points.iter().map(|&p| h.apply(p)).map(|q| view.image.sample(q[0], q[1]) as f64).collect()
}

/// How inked each level is: 0 at `background`, 1 at the level of the stars' centres.
fn ink(view: View, h: &Homography, values: &[f64], background: f64) -> Vec<f64> {
    let core = levels(view, h, patterns().ring0());
    let core = core.iter().sum::<f64>() / core.len() as f64;
    let scale = background - core + view.polarity * 1e-3;
    values.iter().map(|v| (background - v) / scale).collect()
}

/// The band inside ring 0, and the gaps between its stars, must be background.
///
/// In a code with several rings, loops of twelve stars turn up that are not
/// ring 0; most of them have stars where this band should be.
pub(crate) fn band_clear(view: View, h: &Homography) -> bool {
    let p = patterns();
    let (band, between) = (levels(view, h, &p.band), levels(view, h, &p.between));
    let background = median(&[band.as_slice(), between.as_slice()].concat());
    let band_inked = ink(view, h, &band, background).iter().filter(|&&i| i > INKED).count();
    let between_inked = ink(view, h, &between, background).iter().filter(|&&i| i > INKED).count();
    band_inked as f64 <= 0.1 * band.len() as f64 && between_inked <= 1
}

/// A one-ring code has clear margin outside its stars; ring 0 of a bigger
/// code, or a look-alike loop, does not.
pub(crate) fn quiet(view: View, h: &Homography) -> bool {
    let p = patterns();
    let background = median(&levels(view, h, &p.between));
    let margin = levels(view, h, &p.quiet);
    let inked = ink(view, h, &margin, background).iter().filter(|&&i| i > INKED).count();
    inked as f64 <= 0.1 * margin.len() as f64
}

/// Total outline match of ring 0's twelve stars drawn `scale` times their standard size.
fn ring_outline(view: View, h: &Homography, scale: f64) -> f64 {
    patterns().ring0().iter().map(|&c| view.polarity * outline_flat(view.image, h, c, scale)).sum()
}

/// Are the stars the right size for this ring?
///
/// Deep inside a code with many rings, a star's twelve second-nearest
/// neighbours form a loop that passes for ring 0, except that the loop is too
/// small for its stars. Compare the star outline at three sizes.
pub(crate) fn stars_fit(view: View, h: &Homography) -> bool {
    let (small, right, big) = (ring_outline(view, h, 0.85), ring_outline(view, h, 1.0), ring_outline(view, h, 1.18));
    right > 0.0 && (big - small) / right <= 0.06
}

/// Is the middle of this loop itself part of a code: a star just like the
/// loop's own, with more around it?
///
/// That is what the look-alike loops deep inside a big code have. Any other
/// picture is fine there, including a single star of any size.
pub(crate) fn centre_is_code(view: View, h: &Homography) -> bool {
    let p = patterns();
    // The loop's stars may be larger than the model expects (see `stars_fit`); use the size that fits them best.
    let sizes = [1.0, 1.2, 1.4];
    let ring = sizes.map(|k| ring_outline(view, h, k) / 12.0);
    let best = (0..3).max_by(|&a, &b| ring[a].total_cmp(&ring[b])).expect("three sizes");
    let (size, need) = (sizes[best], 0.7 * ring[best]);
    let score = |at: Point| view.polarity * outline_flat(view.image, h, at, size);
    if need <= 0.0 || !p.inner_near.iter().any(|&at| score(at) >= need) {
        return false; // no star like the ring's at the centre
    }
    // Count distinct stars: strongest first, at least 25 units apart.
    let mut scored: Vec<(f64, Point)> =
        super::par::map(p.inner_all.len(), |i| (score(p.inner_all[i]), p.inner_all[i])).into_iter().filter(|s| s.0 >= need).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut stars: Vec<Point> = Vec::new();
    for (_, at) in scored {
        if stars.iter().all(|&s| distance(s, at) >= 25.0) {
            stars.push(at);
        }
    }
    stars.len() >= 5
}
