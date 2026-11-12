//! Reading a code on a surface that is far from flat, from the stars' own shapes.
//!
//! Every star is the same upright pentagram, so each star's outline in the
//! photo says how the surface is stretched, sheared and turned right there
//! ([`fit_shapes`]). The shapes also settle which way is up ([`twist`]), which
//! the flat method gets wrong when the surface twists the stars themselves.

use super::align::{local_fit, locate, Neighbours};
use super::limbs::{contrast, read_levels};
use super::outer::locate_rings;
use super::patterns::{outline_local, patterns};
use super::{Placement, View};
use crate::math::{add, distance, median, offset, sub, Homography, LocalMap, Point};

/// Fit each star's own position, stretch and shear to the image.
///
/// Each star's local map is adjusted, one term at a time, to where the star
/// outline matches best, and its centre likewise; first in coarse steps, then
/// finer ones.
pub(crate) fn fit_shapes(view: View, stars: &Placement) -> Placement {
    /// Relative changes tried on each term of the map, and the centre shift in model units, per pass.
    const PASSES: [(&[f64], f64); 4] =
        [(&[-0.12, -0.06, 0.06, 0.12], 1.0), (&[-0.12, -0.06, 0.06, 0.12], 1.0), (&[-0.03, 0.03], 0.5), (&[-0.015, 0.015], 0.25)];
    // With hundreds of stars, a third of the outline points will do.
    let thin = stars.at.len() > 200;
    let score = |pos: Point, map: &LocalMap| view.polarity * outline_local(view.image, pos, map, thin);

    let fit_one = |i: usize| -> (Point, LocalMap) {
        let (mut pos, mut map) = (stars.at[i], stars.maps[i]);
        let mut best = score(pos, &map);
        for (steps, shift) in PASSES {
            // Stretch along x, along y, then the two shears.
            for (row, col) in [(0, 0), (1, 1), (0, 1), (1, 0)] {
                for &step in steps {
                    let mut change = LocalMap::identity();
                    change[(row, col)] += step;
                    let trial = map * change;
                    let s = score(pos, &trial);
                    if s > best {
                        (map, best) = (trial, s);
                    }
                }
            }
            for delta in [[-shift, 0.0], [shift, 0.0], [0.0, -shift], [0.0, shift]] {
                let trial = add(pos, offset(&map, delta));
                let s = score(trial, &map);
                if s > best {
                    (pos, best) = (trial, s);
                }
            }
        }
        (pos, map)
    };
    let fitted = super::par::map(stars.at.len(), fit_one);
    Placement { at: fitted.iter().map(|f| f.0).collect(), maps: fitted.iter().map(|f| f.1).collect() }
}

/// How far, in degrees, ring 0's fitted stars are turned from what their
/// places on the ring imply.
///
/// Read at the right rotation this is about zero. Read `s` ring positions
/// round, every star is `30 s` degrees off upright; a pentagram looks the same
/// every 72 degrees, so a shape fit quietly settles on the nearest
/// upright-looking match, and each star's own map ends up turned against the
/// direction to its ring neighbours by `30 s` minus the nearest multiple of
/// 72: 6 degrees for five positions, -12 for two, 18 for three and so on, a
/// different multiple of 6 for every `s`. So the star shapes alone say which
/// rotation is right, even on a surface that twists the stars themselves by
/// more than that.
pub(crate) fn twist(stars: &Placement) -> f64 {
    let c = patterns().ring0();
    let angles: Vec<f64> = (0..12)
        .map(|i| {
            let (before, after) = ((i + 11) % 12, (i + 1) % 12);
            // From the star before to the star after: as found, and as this star's map predicts.
            let chord = sub(stars.at[after], stars.at[before]);
            let predicted = offset(&stars.maps[i], sub(c[after], c[before]));
            (predicted[0] * chord[1] - predicted[1] * chord[0]).atan2(predicted[0] * chord[0] + predicted[1] * chord[1])
        })
        .collect();
    // The median: a few badly fitted stars must not sway it.
    median(&angles).to_degrees()
}

/// Ring 0's placement renumbered for the ring read `steps` positions further round.
pub(crate) fn turn(stars: &Placement, steps: usize) -> Placement {
    let (s, c) = (30.0 * steps as f64).to_radians().sin_cos();
    let rotation = LocalMap::new(c, -s, s, c);
    Placement {
        at: (0..12).map(|j| stars.at[(j + steps) % 12]).collect(),
        maps: (0..12).map(|j| stars.maps[(j + steps) % 12] * rotation).collect(),
    }
}

/// The blobs a ring candidate was found from: centres and areas, in image pixels.
#[derive(Clone, Copy)]
pub(crate) struct RingBlobs<'a> {
    pub centres: &'a [Point],
    pub areas: &'a [f64],
}

/// Ring 0 on a surface that is far from flat, worked out from the stars themselves.
///
/// Starts each star from its blob (its centre, and its size relative to the
/// others) where there is one, fits each star's shape, and then checks the
/// rotation against the shapes; see [`twist`]. The homography comes back
/// turned if the rotation had to be corrected. `None` if the shapes do not
/// agree on a rotation.
pub(crate) fn bent_ring0(view: View, h: &Homography, blobs: RingBlobs) -> Option<(Homography, Placement)> {
    let p = patterns();
    let mut stars = Placement::flat(h, p.ring0());
    let flat = stars.at.clone();
    // Match each star to the blob nearest its flat-fit position.
    let mine: Vec<usize> = flat
        .iter()
        .map(|&f| {
            (0..blobs.centres.len())
                .min_by(|&a, &b| distance(f, blobs.centres[a]).total_cmp(&distance(f, blobs.centres[b])))
                .expect("a ring has blobs")
        })
        .collect();
    let mut has: Vec<bool> =
        (0..12).map(|i| distance(flat[i], blobs.centres[mine[i]]) < 0.3 * distance(flat[i], flat[(i + 11) % 12])).collect();
    if has.iter().filter(|&&x| x).count() >= 6 {
        // A blob's area against what the flat fit expects says how much larger or smaller the star is here.
        let size: Vec<f64> = (0..12).map(|i| (blobs.areas[mine[i]] / stars.maps[i].determinant().abs()).sqrt()).collect();
        let typical = median(&(0..12).filter(|&i| has[i]).map(|i| size[i]).collect::<Vec<f64>>());
        for i in (0..12).filter(|&i| has[i]) {
            stars.at[i] = blobs.centres[mine[i]];
            stars.maps[i] *= (size[i] / typical).clamp(0.6, 1.7);
        }
    } else {
        has.fill(false);
    }
    // Stars without a blob: look around where the flat fit puts them.
    let missing: Vec<usize> = (0..12).filter(|&i| !has[i]).collect();
    if !missing.is_empty() {
        let centers: Vec<Point> = missing.iter().map(|&i| p.ring0()[i]).collect();
        for (&i, at) in missing.iter().zip(locate(view, h, &centers, &p.search_wide)) {
            stars.at[i] = at;
        }
    }

    let mut h = *h;
    let mut stars = fit_shapes(view, &stars);
    for _ in 0..3 {
        let sixths = (twist(&stars) / 6.0).round() as i32;
        if sixths == 0 {
            return Some((h, stars));
        }
        // A twist of 6 q degrees means the ring is being read 5 q positions round.
        let steps = (-5 * sixths).rem_euclid(12);
        h = h.after(&Homography::ring_rotation(steps));
        stars = fit_shapes(view, &turn(&stars, steps as usize));
    }
    ((twist(&stars) / 6.0).round() == 0.0).then_some((h, stars))
}

/// A mirror image of a placement of ring 0: the same twelve fitted stars,
/// numbered the other way round. A pentagram is its own mirror image.
pub(crate) fn mirrored(stars: &Placement) -> Placement {
    let flip = LocalMap::new(-1.0, 0.0, 0.0, 1.0);
    Placement { at: (0..12).map(|j| stars.at[(12 - j) % 12]).collect(), maps: (0..12).map(|j| stars.maps[(12 - j) % 12] * flip).collect() }
}

/// If a ring 1 of stars surrounds ring 0, where and how to read ring 0 given both.
///
/// On a surface that bends, ring 0 alone cannot show how it is stretched
/// across the ring, only along it. The stars of ring 1 can. `None` when there
/// is no ring 1: this doubles as the test for "is this a one-ring code".
pub(crate) fn with_ring1(view: View, h: &Homography, ring0: &[Point], shapes: Option<&[LocalMap]>) -> Option<Placement> {
    let p = patterns();
    let found: Vec<Point> = locate_rings(view, h, 2, ring0, shapes).concat();
    let centers: Vec<Point> = p.rings[..2].concat();
    let stars = local_fit(&Homography::fit(&centers, &found), &centers, &found, Neighbours::Surrounded);
    let stand_out = contrast(&read_levels(view.image, &stars), view.polarity);
    let typical = median(&stand_out[..12]);
    let star_like = stand_out[12..].iter().filter(|&&c| c > 0.5 * typical).count();
    (star_like >= 15).then(|| stars.first(12))
}
