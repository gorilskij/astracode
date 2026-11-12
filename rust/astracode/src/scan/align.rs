//! Lining the model up with the image: rotation, star positions, local maps.

use nalgebra::{Matrix3, Matrix3x2};

use super::patterns::{outline_flat, patterns};
use super::{Placement, View};
use crate::math::{add, offset, sub, Homography, Point};

/// Pick which of the twelve ring rotations has the stars upright, starting
/// from a fit `h0` of ring 0 that is right only up to such a rotation.
///
/// Turning the ring by one star position tilts every star by 30 degrees. Five
/// or seven positions leave them only 6 degrees off upright, so this compares
/// the whole star outline, which notices that, rather than a few points per limb.
///
/// Returns the turned fit and a score for how star-like the twelve places
/// are: the correlation between the levels sampled over each star and "ink on
/// the core and limbs, background in the gaps". Above about 0.3 for a real ring.
pub(crate) fn orient(view: View, h0: &Homography) -> (Homography, f64) {
    let p = patterns();
    let turned = super::par::map(12, |steps| {
        let h = h0.after(&Homography::ring_rotation(steps as i32));
        let score: f64 = p.ring0().iter().map(|&c| view.polarity * outline_flat(view.image, &h, c, 1.0)).sum();
        (score, h)
    });
    // The first of equal scores, as a plain loop over the rotations would give.
    let best = turned.iter().fold(turned[0], |best, &t| if t.0 > best.0 { t } else { best });
    (best.1, star_score(view, &best.1))
}

/// Correlation between ring 0's sampled levels and the pattern every star
/// shares whatever its data: ink on the core and limbs, background in the gaps.
fn star_score(view: View, h: &Homography) -> f64 {
    let p = patterns();
    let inked = p.star.len() - 5 * super::patterns::GAP_POINTS;
    let mut levels = Vec::with_capacity(12 * p.star.len());
    let mut target = Vec::with_capacity(12 * p.star.len());
    for &c in p.ring0() {
        for (i, g) in p.star.iter().enumerate() {
            let q = h.apply([c[0] + g[0], c[1] + g[1]]);
            levels.push(view.image.sample(q[0], q[1]) as f64);
            target.push(if i < inked { 1.0 } else { 0.0 });
        }
    }
    let n = levels.len() as f64;
    let (ma, mb) = (levels.iter().sum::<f64>() / n, target.iter().sum::<f64>() / n);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (a, b) in levels.iter().zip(&target) {
        ab += (a - ma) * (b - mb);
        aa += (a - ma) * (a - ma);
        bb += (b - mb) * (b - mb);
    }
    -view.polarity * ab / ((aa * bb).sqrt() + 1e-9)
}

/// Re-locate stars by matching the star outline around where the flat fit `h`
/// puts them, trying each offset in `search`. Returns their image positions.
pub(crate) fn locate(view: View, h: &Homography, centers: &[Point], search: &[Point]) -> Vec<Point> {
    // The outline grid has a pitch of 1.5 units and the search offsets are whole units, so
    // every point of every trial lies on a half-unit lattice around the star. Sample that
    // lattice once per star; each trial is then a weighted sum of values already read.
    let p = patterns();
    let furthest = |points: &[Point]| points.iter().map(|o| o[0].abs().max(o[1].abs())).fold(0.0, f64::max);
    let reach = furthest(&p.grid) + furthest(search);
    let side = (4.0 * reach).round() as usize + 1; // half-unit steps from -reach to +reach
    let cell = |v: f64| ((v + reach) * 2.0).round() as usize;
    let grid_cells: Vec<usize> = p.grid.iter().map(|g| cell(g[1]) * side + cell(g[0])).collect();
    super::par::map(centers.len(), |star| {
        let c = centers[star];
        let (pos, map) = (h.apply(c), h.local_map(c));
        let mut patch = vec![0f32; side * side];
        for (iy, row) in patch.chunks_mut(side).enumerate() {
            for (ix, v) in row.iter_mut().enumerate() {
                let q = add(pos, offset(&map, [0.5 * ix as f64 - reach, 0.5 * iy as f64 - reach]));
                *v = view.image.sample(q[0], q[1]);
            }
        }
        let scores: Vec<f64> = search
            .iter()
            .map(|o| {
                // Moving the outline by an offset moves every lattice index by the same amount.
                let shift = (2.0 * o[1]).round() as isize * side as isize + (2.0 * o[0]).round() as isize;
                let sum: f32 = grid_cells.iter().zip(&p.weights).map(|(&i, w)| w * patch[(i as isize + shift) as usize]).sum();
                -view.polarity * sum as f64
            })
            .collect();
        let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).expect("a search pattern");
        // Refine below the search step: the centroid of the scores next to the best one.
        let near: Vec<usize> = (0..search.len())
            .filter(|&i| (search[i][0] - search[best][0]).abs().max((search[i][1] - search[best][1]).abs()) <= 1.01)
            .collect();
        let floor = near.iter().map(|&i| scores[i]).fold(f64::INFINITY, f64::min);
        let (mut wx, mut wy, mut total) = (0.0, 0.0, 0.0);
        for &i in &near {
            let w = scores[i] - floor + 1e-6;
            wx += w * search[i][0];
            wy += w * search[i][1];
            total += w;
        }
        h.apply([c[0] + wx / total, c[1] + wy / total])
    })
}

/// Re-locate ring 0's stars and refit the homography to them.
pub(crate) fn refine(view: View, h: &Homography) -> Homography {
    let p = patterns();
    Homography::fit(p.ring0(), &locate(view, h, p.ring0(), &p.search))
}

/// How [`local_fit`] weighs a star's neighbours.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Neighbours {
    /// Ring 0 alone: a star has neighbours only along the ring, so the slope is
    /// held towards zero and the fit is little more than a weighted average of
    /// the neighbours' offsets.
    Ring,
    /// Several rings: every star is surrounded, and a tight neighbourhood with
    /// a free slope follows strong bending.
    Surrounded,
}

/// Where and how to read each star on a surface that may not be flat.
///
/// Around each star, how far its neighbours were found from the flat fit `h`
/// is fitted with a small linear model: its constant term moves the star, its
/// slope adjusts the star's local scale and rotation. One star's measured
/// position is noisy; a neighbourhood's is not. This is what a QR reader does
/// with alignment patterns, with every star serving as one.
pub(crate) fn local_fit(h: &Homography, centers: &[Point], found: &[Point], neighbours: Neighbours) -> Placement {
    // Neighbourhood size in model units, and the pull of the slope towards zero.
    let (sigma, ridge) = match neighbours {
        Neighbours::Ring => (45.0, 3000.0),
        Neighbours::Surrounded => (28.0, 30.0),
    };
    let flat = Placement::flat(h, centers);
    let residual: Vec<Point> = found.iter().zip(&flat.at).map(|(&f, &p)| sub(f, p)).collect();
    let reach2 = (4.5 * sigma) * (4.5 * sigma); // beyond this a neighbour's weight is nothing
    let mut out = flat.clone();
    for (m, &c) in centers.iter().enumerate() {
        let mut a = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(0.0, ridge, ridge));
        let mut b = Matrix3x2::<f64>::zeros();
        for (n, &other) in centers.iter().enumerate() {
            let d = sub(other, c);
            let d2 = d[0] * d[0] + d[1] * d[1];
            if d2 > reach2 {
                continue;
            }
            let w = (-d2 / (2.0 * sigma * sigma)).exp();
            let x = nalgebra::Vector3::new(1.0, d[0], d[1]);
            a += w * x * x.transpose();
            b += w * x * nalgebra::RowVector2::new(residual[n][0], residual[n][1]);
        }
        let Some(coef) = a.lu().solve(&b) else { continue };
        out.at[m] = [flat.at[m][0] + coef[(0, 0)], flat.at[m][1] + coef[(0, 1)]];
        // Slope row j, column i: how image coordinate i of the offset changes per model unit along j.
        out.maps[m] += nalgebra::Matrix2::new(coef[(1, 0)], coef[(2, 0)], coef[(1, 1)], coef[(2, 1)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{distance, offset};

    #[test]
    fn a_local_fit_follows_a_bend_that_a_flat_fit_cannot() {
        // Three rings on a surface with a smooth wave across it.
        let centers: Vec<Point> = patterns().rings[..3].concat();
        let h = Homography::scale_then_shift(1.4, 1.4, 500.0, 400.0);
        let bend = |p: Point| [p[0] + 12.0 * (p[1] / 150.0).sin(), p[1] + 9.0 * (p[0] / 120.0 + 1.0).sin()];
        let found: Vec<Point> = centers.iter().map(|&c| h.apply(bend(c))).collect();
        let fit = local_fit(&Homography::fit(&centers, &found), &centers, &found, Neighbours::Surrounded);
        for (i, &c) in centers.iter().enumerate() {
            // The star is placed where it was found...
            assert!(distance(fit.at[i], found[i]) < 1.0, "star {i} placed {} away", distance(fit.at[i], found[i]));
            // ...and its local map carries a small step to the right place.
            let step = [3.0, -2.0];
            let truth = sub(h.apply(bend([c[0] + step[0], c[1] + step[1]])), found[i]);
            assert!(distance(offset(&fit.maps[i], step), truth) < 0.25);
        }
    }
}
