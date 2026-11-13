//! Stage 2: from blobs to rings of twelve.
//!
//! Ring 0 of a code shows up as twelve similar blobs, evenly spaced round an
//! ellipse. Under an affine map, ring star `j` sits at
//! `C + A cos(30 j) + B sin(30 j)`, which is linear in `C`, `A`, `B`; so the
//! next star is a fixed weighted sum of the last few. Starting from three
//! neighbouring blobs, [`find_rings`] walks clockwise round the ring by that
//! prediction until the walk closes onto its first two stars. A few stars may
//! be missing (merged with something, washed out): a slot with no blob near
//! its predicted place is carried as a virtual point.

use std::sync::OnceLock;

use nalgebra::{DMatrix, RowDVector};

use super::blobs::Blob;
use crate::math::Point;

/// Ring neighbours are among a blob's nearest similar blobs; this many are kept.
const NEAREST: usize = 8;
/// Ring-walk steps allowed per call of [`find_rings`].
const WALK_BUDGET: i64 = 300_000;
/// Stars of a ring that may have no blob.
const MAX_MISSING: usize = 3;
/// A blob with more similar neighbours than this is in a crowd: a code with
/// several rings, or a texture.
pub(crate) const CROWD: usize = 12;

/// Weights that predict the next ring position from the last `m` (3, 4 or 5),
/// oldest first. Exact from three; least squares, and so less sensitive to
/// noise, from four or five. Perspective bends the affine model over long
/// arcs, so no more than five are used.
fn predict_weights(m: usize) -> &'static [f64] {
    static WEIGHTS: OnceLock<[Vec<f64>; 3]> = OnceLock::new();
    let all = WEIGHTS.get_or_init(|| {
        [3usize, 4, 5].map(|m| {
            let row = |j: usize| {
                let a = (30.0 * j as f64).to_radians();
                [1.0, a.cos(), a.sin()]
            };
            let a = DMatrix::from_fn(m, 3, |j, c| row(j)[c]);
            let pinv = a.pseudo_inverse(1e-12).expect("three ring positions determine the model");
            (RowDVector::from_row_slice(&row(m)) * pinv).iter().copied().collect()
        })
    });
    &all[m - 3]
}

/// How far a blob may be from its predicted place, as a fraction of the last
/// step, when predicting from 3, 4 or 5 positions.
const TOLERANCE: [f64; 3] = [0.9, 0.6, 0.55];

/// The blobs of one binarised image with their neighbour lists, ready for [`find_rings`].
pub(crate) struct Neighbourhood {
    pts: Vec<Point>,
    /// Square root of each blob's area: its linear size.
    size: Vec<f64>,
    /// Each blob's nearest similar blobs, closest first.
    near: Vec<Vec<u32>>,
    /// The same over twice the distance, for stepping across a missing star.
    far: Vec<Vec<u32>>,
    /// How many similar blobs each blob has around it.
    crowd: Vec<usize>,
}

/// May two blobs be neighbours in a ring, going by their sizes?
fn similar(a: f64, b: f64) -> bool {
    a.max(b) < 2.3 * a.min(b)
}
/// Neighbours are looked for within this many times the larger blob's size, and for
/// the wider list within [`FAR`] times; nearer than [`TOUCHING`] they overlap.
const NEAR: f64 = 6.5;
const FAR: f64 = 13.0;
const TOUCHING: f64 = 0.8;

/// The `keep` closest of a list of blobs given as (distance squared, blob), closest
/// first and the lower index first among equals.
fn closest(mut list: Vec<(f64, u32)>, keep: usize) -> Vec<u32> {
    let order = |a: &(f64, u32), b: &(f64, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    if list.len() > keep {
        list.select_nth_unstable_by(keep - 1, order);
        list.truncate(keep);
    }
    list.sort_unstable_by(order);
    list.into_iter().map(|(_, blob)| blob).collect()
}

/// Blobs sorted into square cells, so that those near a place can be gone
/// through without looking at all the rest.
struct Cells {
    side: f64,
    left: f64,
    top: f64,
    across: usize,
    down: usize,
    /// Cell `c` holds `blobs[starts[c]..starts[c + 1]]`.
    starts: Vec<u32>,
    blobs: Vec<u32>,
}

impl Cells {
    fn new(pts: &[Point], side: f64) -> Cells {
        let low = |k: usize| pts.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
        let high = |k: usize| pts.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
        let (left, top) = if pts.is_empty() { (0.0, 0.0) } else { (low(0), low(1)) };
        let count = |from: f64, to: f64| if pts.is_empty() { 1 } else { ((to - from) / side) as usize + 1 };
        let (across, down) = (count(left, high(0)), count(top, high(1)));
        let mut cells = Cells { side, left, top, across, down, starts: vec![0; across * down + 1], blobs: vec![0; pts.len()] };
        // A counting sort: how many per cell, then where each cell starts, then the blobs in index order.
        let homes: Vec<usize> = pts.iter().map(|&p| cells.home(p)).map(|(x, y)| y * across + x).collect();
        for &home in &homes {
            cells.starts[home + 1] += 1;
        }
        for c in 0..across * down {
            cells.starts[c + 1] += cells.starts[c];
        }
        let mut next = cells.starts.clone();
        for (blob, &home) in homes.iter().enumerate() {
            cells.blobs[next[home] as usize] = blob as u32;
            next[home] += 1;
        }
        cells
    }

    /// The cell a point is in, as column and row.
    fn home(&self, p: Point) -> (usize, usize) {
        ((((p[0] - self.left) / self.side) as usize).min(self.across - 1), (((p[1] - self.top) / self.side) as usize).min(self.down - 1))
    }

    /// Call `f` with each blob in the cells exactly `ring` cells away from `home`
    /// (the square ring round it; `ring` 0 is the cell itself). False if the whole
    /// ring is outside the grid: there is nothing further out either.
    fn ring(&self, home: (usize, usize), ring: usize, mut f: impl FnMut(u32)) -> bool {
        let (hx, hy, r) = (home.0 as isize, home.1 as isize, ring as isize);
        let mut inside = false;
        for y in (hy - r).max(0)..=(hy + r).min(self.down as isize - 1) {
            let edge = (y - hy).abs() == r;
            let mut cell = |x: isize| {
                if x >= 0 && x < self.across as isize {
                    inside = true;
                    let c = y as usize * self.across + x as usize;
                    self.blobs[self.starts[c] as usize..self.starts[c + 1] as usize].iter().for_each(|&b| f(b));
                }
            };
            if edge {
                (hx - r..=hx + r).for_each(&mut cell);
            } else {
                cell(hx - r);
                cell(hx + r);
            }
        }
        inside
    }
}

impl Neighbourhood {
    pub fn new(blobs: &[Blob]) -> Neighbourhood {
        let n = blobs.len();
        let pts: Vec<Point> = blobs.iter().map(|b| [b.x, b.y]).collect();
        let size: Vec<f64> = blobs.iter().map(|b| b.area.sqrt()).collect();
        let mut sizes = size.clone();
        sizes.sort_by(f64::total_cmp);
        // Cells of a couple of blob sizes across, but no more cells than blobs.
        let spread =
            |k: usize| pts.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max) - pts.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
        let side = if n == 0 { 1.0 } else { (2.0 * sizes[n / 2]).max((spread(0) * spread(1) / n as f64).sqrt()).max(1.0) };
        let cells = Cells::new(&pts, side);

        // Each blob looks outwards from its own cell, a ring of cells at a time, and stops
        // when no blob further out could be a neighbour of either kind.
        let found: Vec<(Vec<u32>, Vec<u32>, usize)> = super::par::map(n, |i| {
            // The largest blob that could be this one's neighbour sets how far that is.
            let largest = sizes[sizes.partition_point(|&s| s < 2.3 * size[i]) - 1].max(size[i]);
            // Candidates as (distance squared, blob); squares throughout, so that no roots are taken.
            let (mut near, mut far) = (Vec::<(f64, u32)>::new(), Vec::<(f64, u32)>::new());
            let home = cells.home(pts[i]);
            for ring in 0.. {
                let inside = cells.ring(home, ring, |j| {
                    let j = j as usize;
                    // Similar in size, not overlapping, and within a few sizes of each other.
                    if j == i || !similar(size[i], size[j]) {
                        return;
                    }
                    let big = size[i].max(size[j]);
                    let (dx, dy) = (pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
                    let (d2, big2) = (dx * dx + dy * dy, big * big);
                    if d2 <= TOUCHING * TOUCHING * big2 || d2 >= FAR * FAR * big2 {
                        return;
                    }
                    far.push((d2, j as u32));
                    if d2 < NEAR * NEAR * big2 {
                        near.push((d2, j as u32));
                    }
                });
                // Every blob not yet looked at is further away than this.
                let covered = ring as f64 * cells.side;
                if !inside || covered >= FAR * largest {
                    break;
                }
                // Near neighbours all seen, and enough far ones closer than anything still to come?
                if covered >= NEAR * largest && far.iter().filter(|f| f.0 < covered * covered).count() >= 3 * NEAREST {
                    break;
                }
            }
            let crowd = near.len();
            (closest(near, NEAREST), closest(far, 3 * NEAREST), crowd)
        });
        let (mut near, mut far, mut crowd) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for (n, f, c) in found {
            near.push(n);
            far.push(f);
            crowd.push(c);
        }
        Neighbourhood { near, far, pts, size, crowd }
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }
}

/// A closed ring of twelve places, some of which have a blob.
pub(crate) struct Ring {
    /// Indices of the blobs found, in walking order.
    pub blobs: Vec<u32>,
    /// Which of the twelve places (0..12, clockwise) each of those blobs fills.
    pub slots: Vec<u8>,
}

struct Walker<'a, F: FnMut(&[i32; 12]) -> bool> {
    nb: &'a Neighbourhood,
    budget: i64,
    /// Called with each closed ring (`-1` for a missing star); returns false to stop everything.
    found: F,
    stopped: bool,
}

impl<F: FnMut(&[i32; 12]) -> bool> Walker<'_, F> {
    /// Blobs of compatible size within `radius` of `at`, closest first, taken
    /// from the neighbours of the previous star (or of the one before it, after a gap).
    fn around(&self, at: Point, radius: f64, size: f64, last: i32, before: i32) -> Vec<u32> {
        let pool = if last >= 0 { &self.nb.near[last as usize] } else { &self.nb.far[before as usize] };
        let mut hits: Vec<(f64, u32)> = pool
            .iter()
            .filter_map(|&i| {
                let p = self.nb.pts[i as usize];
                let d2 = (p[0] - at[0]).powi(2) + (p[1] - at[1]).powi(2);
                let s = self.nb.size[i as usize];
                (d2 < radius * radius && size / 2.3 < s && s < size * 2.3).then_some((d2, i))
            })
            .collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits.into_iter().map(|(_, i)| i).collect()
    }

    /// Extend a partial ring by one place.
    fn walk(&mut self, ids: &mut Vec<i32>, pos: &mut Vec<Point>, missing: usize, size: f64) {
        self.budget -= 1;
        if self.budget < 0 || self.stopped {
            return;
        }
        let step = ids.len();
        let m = step.min(5);
        let weights = predict_weights(m);
        let recent = &pos[step - m..];
        let x: f64 = weights.iter().zip(recent).map(|(w, p)| w * p[0]).sum();
        let y: f64 = weights.iter().zip(recent).map(|(w, p)| w * p[1]).sum();
        let (last, before) = (pos[step - 1], pos[step - 2]);
        let tol = TOLERANCE[m - 3] * (last[0] - before[0]).hypot(last[1] - before[1]);

        if step >= 12 {
            // Back at the start: the prediction must land on the first two stars again.
            let q = pos[step - 12];
            if (q[0] - x).hypot(q[1] - y) <= tol {
                if step == 13 {
                    let ring: [i32; 12] = ids[..12].try_into().expect("twelve places");
                    if !(self.found)(&ring) {
                        self.stopped = true;
                    }
                } else {
                    ids.push(ids[0]);
                    pos.push(q);
                    self.walk(ids, pos, missing, size);
                    ids.pop();
                    pos.pop();
                }
            }
            return;
        }

        let candidates: Vec<u32> =
            self.around([x, y], tol, size, ids[step - 1], ids[step - 2]).into_iter().filter(|&i| !ids.contains(&(i as i32))).collect();
        // Early on the prediction is loose, so try every blob in range; later only the closest.
        let tries = if step <= 4 { 3 } else { 1 };
        for &i in candidates.iter().take(tries) {
            ids.push(i as i32);
            pos.push(self.nb.pts[i as usize]);
            self.walk(ids, pos, missing, size);
            ids.pop();
            pos.pop();
        }
        // The fourth star must be real, and no two may be missing in a row.
        if candidates.is_empty() && step > 3 && ids[step - 1] >= 0 && missing < MAX_MISSING {
            ids.push(-1);
            pos.push([x, y]);
            self.walk(ids, pos, missing + 1, size);
            ids.pop();
            pos.pop();
        }
    }
}

/// Find closed rings of twelve evenly spaced, similar blobs, and hand each to
/// `found`. `found` returns false to stop the search.
///
/// Blobs with more than `max_crowd` similar neighbours are skipped: in a dense
/// regular texture (screen pixel grids, halftone, fabric) every blob has many,
/// and they would support endless false rings. With `max_crowd = usize::MAX`
/// nothing is skipped, which is what a code with several rings needs; that
/// search starts from the middle of the crowd, where ring 0 is, because
/// everything further out only offers look-alike loops.
pub(crate) fn find_rings(nb: &Neighbourhood, max_crowd: usize, mut found: impl FnMut(Ring) -> bool) {
    let n = nb.len();
    if n < 12 - MAX_MISSING {
        return;
    }
    let crowded = |i: u32| nb.crowd[i as usize] > max_crowd;

    // Largest blobs first, so that real stars are tried before fine texture.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| nb.size[b as usize].total_cmp(&nb.size[a as usize]));
    let dense: Vec<Point> = (0..n).filter(|&i| nb.crowd[i] > CROWD).map(|i| nb.pts[i]).collect();
    if max_crowd == usize::MAX && dense.len() >= 12 {
        let median = |axis: usize| crate::math::median(&dense.iter().map(|p| p[axis]).collect::<Vec<f64>>());
        let middle = [median(0), median(1)];
        let d2 = |i: u32| (nb.pts[i as usize][0] - middle[0]).powi(2) + (nb.pts[i as usize][1] - middle[1]).powi(2);
        order.sort_by(|&a, &b| d2(a).total_cmp(&d2(b)));
    }

    // Rings reported so far (as sorted blob lists), and which of them each blob is in.
    let mut seen: Vec<Vec<u32>> = Vec::new();
    let mut ring_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut budget = WALK_BUDGET;
    let mut stopped = false;

    for &b in &order {
        if nb.near[b as usize].len() < 2 || crowded(b) || budget < 0 || stopped {
            continue;
        }
        let pb = nb.pts[b as usize];
        for &a in &nb.near[b as usize] {
            if crowded(a) {
                continue;
            }
            let pa = nb.pts[a as usize];
            let l1 = (pb[0] - pa[0]).hypot(pb[1] - pa[1]);
            for &c in &nb.near[b as usize] {
                if c == a || crowded(c) || stopped {
                    continue;
                }
                // Three blobs already together in a reported ring lead to the same ring.
                let shared = ring_of[a as usize].iter().any(|r| ring_of[b as usize].contains(r) && ring_of[c as usize].contains(r));
                if shared {
                    continue;
                }
                let pc = nb.pts[c as usize];
                let l2 = (pc[0] - pb[0]).hypot(pc[1] - pb[1]);
                if !(0.55 < l2 / l1 && l2 / l1 < 1.8) {
                    continue;
                }
                // Clockwise in image coordinates only, turning between 5 and 85 degrees.
                let cross = (pb[0] - pa[0]) * (pc[1] - pb[1]) - (pb[1] - pa[1]) * (pc[0] - pb[0]);
                let dot = (pb[0] - pa[0]) * (pc[0] - pb[0]) + (pb[1] - pa[1]) * (pc[1] - pb[1]);
                if cross <= 0.0 || dot <= 0.0 || cross < 0.087 * dot {
                    continue;
                }
                let size = (nb.size[a as usize] + nb.size[b as usize] + nb.size[c as usize]) / 3.0;
                let mut walker = Walker {
                    nb,
                    budget,
                    stopped: false,
                    found: |ids: &[i32; 12]| {
                        let slots: Vec<u8> = (0..12u8).filter(|&k| ids[k as usize] >= 0).collect();
                        let blobs: Vec<u32> = slots.iter().map(|&k| ids[k as usize] as u32).collect();
                        let mut key = blobs.clone();
                        key.sort_unstable();
                        // The same ring again, or part of one already reported.
                        if seen.iter().any(|ring| key.iter().all(|i| ring.binary_search(i).is_ok())) {
                            return true;
                        }
                        seen.push(key);
                        for &i in &blobs {
                            ring_of[i as usize].push(seen.len() as u32);
                        }
                        found(Ring { blobs, slots })
                    },
                };
                walker.walk(&mut vec![a as i32, b as i32, c as i32], &mut vec![pa, pb, pc], 0, size);
                budget = walker.budget;
                stopped = walker.stopped;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_blobs(skip: &[usize]) -> Vec<Blob> {
        (0..12)
            .filter(|i| !skip.contains(i))
            .map(|i| (30.0 * i as f64).to_radians())
            // An ellipse, as a tilted ring appears.
            .map(|a| Blob { x: 300.0 + 120.0 * a.cos() + 20.0 * a.sin(), y: 200.0 + 70.0 * a.sin(), area: 400.0 })
            .collect()
    }

    /// Neighbour lists the plain way: every blob against every other.
    fn every_pair(blobs: &[Blob]) -> Neighbourhood {
        let n = blobs.len();
        let pts: Vec<Point> = blobs.iter().map(|b| [b.x, b.y]).collect();
        let size: Vec<f64> = blobs.iter().map(|b| b.area.sqrt()).collect();
        let mut near_d: Vec<Vec<(f64, u32)>> = vec![Vec::new(); n];
        let mut far_d: Vec<Vec<(f64, u32)>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in i + 1..n {
                let (big, small) = (size[i].max(size[j]), size[i].min(size[j]));
                // Similar in size, not overlapping, and within a few sizes of each other.
                if big >= 2.3 * small {
                    continue;
                }
                let (dx, dy) = (pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
                // Distances as their squares, as in `Neighbourhood::new`.
                let (d, big2) = (dx * dx + dy * dy, big * big);
                if d <= 0.8 * 0.8 * big2 || d >= 13.0 * 13.0 * big2 {
                    continue;
                }
                far_d[i].push((d, j as u32));
                far_d[j].push((d, i as u32));
                if d < 6.5 * 6.5 * big2 {
                    near_d[i].push((d, j as u32));
                    near_d[j].push((d, i as u32));
                }
            }
        }
        let crowd = near_d.iter().map(Vec::len).collect();
        let closest = |mut list: Vec<(f64, u32)>, keep: usize| -> Vec<u32> {
            list.sort_by(|a, b| a.0.total_cmp(&b.0));
            list.truncate(keep);
            list.into_iter().map(|(_, j)| j).collect()
        };
        Neighbourhood {
            near: near_d.into_iter().map(|l| closest(l, NEAREST)).collect(),
            far: far_d.into_iter().map(|l| closest(l, 3 * NEAREST)).collect(),
            pts,
            size,
            crowd,
        }
    }

    #[test]
    fn neighbours_found_through_cells_are_those_found_pair_by_pair() {
        let mut state = 7u64;
        let mut unit = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 40) as f64 / (1u64 << 24) as f64
        };
        // A dense field of like blobs on a grid (many equal distances), a sparse scatter
        // of every size, a single blob, and none.
        let mut sets: Vec<Vec<Blob>> = Vec::new();
        sets.push((0..900).map(|i| Blob { x: 20.0 * (i % 30) as f64, y: 20.0 * (i / 30) as f64, area: 90.0 + (i % 7) as f64 }).collect());
        sets.push((0..700).map(|_| Blob { x: 1900.0 * unit(), y: 1400.0 * unit(), area: 9.0 + 4000.0 * unit().powi(4) }).collect());
        sets.push((0..400).map(|_| Blob { x: 300.0 * unit(), y: 200.0 * unit(), area: 20.0 + 30.0 * unit() }).collect());
        sets.push(vec![Blob { x: 5.0, y: 5.0, area: 30.0 }]);
        sets.push(Vec::new());
        for blobs in &sets {
            let (quick, plain) = (Neighbourhood::new(blobs), every_pair(blobs));
            assert_eq!(quick.crowd, plain.crowd);
            assert_eq!(quick.near, plain.near);
            assert_eq!(quick.far, plain.far);
        }
    }

    #[test]
    fn prediction_weights_match_the_reference() {
        let w = predict_weights(3);
        assert!((w[0] - 1.0).abs() < 1e-6 && (w[1] + 2.732_051).abs() < 1e-6 && (w[2] - 2.732_051).abs() < 1e-6);
        let w = predict_weights(5);
        assert!((w[0] - 0.625).abs() < 1e-6 && (w[4] - 1.491_025).abs() < 1e-6);
    }

    #[test]
    fn a_ring_of_twelve_is_found_once() {
        let mut blobs = ring_blobs(&[]);
        blobs.push(Blob { x: 50.0, y: 50.0, area: 380.0 }); // clutter
        let nb = Neighbourhood::new(&blobs);
        let mut rings = Vec::new();
        find_rings(&nb, CROWD, |ring| {
            rings.push(ring);
            true
        });
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].blobs.len(), 12);
        assert_eq!(rings[0].slots, (0..12).collect::<Vec<u8>>());
    }

    #[test]
    fn a_ring_with_stars_missing_is_still_found() {
        let nb = Neighbourhood::new(&ring_blobs(&[4, 9]));
        let mut rings = Vec::new();
        find_rings(&nb, CROWD, |ring| {
            rings.push(ring);
            true
        });
        assert!(!rings.is_empty());
        assert_eq!(rings[0].blobs.len(), 10);
        assert_eq!(rings[0].slots.len(), 10);
    }
}
