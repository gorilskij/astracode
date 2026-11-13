//! A grey image and the few operations the scanner needs on it.
//!
//! Pixels are `f32` levels on the usual 0 to 255 scale. Pixel `(x, y)` has its
//! centre at the coordinates `(x, y)`, so sampling at integer coordinates
//! returns the pixel itself.

use crate::math::Homography;

/// A single-channel image with `f32` pixels, stored row by row.
#[derive(Clone)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// Call `f(index, band)` for each band of `size` values of `data`, on all
/// cores when the `parallel` feature is on.
fn for_each_band(data: &mut [f32], size: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(size).enumerate().for_each(|(i, band)| f(i, band));
    }
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(size).enumerate().for_each(|(i, band)| f(i, band));
}

/// Call `f(y, row)` for each row of an image `width` wide.
fn for_each_row(data: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    for_each_band(data, width, f);
}

/// Index `i` reflected into `0..n` without repeating the edge pixel
/// (`... 2 1 | 0 1 2 ... n-1 | n-2 n-3 ...`), the border rule OpenCV uses by default.
#[inline]
pub(crate) fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = i.rem_euclid(period);
    (if m < n { m } else { period - m }) as usize
}

impl Gray {
    /// A black image.
    pub fn new(width: usize, height: usize) -> Gray {
        Gray { width, height, data: vec![0.0; width * height] }
    }

    /// From 8-bit grey pixels, row by row.
    pub fn from_luma8(width: usize, height: usize, pixels: &[u8]) -> Gray {
        assert_eq!(pixels.len(), width * height, "pixel buffer does not match the image size");
        Gray { width, height, data: pixels.iter().map(|&p| p as f32).collect() }
    }

    /// From 8-bit grey pixels stored as rows of `stride` bytes of which the first
    /// `width` are pixels, as camera frames are. The last row may be cut short
    /// after its pixels.
    pub fn from_luma8_rows(width: usize, height: usize, stride: usize, pixels: &[u8]) -> Gray {
        assert!(stride >= width && pixels.len() >= stride * (height - 1) + width, "pixel buffer does not match the image size");
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            data.extend(pixels[y * stride..y * stride + width].iter().map(|&p| p as f32));
        }
        Gray { width, height, data }
    }

    /// Brightness of 8-bit RGB pixels (`r, g, b, r, g, b, ...`), rounded to
    /// whole levels with the same weights and rounding as OpenCV.
    pub fn from_rgb8(width: usize, height: usize, pixels: &[u8]) -> Gray {
        assert_eq!(pixels.len(), 3 * width * height, "pixel buffer does not match the image size");
        let luma = |p: &[u8]| (p[0] as u32 * 4899 + p[1] as u32 * 9617 + p[2] as u32 * 1868 + 8192) >> 14;
        Gray { width, height, data: pixels.chunks_exact(3).map(|p| luma(p) as f32).collect() }
    }

    /// One channel (0 = red, 1 = green, 2 = blue) of 8-bit RGB pixels.
    pub fn from_rgb8_channel(width: usize, height: usize, pixels: &[u8], channel: usize) -> Gray {
        assert_eq!(pixels.len(), 3 * width * height, "pixel buffer does not match the image size");
        Gray { width, height, data: pixels.chunks_exact(3).map(|p| p[channel] as f32).collect() }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// The level at a fractional position, by bilinear interpolation.
    /// Positions outside the image take the nearest edge value.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> f32 {
        let x = x.clamp(0.0, self.width as f64 - 1.001);
        let y = y.clamp(0.0, self.height as f64 - 1.001);
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let i = y0 * self.width + x0;
        // After the clamp x0 <= width - 2 and y0 <= height - 2, so the four pixels read
        // below exist. (The scanner rejects images too small for that to hold.)
        let top = self.data[i] * (1.0 - fx) + self.data[i + 1] * fx;
        let bottom = self.data[i + self.width] * (1.0 - fx) + self.data[i + self.width + 1] * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// As [`Gray::sample`], for a position given in single precision: good to a
    /// thousandth of a pixel in any image a camera makes, and quicker.
    #[inline(always)]
    pub fn sample_near(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, self.width as f32 - 1.001);
        let y = y.clamp(0.0, self.height as f32 - 1.001);
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let (top, below) = (&self.data[y0 * self.width + x0..][..2], &self.data[(y0 + 1) * self.width + x0..][..2]);
        let (top, below) = (top[0] + (top[1] - top[0]) * fx, below[0] + (below[1] - below[0]) * fx);
        top + (below - top) * fy
    }

    /// Convolve rows and then columns with a symmetric kernel of odd length,
    /// reflecting at the borders.
    fn separable(&self, kernel: &[f32]) -> Gray {
        let r = kernel.len() / 2;
        let (w, h) = (self.width, self.height);
        // Each pass adds one shifted copy of its input per kernel tap, which
        // the compiler turns into wide vector operations.
        let mut rows = Gray::new(w, h);
        for_each_row(&mut rows.data, w, |y, out| {
            let line = &self.data[y * w..(y + 1) * w];
            if w > 2 * r {
                for (i, &k) in kernel.iter().enumerate() {
                    for (o, v) in out[r..w - r].iter_mut().zip(&line[i..]) {
                        *o += k * v;
                    }
                }
            }
            // The ends, where the kernel hangs over the edge.
            for x in (0..r.min(w)).chain(w.saturating_sub(r).max(r.min(w))..w) {
                out[x] = kernel.iter().enumerate().map(|(i, k)| k * line[reflect(x as isize + i as isize - r as isize, w)]).sum();
            }
        });
        let mut out = Gray::new(w, h);
        for_each_row(&mut out.data, w, |y, dst| {
            for (i, &k) in kernel.iter().enumerate() {
                let src = reflect(y as isize + i as isize - r as isize, h);
                for (d, s) in dst.iter_mut().zip(&rows.data[src * w..(src + 1) * w]) {
                    *d += k * s;
                }
            }
        });
        out
    }

    /// Gaussian blur. The kernel reaches four standard deviations each way.
    pub fn gaussian_blur(&self, sigma: f64) -> Gray {
        let r = (4.0 * sigma).round().max(1.0) as isize;
        let mut kernel: Vec<f32> = (-r..=r).map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp() as f32).collect();
        let sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= sum);
        self.separable(&kernel)
    }

    /// The mean over a `window` by `window` square around each pixel (`window`
    /// odd), reflecting at the borders. Cost does not depend on the window size.
    pub fn box_blur(&self, window: usize) -> Gray {
        let r = (window / 2) as isize;
        let (w, h) = (self.width, self.height);
        let scale = 1.0 / window as f32;

        // Rows: a running sum over the window as it slides along each line. Only at
        // the two ends does the window hang over the edge and need reflecting.
        let mut rows = Gray::new(w, h);
        for_each_row(&mut rows.data, w, |y, out| {
            let line = &self.data[y * w..(y + 1) * w];
            let at = |x: isize| line[reflect(x, w)];
            let mut sum: f32 = (-r..=r).map(at).sum();
            let (first, last) = ((r as usize).min(w), w.saturating_sub(r as usize + 1));
            for x in 0..w {
                out[x] = sum * scale;
                sum += if x >= first && x < last {
                    line[x + r as usize + 1] - line[x - r as usize]
                } else {
                    at(x as isize + r + 1) - at(x as isize - r)
                };
            }
        });

        // Columns: a running sum of whole rows, restarted for each band of rows so that bands can run side by side.
        const BAND: usize = 64;
        let mut out = Gray::new(w, h);
        let row = |y: isize| {
            let y = reflect(y, h);
            &rows.data[y * w..(y + 1) * w]
        };
        for_each_band(&mut out.data, w * BAND, |band, lines| {
            let top = (band * BAND) as isize;
            let mut sum = vec![0f32; w];
            for y in top - r..=top + r {
                sum.iter_mut().zip(row(y)).for_each(|(s, v)| *s += v);
            }
            for (i, line) in lines.chunks_mut(w).enumerate() {
                let y = top + i as isize;
                line.iter_mut().zip(&sum).for_each(|(o, s)| *o = s * scale);
                let (add, drop) = (row(y + r + 1), row(y - r));
                for ((s, a), d) in sum.iter_mut().zip(add).zip(drop) {
                    *s += a - d;
                }
            }
        });
        out
    }

    /// The mean of each `cell` by `cell` square of pixels, as an image that many
    /// times smaller (rounded up: cells at the right and bottom edges may be cut short).
    pub fn cell_means(&self, cell: usize) -> Gray {
        let (w, h) = (self.width, self.height);
        let mut out = Gray::new(w.div_ceil(cell), h.div_ceil(cell));
        let cells_across = out.width;
        for_each_row(&mut out.data, cells_across, |cy, sums| {
            let rows = cy * cell..((cy + 1) * cell).min(h);
            for y in rows.clone() {
                for (sum, pixels) in sums.iter_mut().zip(self.data[y * w..(y + 1) * w].chunks(cell)) {
                    *sum += pixels.iter().sum::<f32>();
                }
            }
            for (cx, sum) in sums.iter_mut().enumerate() {
                *sum /= (rows.len() * (w - cx * cell).min(cell)) as f32;
            }
        });
        out
    }

    /// Shrink by `scale` (below 1), averaging the source pixels each new pixel covers.
    pub fn shrink(&self, scale: f64) -> Gray {
        let w = ((self.width as f64 * scale).round() as usize).max(1);
        let h = ((self.height as f64 * scale).round() as usize).max(1);
        let (sx, sy) = (self.width as f64 / w as f64, self.height as f64 / h as f64);
        // Weights of the source columns under each destination column, and likewise for rows.
        let spans = |n: usize, step: f64, limit: usize| -> Vec<Vec<(usize, f32)>> {
            (0..n)
                .map(|i| {
                    let (lo, hi) = (i as f64 * step, (i as f64 + 1.0) * step);
                    (lo.floor() as usize..(hi.ceil() as usize).min(limit))
                        .map(|j| (j, ((hi.min(j as f64 + 1.0) - lo.max(j as f64)) / step) as f32))
                        .collect()
                })
                .collect()
        };
        let (cols, rows) = (spans(w, sx, self.width), spans(h, sy, self.height));
        let mut out = Gray::new(w, h);
        for (y, row) in rows.iter().enumerate() {
            for (x, col) in cols.iter().enumerate() {
                let mut v = 0.0;
                for &(j, wy) in row {
                    let line = &self.data[j * self.width..];
                    v += wy * col.iter().map(|&(i, wx)| wx * line[i]).sum::<f32>();
                }
                out.data[y * w + x] = v;
            }
        }
        out
    }

    /// A new `width` by `height` image whose pixel `(u, v)` is this image sampled
    /// at `to_source(u, v)`. Positions outside this image are black.
    pub fn warp(&self, to_source: &Homography, width: usize, height: usize) -> Gray {
        self.warp_with(to_source, width, height, [0.0, 0.0], 0.0, f64::INFINITY, |x, y| self.sample(x, y))
    }

    /// As [`Gray::warp`], but only the pixels whose distance from `centre` is
    /// between `inner` and `outer` are worked out; the rest are left black. For
    /// a caller that is going to look at one ring of the result.
    pub fn warp_between(&self, to_source: &Homography, width: usize, height: usize, centre: [f64; 2], inner: f64, outer: f64) -> Gray {
        self.warp_with(to_source, width, height, centre, inner, outer, |x, y| self.sample_near(x as f32, y as f32))
    }

    /// The pixels of a warp between `inner` and `outer` of `centre`, each `pick`ed
    /// from this image at the position it maps to (which is inside the image).
    #[allow(clippy::too_many_arguments)]
    fn warp_with(
        &self,
        to_source: &Homography,
        width: usize,
        height: usize,
        centre: [f64; 2],
        inner: f64,
        outer: f64,
        pick: impl Fn(f64, f64) -> f32 + Sync,
    ) -> Gray {
        /// The exact position is worked out every this many pixels along a row and
        /// interpolated between. Over so short a stretch a projective map is as good
        /// as straight (hundredths of a pixel off under the strongest perspective a
        /// code is read at), and the division it needs is the costly part.
        const STRETCH: usize = 8;
        let mut out = Gray::new(width, height);
        let (w, h) = (self.width as f64, self.height as f64);
        let m = to_source.matrix();
        for_each_row(&mut out.data, width, |v, line| {
            let v = v as f64;
            let at = |u: f64| {
                let pw = m[(2, 0)] * u + m[(2, 1)] * v + m[(2, 2)];
                ((m[(0, 0)] * u + m[(0, 1)] * v + m[(0, 2)]) / pw, (m[(1, 0)] * u + m[(1, 1)] * v + m[(1, 2)]) / pw)
            };
            // The row crosses the ring in one stretch, or in two with the hole between them.
            let dy = (v - centre[1]).abs();
            if dy > outer {
                return;
            }
            let column = |x: f64| x.clamp(0.0, width as f64) as usize;
            let reach = if outer.is_finite() { (outer * outer - dy * dy).sqrt() } else { f64::INFINITY };
            let (first, last) = (column((centre[0] - reach).ceil()), column((centre[0] + reach).floor() + 1.0));
            let spans = if dy < inner {
                let hole = (inner * inner - dy * dy).sqrt();
                [(first, column((centre[0] - hole).floor() + 1.0)), (column((centre[0] + hole).ceil()), last)]
            } else {
                [(first, last), (0, 0)]
            };
            for (from, to) in spans {
                let mut u = from;
                while u < to {
                    let n = STRETCH.min(to - u);
                    let ((x0, y0), (x1, y1)) = (at(u as f64), at((u + n) as f64));
                    let (dx, dy) = ((x1 - x0) / n as f64, (y1 - y0) / n as f64);
                    for (i, o) in line[u..u + n].iter_mut().enumerate() {
                        let (x, y) = (x0 + dx * i as f64, y0 + dy * i as f64);
                        if x >= 0.0 && y >= 0.0 && x <= w - 1.0 && y <= h - 1.0 {
                            *o = pick(x, y);
                        }
                    }
                    u += n;
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: usize, height: usize) -> Gray {
        let mut g = Gray::new(width, height);
        for y in 0..height {
            for x in 0..width {
                g.data[y * width + x] = (3 * x + 7 * y) as f32;
            }
        }
        g
    }

    #[test]
    fn sampling_interpolates_and_clamps() {
        let g = ramp(8, 6);
        assert_eq!(g.sample(2.0, 3.0), g.get(2, 3));
        assert!((g.sample(2.5, 3.25) - (3.0 * 2.5 + 7.0 * 3.25)).abs() < 1e-4);
        assert!((g.sample(-5.0, -5.0) - g.get(0, 0)).abs() < 1e-4);
        assert!((g.sample(99.0, 99.0) - g.get(7, 5)).abs() < 0.05);
    }

    #[test]
    fn blurs_keep_a_linear_ramp_away_from_the_borders() {
        let g = ramp(40, 30);
        for blurred in [g.box_blur(7), g.gaussian_blur(1.0)] {
            for (x, y) in [(10, 10), (20, 15), (30, 20)] {
                assert!((blurred.get(x, y) - g.get(x, y)).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn box_blur_matches_a_direct_mean() {
        let mut g = Gray::new(9, 7);
        g.data.iter_mut().enumerate().for_each(|(i, v)| *v = ((i * 37) % 11) as f32);
        let b = g.box_blur(3);
        for (x, y) in [(0usize, 0usize), (4, 3), (8, 6), (0, 6)] {
            let mut sum = 0.0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    sum += g.get(reflect(x as isize + dx, 9), reflect(y as isize + dy, 7));
                }
            }
            assert!((b.get(x, y) - sum / 9.0).abs() < 1e-4);
        }
    }

    #[test]
    fn warping_a_ring_matches_warping_everything_inside_the_ring() {
        // A projective map with some perspective in it.
        let g = ramp(120, 100);
        let source = [[10.0, 8.0], [105.0, 14.0], [98.0, 90.0], [6.0, 80.0]];
        let h = Homography::fit(&[[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]], &source);
        let (all, ring) = (g.warp(&h, 64, 64), g.warp_between(&h, 64, 64, [32.0, 30.0], 12.0, 25.0));
        for y in 0..64 {
            for x in 0..64 {
                let d = ((x as f64 - 32.0).powi(2) + (y as f64 - 30.0).powi(2)).sqrt();
                if (12.0..=25.0).contains(&d) {
                    // The same up to where the stretches between exact positions happen to start,
                    // and those hardly show: the ramp comes out as the map says, to a twentieth of a
                    // pixel on a map with far more perspective per pixel than a photo has.
                    assert!((ring.get(x, y) - all.get(x, y)).abs() < 0.3);
                    let at = h.apply([x as f64, y as f64]);
                    assert!((all.get(x, y) as f64 - (3.0 * at[0] + 7.0 * at[1])).abs() < 0.3, "{x} {y}");
                } else if d < 11.0 || d > 26.0 {
                    assert_eq!(ring.get(x, y), 0.0);
                }
            }
        }
    }

    #[test]
    fn cells_hold_the_mean_of_their_pixels() {
        let g = ramp(10, 7);
        let c = g.cell_means(4);
        assert_eq!((c.width, c.height), (3, 2));
        let mean = |xs: std::ops::Range<usize>, ys: std::ops::Range<usize>| {
            let n = (xs.len() * ys.len()) as f32;
            ys.flat_map(|y| xs.clone().map(move |x| (x, y))).map(|(x, y)| g.get(x, y)).sum::<f32>() / n
        };
        assert!((c.get(0, 0) - mean(0..4, 0..4)).abs() < 1e-4);
        assert!((c.get(2, 0) - mean(8..10, 0..4)).abs() < 1e-4);
        assert!((c.get(1, 1) - mean(4..8, 4..7)).abs() < 1e-4);
        assert!((c.get(2, 1) - mean(8..10, 4..7)).abs() < 1e-4);
    }

    #[test]
    fn shrinking_keeps_the_mean() {
        let g = ramp(64, 48);
        let s = g.shrink(0.5);
        assert_eq!((s.width, s.height), (32, 24));
        let mean = |g: &Gray| g.data.iter().sum::<f32>() / g.data.len() as f32;
        assert!((mean(&g) - mean(&s)).abs() < 1e-2);
    }
}
