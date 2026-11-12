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
fn reflect(i: isize, n: usize) -> usize {
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
        let mut out = Gray::new(width, height);
        let (w, h) = (self.width as f64, self.height as f64);
        let m = to_source.matrix();
        for_each_row(&mut out.data, width, |v, line| {
            // Along a row the three homogeneous coordinates advance by a constant step.
            let v = v as f64;
            let (mut px, mut py, mut pw) = (m[(0, 1)] * v + m[(0, 2)], m[(1, 1)] * v + m[(1, 2)], m[(2, 1)] * v + m[(2, 2)]);
            for o in line.iter_mut() {
                let (x, y) = (px / pw, py / pw);
                if x >= 0.0 && y >= 0.0 && x <= w - 1.0 && y <= h - 1.0 {
                    *o = self.sample(x, y);
                }
                px += m[(0, 0)];
                py += m[(1, 0)];
                pw += m[(2, 0)];
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
    fn shrinking_keeps_the_mean() {
        let g = ramp(64, 48);
        let s = g.shrink(0.5);
        assert_eq!((s.width, s.height), (32, 24));
        let mean = |g: &Gray| g.data.iter().sum::<f32>() / g.data.len() as f32;
        assert!((mean(&g) - mean(&s)).abs() < 1e-2);
    }
}
