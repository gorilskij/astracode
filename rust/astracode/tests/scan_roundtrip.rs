//! End to end: draw a code with the generator, turn the drawing into pixels,
//! distort it the way a camera would, and scan it back.

use astracode::format::svg::{svg, Centre, Style};
use astracode::format::{message, ring0};
use astracode::image::Gray;
use astracode::math::Homography;
use astracode::scan::{scan_rgb8, Code, Options, Payload};

/// An RGB image, `r, g, b` per pixel.
struct Rgb {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

/// Render a code to pixels.
fn render(rings: &[Vec<u8>], style: &Style) -> Rgb {
    let tree = resvg::usvg::Tree::from_str(&svg(rings, style), &resvg::usvg::Options::default()).expect("the generator writes valid SVG");
    let mut pixmap = resvg::tiny_skia::Pixmap::new(style.size, style.size).expect("a non-empty image");
    resvg::render(&tree, resvg::tiny_skia::Transform::identity(), &mut pixmap.as_mut());
    let pixels = pixmap.data().chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    Rgb { width: style.size as usize, height: style.size as usize, pixels }
}

/// Place `code` into a `width` by `height` scene of colour `fill`, as seen
/// through `to_code`, which takes scene pixels to code pixels.
fn photograph(code: &Rgb, to_code: &Homography, width: usize, height: usize, fill: [u8; 3]) -> Rgb {
    let mut pixels = vec![0u8; 3 * width * height];
    let inside = Gray::from_luma8(code.width, code.height, &vec![255u8; code.width * code.height]).warp(to_code, width, height);
    for channel in 0..3 {
        let plane = Gray::from_rgb8_channel(code.width, code.height, &code.pixels, channel).warp(to_code, width, height);
        for i in 0..width * height {
            let cover = inside.data[i] / 255.0;
            pixels[3 * i + channel] = (plane.data[i] + (1.0 - cover) * fill[channel] as f32).round().clamp(0.0, 255.0) as u8;
        }
    }
    Rgb { width, height, pixels }
}

/// The code turned by `degrees`, tilted, shrunk by `scale` and moved, in a larger grey scene.
fn tilted(code: &Rgb, degrees: f64, scale: f64) -> Rgb {
    let (s, c) = degrees.to_radians().sin_cos();
    let (cx, cy) = (code.width as f64 / 2.0, code.height as f64 / 2.0);
    // Code pixels -> scene pixels: about the code's centre, rotate, scale, add a little perspective.
    let matrix = projective([[scale * c, -scale * s, 420.0], [scale * s, scale * c, 330.0], [0.000_25, 0.000_15, 1.0]]);
    let to_scene = matrix.after(&Homography::scale_then_shift(1.0, 1.0, -cx, -cy));
    photograph(code, &to_scene.inverse().expect("invertible"), 860, 680, [120, 125, 118])
}

fn projective(m: [[f64; 3]; 3]) -> Homography {
    // Built by fitting to a few points, so that the test needs no matrix library of its own.
    let apply = |p: [f64; 2]| {
        let w = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2];
        [(m[0][0] * p[0] + m[0][1] * p[1] + m[0][2]) / w, (m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]) / w]
    };
    let src = [[-300.0, -300.0], [300.0, -300.0], [300.0, 300.0], [-300.0, 300.0], [0.0, 0.0], [150.0, -70.0]];
    Homography::fit(&src, &src.map(apply))
}

fn scan(image: &Rgb) -> Vec<Code> {
    scan_rgb8(image.width, image.height, &image.pixels, &Options::default())
}

fn one_ring(metadata: u64, payload: u64) -> Vec<Vec<u8>> {
    vec![ring0::encode(ring0::pack(metadata, payload).unwrap()).unwrap().to_vec()]
}

#[test]
fn a_one_ring_code_scans_straight_and_tilted() {
    let style = Style { size: 600, ..Style::default() };
    let code = render(&one_ring(5, 4_320_880_580), &style);
    for image in [scan(&code), scan(&tilted(&code, 37.0, 0.75)), scan(&tilted(&code, 200.0, 0.5))] {
        assert_eq!(image.len(), 1);
        assert_eq!(image[0].number(), Some((5, 4_320_880_580)));
        assert_eq!(image[0].rings, 1);
        assert!(!image[0].mirrored);
    }
}

#[test]
fn colours_and_the_centre_picture_do_not_matter() {
    for (ink, background, centre) in [("#ffffff", "#20104a", Centre::Logo), ("#0a5d2a", "#e8e0d0", Centre::Empty)] {
        let style = Style { ink: ink.into(), background: background.into(), centre, size: 500, ..Style::default() };
        let found = scan(&tilted(&render(&one_ring(42, 2_718_281_828), &style), -80.0, 0.8));
        assert_eq!(found.len(), 1, "{ink} on {background}");
        assert_eq!(found[0].number(), Some((42, 2_718_281_828)));
    }
}

#[test]
fn a_mirror_image_is_read_and_reported_as_mirrored() {
    let code = render(&one_ring(1, 5_550_001_112), &Style { size: 500, ..Style::default() });
    let mut flipped = Rgb { width: code.width, height: code.height, pixels: Vec::with_capacity(code.pixels.len()) };
    for row in code.pixels.chunks_exact(3 * code.width) {
        flipped.pixels.extend(row.chunks_exact(3).rev().flatten());
    }
    let found = scan(&tilted(&flipped, 15.0, 0.9));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].number(), Some((1, 5_550_001_112)));
    assert!(found[0].mirrored);
}

#[test]
fn a_message_code_scans_at_several_sizes() {
    for (rings, level, text) in
        [(2, 3, "hi"), (4, 1, "https://astra-europa.example/"), (9, 2, "nine rings of stars around a small picture")]
    {
        let symbols = message::encode(text.as_bytes(), rings, level).unwrap();
        let code = render(&symbols, &Style { size: 300 + 110 * rings as u32, ..Style::default() });
        for image in [scan(&code), scan(&tilted(&code, 123.0, 0.55))] {
            assert_eq!(image.len(), 1, "{rings} rings");
            assert_eq!(image[0].payload, Payload::Message { level, bytes: text.as_bytes().to_vec() });
            assert_eq!(image[0].rings, rings);
        }
    }
}

#[test]
fn ring_0_of_a_bigger_code_is_not_a_code_by_itself() {
    // Paint over everything outside ring 0 of a three-ring code.
    let symbols = message::encode(b"cropped", 3, 1).unwrap();
    let style = Style { size: 900, ..Style::default() };
    let mut code = render(&symbols, &style);
    let centre = 450.0;
    let keep = 900.0 * 122.0 / (2.0 * astracode::format::half_side(3)); // just outside ring 0's stars
    let background = [code.pixels[0], code.pixels[1], code.pixels[2]];
    for y in 0..code.height {
        for x in 0..code.width {
            if (x as f64 - centre).hypot(y as f64 - centre) > keep {
                code.pixels[3 * (y * code.width + x)..][..3].copy_from_slice(&background);
            }
        }
    }
    assert!(scan(&code).is_empty());
}

#[test]
fn an_image_without_a_code_gives_nothing() {
    let mut pixels = vec![0u8; 3 * 400 * 300];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 2_654_435_761) >> 13) as u8; // noise
    }
    assert!(scan(&Rgb { width: 400, height: 300, pixels }).is_empty());
    assert!(scan(&Rgb { width: 400, height: 300, pixels: vec![200u8; 3 * 400 * 300] }).is_empty());
}
