//! End to end: draw a code with the generator, turn the drawing into pixels,
//! distort it the way a camera would, and scan it back.

use astracode::astra::Number;
use astracode::format::code;
use astracode::format::svg::{svg, Centre, Style};
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

/// The one-ring code of an Astra number.
fn one_ring(metadata: u8, payload: u64) -> Vec<Vec<u8>> {
    let rings = code::encode(&Number::new(metadata, payload).unwrap().to_payload(), 1).unwrap();
    assert_eq!(rings.len(), 1);
    rings
}

/// What a scanned code holds, read as an Astra number.
fn number(code: &Code) -> Option<(u8, u64)> {
    Number::from_payload(&code.payload).map(|n| (n.metadata, n.payload))
}

#[test]
fn a_one_ring_code_scans_straight_and_tilted() {
    let style = Style { size: 600, ..Style::default() };
    let code = render(&one_ring(5, 4_320_880_580), &style);
    for image in [scan(&code), scan(&tilted(&code, 37.0, 0.75)), scan(&tilted(&code, 200.0, 0.5))] {
        assert_eq!(image.len(), 1);
        assert_eq!(number(&image[0]), Some((5, 4_320_880_580)));
        assert_eq!(image[0].rings, 1);
        assert!(!image[0].mirrored);
    }
}

#[test]
fn colours_and_the_centre_picture_do_not_matter() {
    for (ink, background, centre) in [("#ffffff", "#20104a", Centre::Logo), ("#0a5d2a", "#e8e0d0", Centre::Empty)] {
        let style = Style { ink: ink.into(), background: background.into(), centre, size: 500, ..Style::default() };
        let found = scan(&tilted(&render(&one_ring(17, 2_718_281_828), &style), -80.0, 0.8));
        assert_eq!(found.len(), 1, "{ink} on {background}");
        assert_eq!(number(&found[0]), Some((17, 2_718_281_828)));
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
    assert_eq!(number(&found[0]), Some((1, 5_550_001_112)));
    assert!(found[0].mirrored);
}

#[test]
fn a_code_sizes_itself_and_scans_at_several_sizes() {
    // Text and level, and the number of rings that should come out.
    for (text, level, rings) in [
        ("hi", 3, 1),
        ("hello", 3, 2),
        ("https://astra-europa.example/", 1, 4),
        ("nine rings of stars around a small picture, with a little more text to fill them all the way out to the edge", 2, 9),
    ] {
        let symbols = code::encode(&Payload::from_bytes(text), level).unwrap();
        assert_eq!(symbols.len(), rings, "{text:?}");
        let code = render(&symbols, &Style { size: 300 + 110 * rings as u32, ..Style::default() });
        for image in [scan(&code), scan(&tilted(&code, 123.0, 0.55))] {
            assert_eq!(image.len(), 1, "{rings} rings");
            assert_eq!(image[0].payload.as_text(), Some(text));
            assert_eq!((image[0].rings, image[0].level), (rings, if rings == 1 { 0 } else { level }));
            assert_eq!(number(&image[0]), None);
        }
    }
}

/// The image with everything light grown by `radius` pixels, as a camera
/// shows bright shapes on a dark ground: light bleeds outwards.
fn bloomed(image: &Rgb, radius: usize) -> Rgb {
    let (w, h) = (image.width, image.height);
    let mut pixels = image.pixels.clone();
    // The largest value within `radius`, along rows and then along columns.
    for vertical in [false, true] {
        let before = pixels.clone();
        for y in 0..h {
            for x in 0..w {
                for channel in 0..3 {
                    let (at, length) = if vertical { (y, h) } else { (x, w) };
                    let range = at.saturating_sub(radius)..(at + radius + 1).min(length);
                    let brightest =
                        range.map(|i| if vertical { before[3 * (i * w + x) + channel] } else { before[3 * (y * w + i) + channel] }).max();
                    pixels[3 * (y * w + x) + channel] = brightest.expect("a pixel is within reach of itself");
                }
            }
        }
    }
    Rgb { width: w, height: h, pixels }
}

#[test]
fn light_stars_that_look_bigger_than_they_are_still_scan() {
    let text = "light ink on a dark ground, photographed from a screen";
    let symbols = code::encode(&Payload::from_bytes(text), 1).unwrap();
    assert_eq!(symbols.len(), 6);
    // Stars about 26, 40 and 58 pixels tall, grown by one, two and three pixels all round.
    for (size, radius) in [(450, 1), (700, 2), (1000, 3)] {
        let style = Style { ink: "#febc5c".into(), background: "#021639".into(), size, ..Style::default() };
        let found = scan(&bloomed(&render(&symbols, &style), radius));
        assert_eq!(found.len(), 1, "size {size}, grown by {radius}");
        assert_eq!(found[0].payload.as_text(), Some(text));
    }
}

#[test]
fn a_small_payload_in_a_bigger_code_scans() {
    let symbols = code::encode_with_rings(&Payload::from_bytes("hi"), 2, 3).unwrap();
    let found = scan(&render(&symbols, &Style { size: 700, ..Style::default() }));
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].payload.as_text(), found[0].rings, found[0].level), (Some("hi"), 3, 2));
}

#[test]
fn ring_0_of_a_bigger_code_is_not_a_code_by_itself() {
    // Paint over everything outside ring 0 of a three-ring code.
    let symbols = code::encode_with_rings(&Payload::from_bytes("cropped"), 1, 3).unwrap();
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
