# Astracode in Rust

The production implementation of Astracode: the format, a generator and the
scanner. It is a port of the Python reference in `../astracode/`, which remains
the place where the algorithm is worked out; this is the one meant to ship on
phones, on the web and on servers.

```
rust/
  astracode/        the library
  astracode-cli/    the `astracode` command: scan images, draw codes
```

## Using it

```sh
cargo build --release
target/release/astracode number 4320880580 --metadata 5 -o code.svg
target/release/astracode message --rings 4 "https://example.org/" -o code.png
target/release/astracode scan --time photo.jpg
```

From Rust:

```rust
use astracode::scan::{scan_rgb8, Options, Payload};

for code in scan_rgb8(width, height, &rgb_pixels, &Options::default()) {
    match code.payload {
        Payload::Number(value) => println!("one ring: {value}"),
        Payload::Message { bytes, .. } => println!("{} rings: {} bytes", code.rings, bytes.len()),
    }
}
```

`scan_gray` takes a single brightness plane, which is what a camera preview
gives and avoids the colour-channel fallback. `Options::find_size` looks for
stars in a shrunken copy of the frame, for a live preview where frame rate
matters more than small codes.

## How the library is laid out

| Module | What it is |
|---|---|
| `format` | The code itself, with no image code: constants and ring layout (`mod.rs`), finite fields (`gf`), Reed-Solomon (`reed_solomon`), the scramble (`mix`), ring 0 and one-ring codes (`ring0`), multi-ring messages (`message`), star geometry (`geometry`), SVG output (`svg`). |
| `image` | A grey `f32` image: sampling, blurs, shrinking, perspective warp. |
| `math` | Points, local linear maps, homographies (fit, compose, invert). |
| `scan` | The scanner. |

The format is described at the top of `format/mod.rs`. In short: ring `k` has
`12 + 6k` identical upright stars at radius `100 + 40k`; each star's five limbs
are solid or a lighter shade, five bits per star. Ring 0 holds a 40-bit value
protected by four check stars. Either that value is a complete one-ring code,
or it is a header announcing more rings, whose stars together form one long
Reed-Solomon codeword carrying a byte string.

## How the scanner works

`scan/mod.rs` drives these stages, cheapest first:

1. **`blobs`** Binarise against the local mean, in both polarities and at two
   neighbourhood sizes, and collect connected regions that could be stars.
2. **`rings`** Walk rings of twelve evenly spaced, similar blobs by predicting
   each next position from the last few.
3. **`candidate`** For each ring: fit a homography; reject look-alikes
   (**`checks`**); find which of the twelve rotations is upright from the star
   outlines and refine the fit (**`align`**); read ring 0's limbs
   (**`limbs`**). The result is a one-ring code, a header, or nothing.
4. **`outer`** For a header: locate the remaining rings outwards from ring 0,
   each from its already located neighbours, and decode the message.
5. **`bent`** When a ring cannot be read as flat: fit each star's own
   outline, which gives its local stretch, shear and rotation, and settles
   the ring's rotation; then read again.

Things worth knowing before changing it:

* **It must never return a wrong value.** Every acceptance rule errs towards
  returning nothing; the caller scans the next frame. Repairs are limited on
  purpose: one star in ring 0 (or two the reader already doubted), and the
  outer codeword always keeps two check symbols back for detection.
* **Nothing depends on colour, on a particular light-limb shade, or on what is
  drawn in the centre.** The two shades are separated per photo by a small
  model (`limbs::classify`).
* **A `Homography` maps model units to image pixels.** A `Placement` holds,
  per star, where it is in the image and its `LocalMap`: how model offsets
  around it appear there. Flat reading takes both from the homography; bent
  reading fits them per star.
* **Polarity** is `1.0` for dark ink on light and `-1.0` for the reverse, and
  is carried in `View` next to the image.
* **A frame has a budget** (`scan::Budget`): a code with many rings contains
  hundreds of loops of twelve stars that are not ring 0, so the search starts
  where ring 0 would be and gives up after a fixed number of candidates.

## Testing

```sh
cargo test                 # unit tests, format vectors from the Python reference,
                           # and end-to-end tests that draw, distort and scan codes
```

The Python benchmark can drive this scanner on exactly the frames it uses for
the reference, and compare the two:

```sh
cargo build --release
../.venv/bin/python ../tools/rust_bench.py 300                # one ring
../.venv/bin/python ../tools/rust_bench.py 150 --rings 6-12
../.venv/bin/python ../tools/rust_bench.py 300 --wavy 0.3     # bent surfaces
```

At the time of writing both scanners read the same number of frames in every
set, with no wrong reads.

## Performance

Measured on an Apple M-series laptop with 9 threads, on the benchmark's
distorted frames (medians):

| Frame | Read | No code in frame |
|---|---|---|
| One ring, 960 x 720 | 5 ms | 9 ms |
| 2 to 5 rings, 1280 x 960 | 8 ms | 15 ms |
| 6 to 12 rings, 1280 x 960 | 17 ms | 17 ms |
| 14 to 24 rings, 2560 x 1920 | 62 ms | 43 ms |

Per-star work and the image passes run on all cores through `rayon`; build
with `--no-default-features` for a single-threaded library. To see where time
goes, `astracode scan --repeat 200 --threads 1 image.png` gives a profiler
something steady to sample; the release profile keeps function names.

Where the time is, after the first round of optimisation:

* finding stars (two blurs, thresholding, blob labelling) dominates frames
  without a code, and runs up to four times per RGB frame because each colour
  channel is tried when brightness finds nothing;
* matching the star outline dominates reading: around ring 0 while aligning,
  and around every outer star while locating it.
