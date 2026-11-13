# Astracode in Rust

The reference implementation of Astracode: the format, a generator and the
scanner. This is the one that ships on phones, on the web and on servers, and
the one that is developed.

The scanner is the only one there is. It began as a Python prototype, which
was removed once this one had caught up with it (it is in git history up to
`v0.3.0`). Python is still used for two things: a description of the format
(`../astracode/spec.py`, `astra.py`), which writes the test vectors this
library is checked against, and the benchmark (`../bench.py`), which makes
distorted frames and scans them with this scanner.

```
rust/
  astracode/          the library
  astracode-cli/      the `astracode` command: scan images, draw codes
  astracode-android/  the scanner as a native library for the Android app
```

## Using it

```sh
cargo build --release
target/release/astracode encode "https://example.org/" -o code.png   # as many rings as it needs
target/release/astracode encode --hex c0ffee -o code.svg             # any bytes; up to 4 fit in one ring
target/release/astracode astra 4320880580 --metadata 5 -o code.svg   # an Astra member number: one ring
target/release/astracode scan --time photo.jpg
```

From Rust:

```rust
use astracode::astra::Number;
use astracode::format::{code, Payload};
use astracode::scan::{scan_rgb8, Options};

// Making a code: the payload decides how many rings it gets.
let rings = code::encode(&Payload::from_bytes("hello"), 1)?;      // error-correction level 1
let svg = astracode::format::svg::svg(&rings, &Default::default());

// Reading codes.
for code in scan_rgb8(width, height, &rgb_pixels, &Options::default()) {
    if let Some(number) = Number::from_payload(&code.payload) {
        println!("Astra member {}", number.member_id());
    } else if let Some(text) = code.payload.as_text() {
        println!("{} rings: {text}", code.rings);
    } else {
        println!("{} rings: {} bits", code.rings, code.payload.len());
    }
}
```

A code holds a `Payload`: a string of bits, usually whole bytes, that the
format gives no meaning. It has as many rings as the payload needs; one ring
holds 38 bits. `astra` is one convention on top, kept out of the format and
the scanner: a payload of exactly 38 bits is an Astra number (metadata and a
ten-digit payload), which therefore always fills one ring and can never be
mistaken for a byte string.

`scan_gray` takes a single brightness plane, which is what a camera preview
gives and avoids the colour-channel fallback. `Options::find_size` looks for
stars in a shrunken copy of the frame, for a live preview where frame rate
matters more than small codes.

## On Android

The app in `../android` contains no scanner of its own. `astracode-android`
exposes one native function, `NativeScanner.scan`, which takes the camera
frame's brightness plane and returns the codes found packed into a byte array;
the layout is documented at `encode` in its `lib.rs` and unpacked by
`Scanner.kt`.

`../android/build-rust.sh` builds it with the NDK into the app's `jniLibs`,
and Gradle runs that script before every build, so `./gradlew assembleDebug`
is all that is needed once the Rust target is installed:

```sh
rustup target add aarch64-linux-android
```

## How the library is laid out

| Module | What it is |
|---|---|
| `format` | The code itself, with no image code: constants and ring layout (`mod.rs`), the payload as a string of bits (`payload`), finite fields (`gf`), Reed-Solomon (`reed_solomon`), the scramble (`mix`), ring 0 (`ring0`), a whole code from payload to stars and back (`code`), star geometry (`geometry`), SVG output (`svg`). |
| `astra` | Astra's own use of the code: a member number in one ring. Nothing else depends on it. |
| `image` | A grey `f32` image: sampling, blurs, shrinking, perspective warp. |
| `math` | Points, local linear maps, homographies (fit, compose, invert). |
| `scan` | The scanner. |

The format is described at the top of `format/mod.rs`. In short: ring `k` has
`12 + 6k` identical upright stars at radius `100 + 40k`; each star's five limbs
are solid or a lighter shade, five bits per star. Ring 0 holds a 40-bit value
protected by four check stars: how many rings the code has, and the first bits
of the payload. If the payload is short enough (38 bits) that is the whole
code; otherwise the stars of the other rings together form one long
Reed-Solomon codeword carrying the rest.

## How the scanner works

`scan/mod.rs` drives these stages, cheapest first:

1. **`blobs`** Binarise against the local mean, in both polarities and at two
   neighbourhood sizes, and collect connected regions that could be stars.
2. **`rings`** Walk rings of twelve evenly spaced, similar blobs by predicting
   each next position from the last few.
3. **`candidate`** For each ring: fit a homography; reject look-alikes
   (**`checks`**); find which of the twelve rotations is upright from the star
   outlines and refine the fit (**`align`**); read ring 0's limbs
   (**`limbs`**). Ring 0 says how many rings there are; the result is a
   whole one-ring code, the start of a bigger one, or nothing.
4. **`outer`** For a bigger code: locate the remaining rings outwards from
   ring 0, each from its already located neighbours, and decode them.
5. **`bent`** When a ring cannot be read as flat: fit each star's own
   outline, which gives its local stretch, shear and rotation, and settles
   the ring's rotation; then read again.

Things worth knowing before changing it:

* **It must never return a wrong value.** Every acceptance rule errs towards
  returning nothing; the caller scans the next frame. Repairs are limited on
  purpose: one star in ring 0 (or two the reader already doubted), and the
  outer codeword always keeps two check symbols back for detection.
* **The outer error correction is told what the reader doubts.** If the plain
  repair fails, the least certain star pairs are treated as unknown, which
  costs half as many check symbols each (`code::decode_outer_doubting`). This
  does little against even noise, where the doubtful stars are right half the
  time, and a lot against glare or anything covering part of a code, where
  they are simply gone.
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
  where ring 0 would be and gives up after a fixed number of candidates. Each
  pass of the search has its own allowance of loops, so that texture or noise
  that fills one pass with chance loops does not keep the scanner from the
  pass that has the code's stars.
* **The star-finding threshold rises with the frame's noise**
  (`blobs::noise_level`, measured in the flattest parts of the frame).
  Otherwise noise itself becomes thousands of small blobs.

## Testing

```sh
cargo test                 # unit tests, format vectors written by the Python format description,
                           # and end-to-end tests that draw, distort and scan codes
```

The benchmark makes distorted frames in Python and scans them with this
scanner:

```sh
cargo build --release
../.venv/bin/python ../bench.py 300                # one ring
../.venv/bin/python ../bench.py 150 --rings 6-12
../.venv/bin/python ../bench.py 300 --wavy 0.3     # bent surfaces
```

It reports how many frames were read, missed and read wrongly (which must be
none), scan times, and the read rate by star size, tilt, blur and so on.

## Performance

Measured on an Apple M-series laptop with 9 threads, on the benchmark's
distorted frames (medians):

| Frame | Read | No code in frame |
|---|---|---|
| One ring, 960 x 720 | 4 ms | 7 ms |
| 2 to 5 rings, 1280 x 960 | 6 ms | 11 ms |
| 6 to 12 rings, 1280 x 960 | 9 ms | 14 ms |
| 14 to 24 rings, 2560 x 1920 | 24 ms | 31 ms |

On one core, a clean phone frame (1440 x 1920) with the code filling it takes
about 15 ms for one to three rings, 23 ms for five, 33 ms for twelve and 78 ms
for twenty-four; a frame with no code in it, 5 ms per plane tried.

Per-star work and the image passes run on all cores through `rayon`; build
with `--no-default-features` for a single-threaded library. To see where time
goes, `astracode scan --repeat 200 --threads 1 image.png` gives a profiler
something steady to sample; the release profile keeps function names.

Where the time is, after the second round of optimisation:

* finding stars is one pass over the image (smoothing, thresholding against
  local means kept per 4-pixel cell, one byte per pixel out) and then blob
  labelling. It runs up to four times per RGB frame, because each colour
  channel is tried when brightness finds nothing; a grey frame, which is what
  a phone camera hands over, is gone through once;
* matching the star outline dominates reading: around ring 0 while aligning,
  and around every outer star while locating it. The outer stars are matched
  in a straightened copy of the image, of which only the band the next rings
  are in is made;
* the outer stars are looked for quickly first: on a coarse grid, at one star
  size once the place is known. That finds a flat surface's stars as well as a
  careful search does, at a third of the cost. Only if the code cannot be read
  from that, and the stars found say the surface bends, does the careful
  search run, followed by fitting every star's own shape, the dearest step
  there is;
* so a big code that is found but cannot be read costs little more than one
  that can, as long as it is on a flat surface, which is most of them.

What did not pay: taking the nearest pixel instead of interpolating when
straightening (a tenth quicker on big codes, six reads fewer in 810); and the
quick search on its own, without the careful one behind it (three reads fewer
in 840 on bent surfaces).
