//! Philox-4x32-10, its key, its uniform maps and the counter sites of the Euler kernels; the Metal
//! kernel crate includes this file, so it takes no `std` and no transcendental.
//! Salmon, Moraes, Dror & Shaw (2011), "Parallel random numbers: as easy as 1, 2, 3", SC'11.

/// Normal noise, one site per component (`+ k`, `k < 4`); the counter is `(path, step, site, 0)`.
pub const SITE_NOISE: u32 = 0;

/// The step's four uniforms, one word each.
pub const SITE_UNIFORMS: u32 = 4;

/// The Poisson count, two uniforms a call; the index counts the calls.
pub const SITE_COUNT: u32 = 5;

/// Jump sizes: index 0 for a one-draw law, the jump's number for a compound one.
pub const SITE_SIZES: u32 = 6;

/// The first gamma draw (`+ 1` the second): index `2t` the try's normal, `2t + 1` its acceptance
/// uniform, [`GAMMA_BOOST`] the shape boost's uniform.
pub const SITE_GAMMA: u32 = 7;

/// A series term's arrival gap and exponential, at step 0 and index `j`.
pub const SITE_SERIES: u32 = 9;

/// A series term's arrival time on the horizon, at step 0 and index `j`.
pub const SITE_SERIES_TIME: u32 = 10;

/// A clock table node's two uniforms, at step `attempt` and index `node`.
pub const SITE_TABLE: u32 = 11;

/// A series term's two uniforms, at step 0 and index `j`.
pub const SITE_SERIES_MARKS: u32 = 12;

/// Past the 24 tries' indices, where the shape boost draws.
pub const GAMMA_BOOST: u32 = 48;

/// `23 ln 2`: an exponential's zero cell `[0, 2⁻²³)` refined by a second word starts here.
pub const LN_2_23: f32 = 15.942_385;

const M0: u32 = 0xD251_1F53;

const M1: u32 = 0xCD9E_8D57;

const W0: u32 = 0x9E37_79B9;

const W1: u32 = 0xBB67_AE85;

/// The whole 64-bit seed as the two key words.
pub fn key(seed: u64) -> [u32; 2] {
  [seed as u32, (seed >> 32) as u32]
}

/// Ten rounds on the counter, round `r` under the key bumped `r` times by the Weyl constants.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
  let [mut c0, mut c1, mut c2, mut c3] = counter;
  let [k0, k1] = key;
  // Unrolled: the Metal compiler's numerical proofs visit straight-line code once per call site.
  macro_rules! rounds {
    ($($round:literal)*) => {$(
      let low = (M0 as u64).wrapping_mul(c0 as u64);
      let high = (M1 as u64).wrapping_mul(c2 as u64);
      let first = ((high >> 32) as u32) ^ c1 ^ k0.wrapping_add(W0.wrapping_mul($round));
      let third = ((low >> 32) as u32) ^ c3 ^ k1.wrapping_add(W1.wrapping_mul($round));
      c0 = first;
      c1 = high as u32;
      c2 = third;
      c3 = low as u32;
    )*};
  }
  rounds!(0 1 2 3 4 5 6 7 8 9);
  [c0, c1, c2, c3]
}

/// `(2j + 1)·2⁻²⁴` from the top 23 bits: exact in `f32`, never 0 or 1.
pub fn u01_f32(word: u32) -> f32 {
  ((word >> 9) as f32 + 0.5) * f32::EPSILON
}

/// `(2j + 1)·2⁻⁵³` from 52 bits, all of the first word and the second's top 20: never 0 or 1.
pub fn u01_f64(first: u32, second: u32) -> f64 {
  ((((first as u64) << 20) | (second >> 12) as u64) as f64 + 0.5) * f64::EPSILON
}

/// `(2w + 1)·2⁻³³` from one whole word, the `f64` kernels' per-step uniform.
pub fn u01_f64_word(word: u32) -> f64 {
  (word as f64 + 0.5) * (1.0 / 4_294_967_296.0)
}
