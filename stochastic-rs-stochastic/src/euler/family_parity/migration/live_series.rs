//! Live-series storage and the independently reconstructed 512th term.

use super::*;

fn process(terms: usize) -> crate::volatility::svcgmy::Svcgmy<f32, Deterministic> {
  crate::volatility::svcgmy::Svcgmy::new(
    3.0,
    4.0,
    0.7,
    1.0,
    0.25,
    1.0,
    0.3,
    N,
    terms,
    Some(0.0),
    Some(0.25),
    Some(1.0),
    Deterministic::new(11),
  )
}

#[test]
fn live_series_accepts_the_final_private_slot() {
  for terms in [511, 512] {
    let process = process(terms);
    assert!(process.device_ready(), "{terms} terms should fit");
    assert_eq!(
      crate::euler::series_terms(115, N, process.series_terms()),
      terms as u32,
    );
  }
  assert!(process(513).device_fallback().is_some());
  assert!(std::panic::catch_unwind(|| crate::euler::series_terms(115, N, Some(513))).is_err());
}

#[test]
fn live_series_storage_preserves_one_based_random_counters() {
  use crate::euler::families::Family;
  use crate::euler::kernel;

  let shape = kernel::Shape::new(Family::StochasticVolatilityCgmy, false, false);
  for (name, language) in [
    ("CUDA f32", kernel::cuda_language("float")),
    ("CUDA f64", kernel::cuda_language("double")),
  ] {
    let source = kernel::render_for(&language, shape);
    assert!(
      source.contains("block[512]"),
      "{name}: live storage capacity"
    );
    assert!(
      source.contains("block[j - 1u] = gj;") && source.contains("gj = block[j - 1u];"),
      "{name}: all 512 one-based terms must fit in the zero-based storage",
    );
    assert_eq!(
      source.matches("(j * 40503u)").count(),
      2,
      "{name}: draw and replay must keep the original one-based term counter",
    );
  }
}

fn base() -> SystemProbe<2> {
  SystemProbe {
    spec: EulerSpec::StochasticVolatilityCgmy {
      rate0: 100.0,
      inv_alpha: 1.0,
      lambda_plus: 0.001,
      lambda_minus: 0.001,
      bcoef: 0.0,
      twoc: 0.01,
      ek: 0.0,
      rho: 0.0,
    },
    x0: [0.0, 100.0],
    lift: None,
    program: None,
  }
}

fn avalanche(mut value: u32) -> u32 {
  value ^= value >> 16;
  value = value.wrapping_mul(2_246_822_519);
  value ^= value >> 13;
  value = value.wrapping_mul(3_266_489_917);
  value ^ (value >> 16)
}

fn term_uniform(path: u32, term: u32, seed: u32, salt: u32) -> f32 {
  let key = path.wrapping_mul(2_654_435_761) ^ term.wrapping_mul(40_503) ^ 3_266_489_917;
  let word = avalanche(key ^ salt ^ seed.wrapping_mul(2_654_435_761));
  word as f32 * 2.328_306_4e-10
}

fn exponential(uniform: f32) -> f32 {
  -(uniform * 0.999_998 + 1.0e-6).ln()
}

/// The reference keeps no indexed arrival cache: it accumulates the clock
/// once and adds each term directly to the cell identified by its own hash.
fn reference(paths: &Array3<f32>, first: usize, seed: u64) -> (Array3<f32>, Array3<f32>) {
  let seed = (seed ^ (seed >> 32)) as u32;
  let mut expected = Array3::<f32>::zeros((1, ROWS, N));
  let mut last = Array3::<f32>::zeros((1, ROWS, N));
  for path in 0..ROWS {
    let mut arrival = 0.0f32;
    for term in 1..=512 {
      let uniform = |salt| term_uniform((first + path) as u32, term, seed, salt);
      arrival += exponential(uniform(2_654_435_769));
      let cell = (uniform(374_761_393) * (N - 1) as f32).ceil() as usize;
      let cell = cell.clamp(1, N - 1);
      let variance = paths[[1, path, cell - 1]].max(1.0e-12);
      let cap = 1.0 / (arrival * 100.0 / variance);
      let draw = exponential(uniform(2_246_822_507)) * uniform(3_266_489_909) / 0.001;
      let magnitude = cap.min(draw);
      let size = if uniform(668_265_263) < 0.5 {
        magnitude
      } else {
        -magnitude
      };
      expected[[0, path, cell]] += size;
      if term == 512 {
        last[[0, path, cell]] = size;
      }
    }
    for step in 1..N {
      expected[[0, path, step]] += expected[[0, path, step - 1]];
      last[[0, path, step]] += last[[0, path, step - 1]];
    }
  }
  (expected, last)
}

#[test]
#[ignore = "Run the live-series capacity regression explicitly on Metal hardware"]
fn all_512_live_terms_match_an_independent_reference() {
  let device = Metal::default();
  let base = base();
  let mut probe = ShapeProbe {
    base: &base,
    jumps: false,
    gamma: false,
    step_first: false,
    terms: Some(512),
  };
  for seed in [SEEDS[0], SEEDS[2]] {
    let full = launch(&device, &probe, FIRST, ROWS, seed);
    assert!(full.iter().all(|value| value.is_finite()));
    let repeated = launch(&device, &probe, FIRST, ROWS, seed);
    assert_same(&repeated, &full, "512-term repeat");
    check_chunking(&device, &probe, &full, FIRST, seed, "512-term chunks");
    let (expected, last) = reference(&full, FIRST, seed);
    probe.terms = Some(511);
    let previous = launch(&device, &probe, FIRST, ROWS, seed);
    probe.terms = Some(512);
    let mut observable_last_term = false;
    for path in 0..ROWS {
      for step in 0..N {
        let actual = full[[0, path, step]];
        let want = expected[[0, path, step]];
        let tolerance = 2.0e-5 * (1.0 + want.abs());
        assert!(
          (actual - want).abs() <= tolerance,
          "512-term reference: seed={seed}, path={path}, step={step}: {actual} != {want}",
        );
        assert_eq!(
          full[[1, path, step]].to_bits(),
          previous[[1, path, step]].to_bits(),
          "series count must not change the variance noise",
        );
        let delta = actual - previous[[0, path, step]];
        let want_delta = last[[0, path, step]];
        assert!(
          (delta - want_delta).abs() <= tolerance,
          "term 512: seed={seed}, path={path}, step={step}: {delta} != {want_delta}",
        );
        observable_last_term |= want_delta.abs() > 10.0 * tolerance;
      }
    }
    assert!(
      observable_last_term,
      "term 512 must exceed the comparison tolerance"
    );
  }
}
