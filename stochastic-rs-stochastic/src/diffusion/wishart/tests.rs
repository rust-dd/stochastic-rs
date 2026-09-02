//! Tests of the exact Wishart sampler against the closed forms of Ahdida &
//! Alfonsi (2013): the affine mean, the Laplace transform (10) and the
//! characteristic-function values of their Table 1.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// Symmetric with a non-negative spectrum, judged by the extended Cholesky's
/// own tolerance (it panics on a significantly negative Schur complement).
fn assert_in_the_cone(x: &Array2<f64>) {
  assert_symmetric(x, "path matrix");
  let _ = extended_cholesky(x);
}

fn terminal_matrices(process: &Wishart<f64, Deterministic>, paths: usize) -> Vec<Array2<f64>> {
  process
    .sample_par(paths)
    .into_iter()
    .map(|p| p.slice(s![p.dim().0 - 1, .., ..]).to_owned())
    .collect()
}

fn general_two_dimensional(seed: u64) -> Wishart<f64, Deterministic> {
  Wishart::new(
    2.5,
    array![[-0.5_f64, 0.1], [0.05, -0.3]],
    array![[0.3_f64, 0.1], [0.0, 0.2]],
    array![[1.0_f64, 0.2], [0.2, 0.5]],
    5,
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// `E[X_T] = m_T x₀ m_Tᵀ + α q_T` for a full drift and a non-diagonal `a`,
/// over four exact steps; best of three seeds against a 3σ band.
#[test]
fn terminal_mean_matches_the_affine_moment_formula() {
  let best = [11_u64, 22, 33]
    .into_iter()
    .map(|seed| {
      let process = general_two_dimensional(seed);
      let want = process.mean(1.0);
      let terminal = terminal_matrices(&process, 20_000);
      let n = terminal.len() as f64;
      let mut worst = 0.0_f64;
      for i in 0..2 {
        for j in 0..2 {
          let got = terminal.iter().map(|x| x[(i, j)]).sum::<f64>() / n;
          worst = worst.max((got - want[(i, j)]).abs());
        }
      }
      worst
    })
    .fold(f64::INFINITY, f64::min);
  assert!(best < 0.02, "worst entry deviation {best}");
}

/// Monte Carlo `E[exp(Tr(v X_T))]` against eq. (10) for a negative definite
/// `v`, which pins the whole law of `X_T` rather than its first moment.
#[test]
fn laplace_transform_matches_the_monte_carlo_average() {
  let v = array![[-0.4_f64, -0.12], [-0.12, -0.4]];
  let best = [5_u64, 6, 7]
    .into_iter()
    .map(|seed| {
      let process = general_two_dimensional(seed);
      let want = process.laplace_transform(&v, 1.0);
      let terminal = terminal_matrices(&process, 20_000);
      let got = terminal
        .iter()
        .map(|x| {
          let vx = v.dot(x);
          (vx[(0, 0)] + vx[(1, 1)]).exp()
        })
        .sum::<f64>()
        / terminal.len() as f64;
      (got - want).abs()
    })
    .fold(f64::INFINITY, f64::min);
  assert!(best < 0.01, "Laplace transform deviation {best}");
}

/// Table 1 of Ahdida & Alfonsi (2013) for `a = I₃`, `b = 0`, `x₀ = 10 I₃`,
/// `v = 0.09 I₃`, sampled in one exact step. The table lists
/// `−0.527090 − 0.228251 i` (α = 3.5) and `−0.591411 − 0.036346 i` (α = 2.2);
/// evaluating their eq. (11) at `v = ±0.09 i I₃` shows the listed imaginary
/// sign is that of `E[exp(+i Tr(v X_1))] = E[cos] + i E[sin]`, which is what
/// the Monte Carlo average below forms.
/// `α = 3.5` is the regular regime; `α = 2.2` puts the squared Bessel degree
/// at `0.2 < 1`, the Poisson-mixture branch of the noncentral χ².
#[test]
fn characteristic_function_matches_ahdida_alfonsi_table_1() {
  for (alpha, want_re, want_im) in [
    (3.5_f64, -0.527090_f64, -0.228251_f64),
    (2.2, -0.591411, -0.036346),
  ] {
    let best = [1_u64, 2, 3]
      .into_iter()
      .map(|seed| {
        let process = Wishart::new(
          alpha,
          Array2::<f64>::zeros((3, 3)),
          Array2::<f64>::eye(3),
          Array2::<f64>::eye(3) * 10.0,
          2,
          Some(1.0),
          Deterministic::new(seed),
        );
        let terminal = terminal_matrices(&process, 40_000);
        let n = terminal.len() as f64;
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for x in &terminal {
          let phase = 0.09 * (x[(0, 0)] + x[(1, 1)] + x[(2, 2)]);
          re += phase.cos();
          im += phase.sin();
        }
        (re / n - want_re).abs().max((im / n - want_im).abs())
      })
      .fold(f64::INFINITY, f64::min);
    assert!(best < 0.015, "alpha {alpha}: deviation {best}");
  }
}

/// With `a = diag(1, 0)` the noise acts on one coordinate only, so the
/// second row and column stay exactly zero and the corner is a CIR with
/// mean `α t`.
#[test]
fn rank_deficient_volatility_keeps_the_process_in_the_lower_rank_cone() {
  let process = Wishart::new(
    1.5,
    Array2::<f64>::zeros((2, 2)),
    array![[1.0_f64, 0.0], [0.0, 0.0]],
    Array2::<f64>::zeros((2, 2)),
    4,
    Some(1.0),
    Deterministic::new(4),
  );
  assert_eq!(process.noise_rank(), 1);
  let paths = process.sample_par(4_000);
  let mut corner = 0.0_f64;
  for p in &paths {
    for j in 0..4 {
      assert_eq!(p[(j, 1, 1)], 0.0);
      assert_eq!(p[(j, 0, 1)], 0.0);
      assert_eq!(p[(j, 1, 0)], 0.0);
      assert!(p[(j, 0, 0)] >= 0.0);
    }
    corner += p[(3, 0, 0)];
  }
  corner /= paths.len() as f64;
  assert!((corner - 1.5).abs() < 0.08, "corner mean {corner}");
}

/// `α = d − 1` with a positive definite start drives the squared Bessel
/// degree to zero; the Poisson-mixture boundary sampler keeps the mean
/// `x₀ + α t I` and the paths in the cone.
#[test]
fn boundary_degree_uses_the_dimension_zero_squared_bessel() {
  let process = Wishart::new(
    1.0,
    Array2::<f64>::zeros((2, 2)),
    Array2::<f64>::eye(2),
    Array2::<f64>::eye(2),
    3,
    Some(1.0),
    Deterministic::new(8),
  );
  let paths = process.sample_par(10_000);
  let mut mean = Array2::<f64>::zeros((2, 2));
  for p in &paths {
    for j in 0..3 {
      assert_in_the_cone(&p.slice(s![j, .., ..]).to_owned());
    }
    mean += &p.slice(s![2, .., ..]);
  }
  mean /= paths.len() as f64;
  let want = process.mean(1.0);
  for i in 0..2 {
    for j in 0..2 {
      assert!((want[(i, j)] - if i == j { 2.0 } else { 0.0 }).abs() < 1e-12);
      assert!(
        (mean[(i, j)] - want[(i, j)]).abs() < 0.06,
        "entry {i},{j}: {}",
        mean[(i, j)]
      );
    }
  }
}

/// `d = 1` is CIR: `E[X_t] = e^{2bt} x₀ + α a² (e^{2bt} − 1) / (2b)`, so the
/// block-exponential drift maps must reproduce that closed form.
#[test]
fn one_dimensional_mean_is_the_cir_mean() {
  let process = Wishart::new(
    2.0,
    array![[-0.8_f64]],
    array![[0.4_f64]],
    array![[0.5_f64]],
    8,
    Some(0.7),
    Deterministic::new(1),
  );
  let t = 0.7_f64;
  let e = (2.0 * -0.8 * t).exp();
  let want = e * 0.5 + 2.0 * 0.16 * (e - 1.0) / (2.0 * -0.8);
  assert!((process.mean(t)[(0, 0)] - want).abs() < 1e-12);
  let path = process.sample();
  assert_eq!(path.dim(), (8, 1, 1));
  assert_eq!(path[(0, 0, 0)], 0.5);
  assert!(path.iter().all(|x| *x >= 0.0));
}

#[test]
fn deterministic_seed_reproduces_and_consecutive_paths_differ() {
  let a = general_two_dimensional(7).sample();
  let b = general_two_dimensional(7).sample();
  assert_eq!(a, b);
  let mut sampler = general_two_dimensional(7).sampler();
  let first = sampler.sample();
  let second = sampler.sample();
  assert_ne!(first, second);
  let chunks = general_two_dimensional(7).sample_par(2);
  assert_ne!(chunks[0], chunks[1]);
}

#[test]
#[should_panic(expected = "alpha must be at least d - 1")]
fn rejects_a_degree_below_d_minus_one() {
  let _ = Wishart::new(
    0.9,
    Array2::<f64>::zeros((2, 2)),
    Array2::<f64>::eye(2),
    Array2::<f64>::eye(2),
    4,
    None,
    Unseeded,
  );
}

#[test]
#[should_panic(expected = "not positive semidefinite")]
fn rejects_an_indefinite_start() {
  let _ = Wishart::new(
    3.0,
    Array2::<f64>::zeros((2, 2)),
    Array2::<f64>::eye(2),
    array![[1.0_f64, 2.0], [2.0, 1.0]],
    4,
    None,
    Unseeded,
  );
}
