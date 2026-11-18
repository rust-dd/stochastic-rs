use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn model(hurst: f64, n: usize, seed: u64) -> RoughBergomi<f64, Deterministic> {
  RoughBergomi::new(
    hurst,
    0.4,
    0.2,
    100.0,
    0.0,
    -0.6,
    n,
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// Bennedsen–Lunde–Pakkanen (2017) Proposition 2.8: the `k = 2` weight is
/// `Δ^α (2^{α+1} − 1)/(α+1)`; at `H = ½` every weight is one and the exact integral the increment.
#[test]
fn hybrid_weights_match_the_closed_forms() {
  let (kernel, on_increment, residual_sd) =
    RoughBergomiSampler::<f64, Deterministic>::hybrid_weights(0.1, 0.01, 6);
  let alpha = -0.4;
  let expected_k2 = 0.01_f64.powf(alpha) * (2.0_f64.powf(alpha + 1.0) - 1.0) / (alpha + 1.0);
  assert!((kernel[2] - expected_k2).abs() < 1e-12);
  assert!((on_increment - 0.01_f64.powf(alpha) / (alpha + 1.0)).abs() < 1e-12);
  let expected_var =
    0.01_f64.powf(2.0 * alpha + 1.0) * (1.0 / (2.0 * alpha + 1.0) - 1.0 / (alpha + 1.0).powi(2));
  assert!((residual_sd * residual_sd - expected_var).abs() < 1e-14);
  assert!(
    kernel[2] > kernel[3] && kernel[3] > kernel[4],
    "weights decay"
  );
  let (brownian, on_inc, sd) =
    RoughBergomiSampler::<f64, Deterministic>::hybrid_weights(0.5, 0.01, 6);
  assert!(brownian[2..].iter().all(|w| (w - 1.0).abs() < 1e-12));
  assert!((on_inc - 1.0).abs() < 1e-12 && sd.abs() < 1e-9);
}

#[test]
fn the_hybrid_weights_keep_their_digits_in_single_precision() {
  let (w, _, _) = RoughBergomiSampler::<f32, Unseeded>::hybrid_weights(0.1, 1.0 / 4096.0, 4001);
  let exact = [
    (2usize, 23.944_391_87_f64),
    (100, 4.424_006_643),
    (1000, 1.758_048_573),
    (4000, 1.009_582_234),
  ];
  for (k, e) in exact {
    assert!(
      (w[k] as f64 / e - 1.0).abs() < 1e-6,
      "k = {k}: got {}, exact {e}",
      w[k]
    );
  }
}

/// `E[v_t] = v_0²` for every `t`, since the `½ν²t^{2H}` compensator is exact when
/// `Var[W^H_t] = t^{2H}`, and the sample variance of the implied `W^H_t` tracks `t^{2H}`.
#[test]
fn variance_driver_is_a_martingale_with_t_2h_variance() {
  let n = 65;
  let paths = 20_000;
  let m = model(0.1, n, 7);
  let v2 = m
    .sample_par(paths)
    .iter()
    .map(|[_, v2]| v2.clone())
    .collect::<Vec<_>>();
  let nu = 0.4;
  for i in [8usize, 32, 64] {
    let t = i as f64 / 64.0;
    let mean: f64 = v2.iter().map(|p| p[i]).sum::<f64>() / paths as f64 / 0.04;
    assert!((mean - 1.0).abs() < 0.04, "t {t}: E[v]/v0² = {mean}");
    let logs: Vec<f64> = v2
      .iter()
      .map(|p| ((p[i] / 0.04).ln() + 0.5 * nu * nu * t.powf(0.2)) / nu)
      .collect();
    let mean_w = logs.iter().sum::<f64>() / paths as f64;
    let var_w = logs.iter().map(|w| (w - mean_w).powi(2)).sum::<f64>() / paths as f64;
    assert!(
      (var_w / t.powf(0.2) - 1.0).abs() < 0.05,
      "t {t}: Var[W^H] = {var_w}, t^2H = {}",
      t.powf(0.2)
    );
  }
}

/// The lag-one autocorrelation of the log-variance increments is fGN's `½(2^{2H} − 2)` at
/// `H = 0.1`, a negative number independent increments cannot produce.
#[test]
fn log_variance_increments_are_antipersistent() {
  let n = 257;
  let paths = 4_000;
  let m = model(0.1, n, 3);
  let (mut num, mut den) = (0.0, 0.0);
  for [_, v2] in m.sample_par(paths) {
    let w: Vec<f64> = v2.iter().map(|v| (v / 0.04).ln()).collect();
    let d: Vec<f64> = w.windows(2).map(|p| p[1] - p[0]).collect();
    for k in 128..d.len() - 1 {
      num += d[k] * d[k + 1];
      den += d[k] * d[k];
    }
  }
  let rho1 = num / den;
  let fgn = 0.5 * (2.0_f64.powf(0.2) - 2.0);
  assert!(
    rho1 < -0.3 && (rho1 - fgn).abs() < 0.08,
    "rho1 {rho1} vs fGN {fgn}"
  );
}
