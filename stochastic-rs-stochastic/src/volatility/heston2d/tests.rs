use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// MATLAB `Heston2D.m` example ordering over `(Z1, Z2, W1, W2)`.
fn rho_default<T: FloatExt>() -> [T; 6] {
  [
    T::from_f64_fast(0.5),
    T::from_f64_fast(-0.5),
    T::zero(),
    T::zero(),
    T::from_f64_fast(-0.5),
    T::from_f64_fast(0.5),
  ]
}

struct ValidInputs {
  x0: [Option<f64>; 2],
  v0: [Option<f64>; 2],
  mu: [f64; 2],
  theta: [f64; 2],
  kappa: [f64; 2],
  sigma: [f64; 2],
  rho: [f64; 6],
  t: Option<f64>,
}

impl Default for ValidInputs {
  fn default() -> Self {
    Self {
      x0: [Some(0.0), Some(0.0)],
      v0: [Some(0.4), Some(0.4)],
      mu: [0.0, 0.0],
      theta: [0.4, 0.4],
      kappa: [2.0, 2.0],
      sigma: [1.0, 1.0],
      rho: rho_default(),
      t: Some(1.0),
    }
  }
}

impl ValidInputs {
  fn build(self) -> Heston2D<f64, Deterministic> {
    Heston2D::new(
      self.x0,
      self.v0,
      self.mu,
      self.theta,
      self.kappa,
      self.sigma,
      self.rho,
      16,
      self.t,
      Some(false),
      Deterministic::new(17),
    )
  }
}

#[test]
fn shapes_match_n() {
  let h = Heston2D::<f64, _>::new(
    [Some(0.0), Some(0.0)],
    [Some(0.4), Some(0.4)],
    [0.0, 0.0],
    [0.4, 0.4],
    [2.0, 2.0],
    [1.0, 1.0],
    rho_default(),
    512,
    Some(1.0),
    Some(false),
    Deterministic::new(1),
  );
  let [x1, v1, x2, v2] = h.sample();
  assert_eq!(x1.len(), 512);
  assert_eq!(v1.len(), 512);
  assert_eq!(x2.len(), 512);
  assert_eq!(v2.len(), 512);
  assert!(v1.iter().all(|x| *x >= 0.0));
  assert!(v2.iter().all(|x| *x >= 0.0));
}

#[test]
fn seeded_is_deterministic() {
  let mk = || {
    Heston2D::<f64, Deterministic>::new(
      [Some(0.0), Some(0.0)],
      [Some(0.4), Some(0.4)],
      [0.0, 0.0],
      [0.4, 0.4],
      [2.0, 2.0],
      [1.0, 1.0],
      rho_default(),
      128,
      Some(1.0),
      Some(false),
      Deterministic::new(42),
    )
  };
  let [a, _, b, _] = mk().sample();
  let [c, _, d, _] = mk().sample();
  for i in 0..a.len() {
    assert!((a[i] - c[i]).abs() < 1e-12);
    assert!((b[i] - d[i]).abs() < 1e-12);
  }
}

/// A long constant-mean path recovers the configured price-driver correlation.
#[test]
fn cross_correlation_matches_rho() {
  let rho_w1w2 = 0.8_f64;
  let rho = [0.0, 0.0, 0.0, 0.0, 0.0, rho_w1w2];
  let h = Heston2D::<f64, Deterministic>::new(
    [Some(0.0), Some(0.0)],
    [Some(0.4), Some(0.4)],
    [0.0, 0.0],
    [0.4, 0.4],
    [2.0, 2.0],
    [0.5, 0.5],
    rho,
    20_000,
    Some(1.0),
    Some(false),
    Deterministic::new(7),
  );
  let [x1, _, x2, _] = h.sample();
  let r1 = Array1::from_iter((1..x1.len()).map(|i| x1[i] - x1[i - 1]));
  let r2 = Array1::from_iter((1..x2.len()).map(|i| x2[i] - x2[i - 1]));
  let n = r1.len() as f64;
  let mean1 = r1.iter().sum::<f64>() / n;
  let mean2 = r2.iter().sum::<f64>() / n;
  let cov = r1
    .iter()
    .zip(r2.iter())
    .map(|(a, b)| (a - mean1) * (b - mean2))
    .sum::<f64>()
    / n;
  let var1 = r1.iter().map(|a| (a - mean1).powi(2)).sum::<f64>() / n;
  let var2 = r2.iter().map(|b| (b - mean2).powi(2)).sum::<f64>() / n;
  let corr = cov / (var1.sqrt() * var2.sqrt());
  assert!(
    (corr - rho_w1w2).abs() < 0.05,
    "sample corr {corr:.4} far from target {rho_w1w2}"
  );
}

#[test]
#[should_panic(expected = "not PSD")]
fn rejects_non_psd_correlation() {
  let bad = [0.99, 0.99, 0.99, 0.99, 0.99, -0.99];
  let _ = Heston2D::<f64, _>::new(
    [Some(0.0), Some(0.0)],
    [Some(0.4), Some(0.4)],
    [0.0, 0.0],
    [0.4, 0.4],
    [2.0, 2.0],
    [1.0, 1.0],
    bad,
    16,
    Some(1.0),
    Some(false),
    Deterministic::new(3),
  );
}

#[test]
#[should_panic(expected = "correlation matrix not PSD at pivot 2")]
fn rejects_inconsistent_zero_cholesky_pivot() {
  let inputs = ValidInputs {
    rho: [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
fn accepts_consistent_zero_cholesky_pivot() {
  let inputs = ValidInputs {
    rho: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ..ValidInputs::default()
  };
  let model = inputs.build();
  let [x1, v1, x2, v2] = model.sample();
  assert!(x1.iter().chain(v1.iter()).all(|value| value.is_finite()));
  assert!(x2.iter().chain(v2.iter()).all(|value| value.is_finite()));
}

#[test]
#[should_panic(expected = "t must be positive")]
fn rejects_negative_time_horizon() {
  let inputs = ValidInputs {
    t: Some(-1.0),
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "t must be positive")]
fn rejects_zero_time_horizon() {
  let inputs = ValidInputs {
    t: Some(0.0),
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "t must be finite")]
fn rejects_non_finite_time_horizon() {
  let inputs = ValidInputs {
    t: Some(f64::NAN),
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "x0[0] must be finite")]
fn rejects_non_finite_initial_log_price() {
  let inputs = ValidInputs {
    x0: [Some(f64::INFINITY), Some(0.0)],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "v0[1] must be finite")]
fn rejects_non_finite_initial_variance() {
  let inputs = ValidInputs {
    v0: [Some(0.4), Some(f64::NAN)],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "mu[1] must be finite")]
fn rejects_non_finite_drift() {
  let inputs = ValidInputs {
    mu: [0.0, f64::NEG_INFINITY],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "theta[0] must be finite")]
fn rejects_non_finite_long_run_variance() {
  let inputs = ValidInputs {
    theta: [f64::NAN, 0.4],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "kappa[1] must be finite")]
fn rejects_non_finite_mean_reversion() {
  let inputs = ValidInputs {
    kappa: [2.0, f64::INFINITY],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "sigma[0] must be finite")]
fn rejects_non_finite_volatility_of_volatility() {
  let inputs = ValidInputs {
    sigma: [f64::NAN, 1.0],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

#[test]
#[should_panic(expected = "rho[4] must be finite")]
fn rejects_non_finite_correlation() {
  let inputs = ValidInputs {
    rho: [0.5, -0.5, 0.0, 0.0, f64::NAN, 0.5],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}

/// Reference: FSDA `Heston2D.m` rejects `2*kappa*theta - sigma^2 < 0`.
#[test]
#[should_panic(expected = "does not satisfy the Feller condition")]
fn rejects_matlab_feller_violation() {
  let _ = Heston2D::<f64, _>::new(
    [Some(0.0), Some(0.0)],
    [Some(0.4), Some(0.4)],
    [0.0, 0.0],
    [0.04, 0.4],
    [1.0, 2.0],
    [1.0, 1.0],
    rho_default(),
    16,
    Some(1.0),
    Some(false),
    Deterministic::new(5),
  );
}

/// Reference: FSDA `Heston2D.m` requires both initial variances to be positive.
#[test]
#[should_panic(expected = "v0[0] must be positive")]
fn rejects_zero_initial_variance_like_matlab() {
  let _ = Heston2D::<f64, _>::new(
    [Some(0.0), Some(0.0)],
    [Some(0.0), Some(0.4)],
    [0.0, 0.0],
    [0.4, 0.4],
    [2.0, 2.0],
    [1.0, 1.0],
    rho_default(),
    16,
    Some(1.0),
    Some(false),
    Deterministic::new(11),
  );
}

#[test]
#[should_panic(expected = "both initial variances v0 must be specified")]
fn rejects_missing_initial_variance_like_matlab() {
  let inputs = ValidInputs {
    v0: [None, Some(0.4)],
    ..ValidInputs::default()
  };
  let _ = inputs.build();
}
