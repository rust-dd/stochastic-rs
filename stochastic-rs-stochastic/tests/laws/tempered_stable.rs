//! The tempered stable series at α > 1, where the terms past the `J`-th carry real mean and
//! variance: each law against its Lévy measure's closed forms, at a short and a long series.
//!
//! Reference: Rosiński (2007), Thm 5.4, SPA 117(6), 677–707, DOI 10.1016/j.spa.2006.10.003
//! Reference: Asmussen & Rosiński (2001), J. Appl. Probab. 38(2), 482–493, DOI 10.1239/jap/996986757

use ndarray::Array1;
use num_complex::Complex64;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::kobol::Kobol;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::characteristic_function_holds;
use super::common::holds_on_one_of;
use super::common::over_paths;
use super::common::spread_over_paths;
use super::common::terminals;

/// Paths per seed: the standard error of a unit variance is near 0.025.
const PATHS: usize = 4_096;

const SEEDS: [u64; 3] = [2718, 999, 42];

/// A short and a long series: the law may not depend on where the series stops.
const TERMS: [usize; 2] = [64, 512];

/// The mean and variance of `X_t` on one of the seeds, each within five standard errors.
fn mean_and_variance_hold(
  sample: impl Fn(u64) -> Vec<Array1<f64>>,
  mean: f64,
  variance: f64,
  at: &str,
) {
  let x = SEEDS.map(|seed| terminals(&sample(seed)));
  let means = x.each_ref().map(|x| over_paths(x));
  holds_on_one_of(&means, mean, &format!("{at}: mean"));
  let variances = x.each_ref().map(|x| spread_over_paths(x));
  holds_on_one_of(&variances, variance, &format!("{at}: variance"));
}

/// The standard CTS: mean 0 by `b_T`, variance `t` by the scale `C` (Kim et al. 2010).
#[test]
fn classical_tempered_stable_keeps_its_unit_variance_above_one() {
  for (alpha, t) in [(1.5, 1.0), (1.8, 2.0)] {
    for j in TERMS {
      mean_and_variance_hold(
        |seed| {
          Cts::<f64, _>::new(
            2.0,
            6.0,
            alpha,
            2,
            j,
            None,
            Some(t),
            Deterministic::new(seed),
          )
          .sample_par(PATHS)
        },
        0.0,
        t,
        &format!("CTS at α = {alpha}, T = {t}, J = {j}"),
      );
    }
  }
}

/// The standard RDTS, as [`classical_tempered_stable_keeps_its_unit_variance_above_one`] with
/// the Gaussian tempering.
#[test]
fn rapidly_decreasing_tempered_stable_keeps_its_unit_variance_above_one() {
  for alpha in [1.5, 1.8] {
    for j in TERMS {
      mean_and_variance_hold(
        |seed| {
          Rdts::<f64, _>::new(2.0, 6.0, alpha, 2, j, None, None, Deterministic::new(seed))
            .sample_par(PATHS)
        },
        0.0,
        1.0,
        &format!("RDTS at α = {alpha}, J = {j}"),
      );
    }
  }
}

/// CGMY above `Y = 1` carries no drift, so its mean is the exponent's own,
/// `C Γ(1 − Y)(λ₊^{Y−1} − λ₋^{Y−1}) t`; variance `C Γ(2 − Y)(λ₊^{Y−2} + λ₋^{Y−2}) t`.
#[test]
fn cgmy_has_the_moments_of_its_exponent_above_one() {
  let (c, plus, minus, t) = (0.5_f64, 4.0_f64, 7.0_f64, 1.5_f64);
  for y in [1.5_f64, 1.0] {
    let mean = if y == 1.0 {
      -c * (plus.ln() - minus.ln()) * t
    } else {
      c * gamma(1.0 - y) * (plus.powf(y - 1.0) - minus.powf(y - 1.0)) * t
    };
    let variance = c * gamma(2.0 - y) * (plus.powf(y - 2.0) + minus.powf(y - 2.0)) * t;
    for j in TERMS {
      mean_and_variance_hold(
        |seed| {
          Cgmy::<f64, _>::new(
            c,
            plus,
            minus,
            y,
            2,
            j,
            None,
            Some(t),
            Deterministic::new(seed),
          )
          .sample_par(PATHS)
        },
        mean,
        variance,
        &format!("CGMY at Y = {y}, J = {j}"),
      );
    }
  }
}

/// KoBoL with unequal sides: above `α = 1` the mean is `Γ(1 − α) D (p λ₊^{α−1} − q λ₋^{α−1}) t`,
/// which the uncentred series reaches only with its remainder; below, the drift makes it zero.
#[test]
fn kobol_with_unequal_sides_has_the_moments_of_its_exponent() {
  let (d, plus, minus) = (0.4_f64, 3.0_f64, 5.0_f64);
  for (alpha, p, q) in [
    (1.5_f64, 2.0_f64, 1.0_f64),
    (1.2, 0.5, 1.5),
    (0.5, 2.0, 1.0),
  ] {
    let mean = if alpha > 1.0 {
      gamma(1.0 - alpha) * d * (p * plus.powf(alpha - 1.0) - q * minus.powf(alpha - 1.0))
    } else {
      0.0
    };
    let variance =
      d * gamma(2.0 - alpha) * (p * plus.powf(alpha - 2.0) + q * minus.powf(alpha - 2.0));
    for j in TERMS {
      mean_and_variance_hold(
        |seed| {
          Kobol::<f64, _>::new(
            d,
            p,
            q,
            plus,
            minus,
            alpha,
            2,
            j,
            None,
            None,
            Deterministic::new(seed),
          )
          .sample_par(PATHS)
        },
        mean,
        variance,
        &format!("KoBoL at α = {alpha}, p = {p}, q = {q}, J = {j}"),
      );
    }
  }
}

/// The whole CTS law at `α = 1.5`: `ln φ(u) = Σ_± C Γ(−α)((λ_± ∓ iu)^α − λ_±^α ± iuα λ_±^{α−1})`.
/// The Gaussian remainder's own error, `u³ ε V / 6`, is under 0.002 at `J = 2048`.
#[test]
fn classical_tempered_stable_matches_its_characteristic_function_above_one() {
  let (plus, minus, alpha) = (2.0_f64, 6.0_f64, 1.5_f64);
  let c = 1.0 / (gamma(2.0 - alpha) * (plus.powf(alpha - 2.0) + minus.powf(alpha - 2.0)));
  let paths = Cts::<f64, _>::new(
    plus,
    minus,
    alpha,
    2,
    2_048,
    None,
    None,
    Deterministic::new(7),
  )
  .sample_par(8_192);
  let sample = terminals(&paths);
  for u in [0.3, 0.7, 1.5, 2.5] {
    let side = |lambda: f64, sign: f64| {
      let iu = Complex64::new(0.0, sign * u);
      (Complex64::from(lambda) - iu).powf(alpha) - lambda.powf(alpha)
        + iu * alpha * lambda.powf(alpha - 1.0)
    };
    let exponent = (side(plus, 1.0) + side(minus, -1.0)) * c * gamma(-alpha);
    characteristic_function_holds(&sample, u, exponent.exp(), "CTS at α = 1.5");
  }
}
