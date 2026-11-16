use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::pareto::SimdPareto;

use super::*;
use crate::StatsError;

/// Integer LCG uniforms, bit-identical to the Python generator behind the
/// `scipy` reference numbers below.
fn lcg(seed: u64, n: usize) -> Vec<f64> {
  let mut x = seed;
  (0..n)
    .map(|_| {
      x = (1_103_515_245 * x + 12_345) % 2_147_483_648;
      x as f64 / 2_147_483_648.0
    })
    .collect()
}

/// GPD(σ = 2, ξ = 0.25) excesses by inverse CDF; reference MLE from
/// `scipy.optimize.minimize(..., method="Nelder-Mead", xatol=fatol=1e-12)`
/// on the same negative log-likelihood (and `genpareto.fit(y, floc=0)`
/// agreeing to 2e-5).
fn gpd_sample() -> Array1<f64> {
  lcg(3, 2_000)
    .into_iter()
    .map(|u| 2.0 * ((1.0 - u).powf(-0.25) - 1.0) / 0.25)
    .collect()
}

/// GEV(μ = 1, σ = 0.5, ξ = 0.1) maxima by inverse CDF; reference MLE as
/// above (and `genextreme.fit` agreeing to 2e-5).
fn gev_sample() -> Array1<f64> {
  lcg(5, 500)
    .into_iter()
    .map(|u| 1.0 - 0.5 / 0.1 * (1.0 - (-u.ln()).powf(-0.1)))
    .collect()
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
  assert!((got - want).abs() < tol, "{what}: {got} vs {want}");
}

#[test]
fn samples_are_bit_identical_to_the_python_generator() {
  let y = gpd_sample();
  assert_close(y[0], 1.722_513_070_454_791_5, 1e-12, "y[0]");
  assert_close(y[1], 0.504_286_318_702_410_7, 1e-12, "y[1]");
  assert_close(y[1_999], 2.333_608_896_524_751_2, 1e-12, "y[1999]");
  let z = gev_sample();
  assert_close(z[0], 1.295_366_923_979_615_3, 1e-12, "z[0]");
  assert_close(z[499], 2.456_398_183_053_753, 1e-12, "z[499]");
}

#[test]
fn gpd_fit_matches_scipy() {
  let fit = fit_gpd(gpd_sample().view()).unwrap();
  assert!(fit.converged);
  assert_eq!(fit.nobs, 2_000);
  assert_close(fit.sigma, 1.944_730_724_656_302_8, 1e-4, "sigma");
  assert_close(fit.xi, 0.275_986_903_122_822_65, 1e-4, "xi");
  let loglik = -3_882.220_859_462_546;
  assert!(
    fit.log_likelihood >= loglik - 1e-7,
    "{}",
    fit.log_likelihood
  );
  assert_close(fit.log_likelihood, loglik, 1e-6, "log-likelihood");
  assert_close(fit.aic, 4.0 - 2.0 * fit.log_likelihood, 1e-9, "aic");
  // Smith (1987) asymptotics: Var(ξ̂) ≈ (1+ξ)²/n, Var(σ̂) ≈ 2σ²(1+ξ)/n.
  let n = 2_000_f64;
  let se_xi = (1.0 + fit.xi) / n.sqrt();
  let se_sigma = fit.sigma * (2.0 * (1.0 + fit.xi)).sqrt() / n.sqrt();
  assert!(
    ((fit.std_errors[1] - se_xi) / se_xi).abs() < 0.15,
    "se(xi) = {}",
    fit.std_errors[1]
  );
  assert!(
    ((fit.std_errors[0] - se_sigma) / se_sigma).abs() < 0.15,
    "se(sigma) = {}",
    fit.std_errors[0]
  );
  assert_close(
    fit.covariance[[0, 1]],
    fit.covariance[[1, 0]],
    1e-12,
    "symmetry",
  );
}

/// With the threshold at zero every observation exceeds it, so the POT
/// quantiles reduce to the GPD's own: VaR₉₉ and ES₉₉ from eq. 5.18 / 5.20
/// at the reference optimum.
#[test]
fn pot_var_and_es_match_the_qrm_formulas() {
  let pot = fit_pot(gpd_sample().view(), 0.0).unwrap();
  assert_eq!((pot.n_exceedances, pot.nobs), (2_000, 2_000));
  assert_close(pot.exceedance_rate, 1.0, 1e-15, "rate");
  let var = pot.quantile(0.99);
  let es = pot.expected_shortfall(0.99);
  assert!(
    ((var - 18.069_212_156_928_163) / 18.069_212_156_928_163).abs() < 1e-3,
    "VaR {var}"
  );
  assert!(
    ((es - 27.643_067_463_708_682) / 27.643_067_463_708_682).abs() < 1e-3,
    "ES {es}"
  );
  assert!(es > var);
  // A higher threshold keeps only the tail and re-anchors the rate.
  let high = fit_pot(gpd_sample().view(), 5.0).unwrap();
  assert!(high.n_exceedances < 2_000 && high.n_exceedances >= 10);
  assert_close(
    high.exceedance_rate,
    high.n_exceedances as f64 / 2_000.0,
    1e-15,
    "rate",
  );
  assert!(high.quantile(0.999) > high.threshold);
}

#[test]
fn gev_fit_matches_scipy() {
  let fit = fit_gev(gev_sample().view()).unwrap();
  assert!(fit.converged);
  assert_eq!(fit.nobs, 500);
  assert_close(fit.mu, 1.051_093_874_083_600_7, 1e-4, "mu");
  assert_close(fit.sigma, 0.545_236_613_390_745_7, 1e-4, "sigma");
  assert_close(fit.xi, 0.089_586_741_117_853_5, 1e-4, "xi");
  let loglik = -510.747_071_702_915_9;
  assert!(
    fit.log_likelihood >= loglik - 1e-7,
    "{}",
    fit.log_likelihood
  );
  assert_close(fit.log_likelihood, loglik, 1e-6, "log-likelihood");
  let z100 = fit.return_level(100.0);
  assert!(
    ((z100 - 4.155_045_581_306_86) / 4.155_045_581_306_86).abs() < 1e-3,
    "z_100 = {z100}"
  );
  assert!(fit.std_errors.iter().all(|s| s.is_finite() && *s > 0.0));
  assert_close(
    fit.bic,
    3.0 * 500_f64.ln() - 2.0 * fit.log_likelihood,
    1e-9,
    "bic",
  );
}

/// Hill by hand on ten values with k = 4: the four largest are 16, 13,
/// 11, 8 and the threshold is the fifth, 7.
#[test]
fn hill_matches_the_hand_computation() {
  let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0, 3.0, 5.0, 7.0, 11.0, 13.0];
  let h = fit_hill(x.view(), 4).unwrap();
  assert_close(h.xi, 0.507_808_574_489_568, 1e-12, "xi");
  assert_close(h.alpha, 1.0 / 0.507_808_574_489_568, 1e-12, "alpha");
  assert_close(h.std_error, 0.507_808_574_489_568 / 2.0, 1e-12, "se");
  assert_eq!((h.k, h.threshold, h.nobs), (4, 7.0, 10));
}

/// On an exact Pareto(α = 3) tail the Hill estimate is unbiased for
/// ξ = 1/3; best of three pinned seeds within 3 standard errors.
#[test]
fn hill_recovers_a_pareto_tail_index() {
  let closest = [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let mut dist = SimdPareto::<f64>::new(1.0, 3.0).seeded(&Deterministic::new(seed));
      let mut xs = vec![0.0; 20_000];
      dist.fill_slice(&mut xs);
      let h = fit_hill(Array1::from(xs).view(), 500).unwrap();
      (h.xi - 1.0 / 3.0).abs() / h.std_error
    })
    .fold(f64::INFINITY, f64::min);
  assert!(
    closest < 3.0,
    "every seed missed 1/3 by {closest} standard errors"
  );
}

#[test]
fn block_maxima_drop_the_partial_block() {
  let x = array![1.0_f64, 5.0, 2.0, 9.0, 3.0, 4.0, 7.0];
  assert_eq!(block_maxima(x.view(), 3).to_vec(), vec![5.0, 9.0]);
}

#[test]
fn mean_excess_is_the_mean_residual_life() {
  let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
  let e = mean_excess(x.view(), array![0.0, 2.5, 10.0].view());
  assert_close(e[0], 3.0, 1e-15, "e(0)");
  assert_close(e[1], 1.5, 1e-15, "e(2.5)");
  assert!(e[2].is_nan());
}

/// A NaN loss lies on no side of a threshold, so no mean excess is known.
#[test]
fn a_nan_observation_makes_every_mean_excess_nan() {
  let x = array![1.0_f64, f64::NAN, 3.0, 4.0];
  let e = mean_excess(x.view(), array![0.0, 2.0, 10.0].view());
  assert!(e.iter().all(|v| v.is_nan()), "{e:?}");
}

#[test]
#[should_panic(expected = "k must satisfy `k >= 1`, got k = 0")]
fn hill_rejects_zero_k() {
  let _ = fit_hill(array![1.0_f64, 2.0].view(), 0);
}

#[test]
fn hill_rejects_too_large_k() {
  assert_eq!(
    fit_hill(array![1.0_f64, 2.0, -1.0, 3.0].view(), 3).unwrap_err(),
    StatsError::InsufficientData { needed: 4, got: 3 }
  );
}

/// A NaN fails `x > 0`, so the positivity filter alone would drop it silently.
#[test]
fn hill_validates_before_its_positivity_filter() {
  assert!(matches!(
    fit_hill(array![1.0_f64, f64::NAN, -1.0, 3.0].view(), 1),
    Err(StatsError::InvalidObservation {
      index: 1,
      constraint: "finite",
      ..
    })
  ));
}

#[test]
fn gpd_fit_rejects_negative_excesses() {
  let mut y = gpd_sample();
  y[0] = -1.0;
  assert_eq!(
    fit_gpd(y.view()).unwrap_err(),
    StatsError::InvalidObservation {
      index: 0,
      value: -1.0,
      constraint: "finite and >= 0"
    }
  );
}

#[test]
fn gev_fit_rejects_a_short_series() {
  assert_eq!(
    fit_gev(array![1.0_f64, 2.0, 3.0].view()).unwrap_err(),
    StatsError::InsufficientData { needed: 10, got: 3 }
  );
}

#[test]
fn pot_fit_rejects_a_threshold_too_high() {
  assert_eq!(
    fit_pot(gpd_sample().view(), 1_000.0).unwrap_err(),
    StatsError::InsufficientData { needed: 10, got: 0 }
  );
}

#[test]
#[should_panic(expected = "block_size must satisfy `block_size >= 1`, got block_size = 0")]
fn block_maxima_rejects_zero_block() {
  let _ = block_maxima(array![1.0_f64].view(), 0);
}

#[test]
fn a_return_period_of_at_most_one_block_gives_nan() {
  let fit = fit_gev(gev_sample().view()).unwrap();
  for period in [1.0, 0.5, -2.0, f64::NAN] {
    assert!(fit.return_level(period).is_nan(), "period {period}");
  }
}

/// The level check comes first, so the `ξ ≥ 1` branch answers `+∞` only inside `(0, 1)`.
#[test]
fn a_level_outside_the_unit_interval_gives_nan() {
  let pot = fit_pot(gpd_sample().view(), 0.0).unwrap();
  let heavy = PotFit {
    gpd: GpdFit {
      xi: 1.5,
      ..pot.gpd.clone()
    },
    ..pot.clone()
  };
  for p in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
    assert!(pot.quantile(p).is_nan(), "quantile at {p}");
    assert!(pot.expected_shortfall(p).is_nan(), "ES at {p}");
    assert!(
      heavy.expected_shortfall(p).is_nan(),
      "ES at {p} for xi >= 1"
    );
  }
  assert_eq!(heavy.expected_shortfall(0.99), f64::INFINITY);
}

/// Hosking–Wallis PWM on the reference GPD sample: the closed-form
/// estimates reproduce the Python evaluation of eqs. 15-16 to rounding,
/// and land near the σ = 2, ξ = 0.25 truth like the MLE does.
#[test]
fn gpd_pwm_matches_the_hosking_wallis_formulas() {
  let est = fit_gpd_pwm(gpd_sample().view()).unwrap();
  assert_eq!(est.nobs, 2_000);
  assert_close(est.a0, 2.673_026_587_667_958, 1e-12, "a0");
  assert_close(est.a1, 0.561_686_053_844_953, 1e-12, "a1");
  assert_close(est.xi, 0.275_082_205_611_526_74, 1e-12, "xi");
  assert_close(est.sigma, 1.937_724_538_274_003, 1e-12, "sigma");
  let mle = fit_gpd(gpd_sample().view()).unwrap();
  assert!((est.xi - mle.xi).abs() < 0.01 && (est.sigma - mle.sigma).abs() < 0.02);
}

#[test]
fn gpd_pwm_rejects_a_constant_sample() {
  for c in [0.0, 0.3] {
    assert_eq!(
      fit_gpd_pwm(Array1::<f64>::from_elem(400, c).view()).unwrap_err(),
      StatsError::DegenerateData("constant excesses"),
      "{c}"
    );
  }
}

#[test]
fn gpd_pwm_rejects_a_single_excess() {
  assert_eq!(
    fit_gpd_pwm(array![1.0_f64].view()).unwrap_err(),
    StatsError::InsufficientData { needed: 2, got: 1 }
  );
}

/// A NaN observation is a NaN block maximum; `f64::max` would hide it behind its neighbours.
#[test]
fn a_nan_observation_makes_its_block_maximum_nan() {
  let maxima = block_maxima(array![1.0_f64, 2.0, f64::NAN, 0.5, 3.0, 1.0].view(), 3);
  assert!(maxima[0].is_nan());
  assert_eq!(maxima[1], 3.0);
}
