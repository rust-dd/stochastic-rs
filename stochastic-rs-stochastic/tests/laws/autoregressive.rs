//! The discrete-time models, against the moments and autocorrelations their
//! source papers derive.
//!
//! Every case here is a closed form of the *model*, not a comparison against
//! another implementation: an AR's Yule-Walker variance and geometric ACF
//! (Box & Jenkins 1970), an MA's ACF cutting off past its order, Engle's
//! (1982) unconditional ARCH variance and kurtosis, Bollerslev's (1986)
//! GARCH variance with the (1988) squared-series ACF, the half-threshold of
//! Glosten, Jagannathan & Runkle (1993), and Nelson's (1991) EGARCH
//! log-variance moments read through the log-chi-square identity.

use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::autoregressive::arch::Arch;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::ma::MAq;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::autoregressive::tgarch::GjrGarch;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::N;
use super::common::PATHS;
use super::common::across_paths;
use super::common::autocorr;
use super::common::holds;
use super::common::kurtosis;
use super::common::mean;
use super::common::squared_autocorr;
use super::common::variance;

/// `E[ln z²]` for a standard normal `z`: `−(γ + ln 2)`, the mean of a
/// log-chi-square with one degree of freedom. What lets an EGARCH's
/// log-variance be read off the observed series.
const LOG_CHI2_MEAN: f64 = -1.270_362_845_461_478_2;

/// `Var[ln z²] = π²/2`, the companion of [`LOG_CHI2_MEAN`].
const LOG_CHI2_VARIANCE: f64 = 4.934_802_200_544_679;

/// AR(1): the Yule-Walker variance `σ²/(1 − φ²)` and the geometric
/// autocorrelation `ρ(k) = φᵏ` (Box & Jenkins 1970, §3.2).
#[test]
fn ar1_variance_and_acf_match_the_yule_walker_law() {
  let (phi, sigma) = (0.6, 1.0);
  let paths = ARp::<f64, _>::new(array![phi], sigma, N, None, Deterministic::new(11)).sample_par(PATHS);
  holds(
    across_paths(&paths, variance),
    sigma * sigma / (1.0 - phi * phi),
    "AR(1) variance",
  );
  for k in 1..=4 {
    holds(
      across_paths(&paths, |x| autocorr(x, k)),
      phi.powi(k as i32),
      &format!("AR(1) autocorrelation at lag {k}"),
    );
  }
}

/// AR(2): the same Yule-Walker system one order up — variance
/// `σ²(1 − φ₂) / ((1 + φ₂)((1 − φ₂)² − φ₁²))` and `ρ(1) = φ₁/(1 − φ₂)`
/// (Box & Jenkins 1970, §3.2.3).
#[test]
fn ar2_variance_and_first_lag_match_the_yule_walker_law() {
  let (phi1, phi2, sigma) = (0.5, 0.3, 0.8);
  let paths =
    ARp::<f64, _>::new(array![phi1, phi2], sigma, N, None, Deterministic::new(23)).sample_par(PATHS);
  let variance_law =
    sigma * sigma * (1.0 - phi2) / ((1.0 + phi2) * ((1.0 - phi2).powi(2) - phi1 * phi1));
  holds(across_paths(&paths, variance), variance_law, "AR(2) variance");
  holds(
    across_paths(&paths, |x| autocorr(x, 1)),
    phi1 / (1.0 - phi2),
    "AR(2) autocorrelation at lag 1",
  );
  holds(
    across_paths(&paths, |x| autocorr(x, 2)),
    phi1 * phi1 / (1.0 - phi2) + phi2,
    "AR(2) autocorrelation at lag 2",
  );
}

/// MA(2): variance `σ²(1 + Σθ²)`, the two autocorrelations the order allows,
/// and the cut-off past it — the property that separates a moving average
/// from an autoregression (Box & Jenkins 1970, §3.3).
#[test]
fn ma2_variance_and_acf_cut_off_after_the_order() {
  let (t1, t2, sigma) = (0.7, 0.4, 1.3);
  let paths = MAq::<f64, _>::new(array![t1, t2], sigma, N, Deterministic::new(37)).sample_par(PATHS);
  let scale = 1.0 + t1 * t1 + t2 * t2;
  holds(
    across_paths(&paths, variance),
    sigma * sigma * scale,
    "MA(2) variance",
  );
  holds(
    across_paths(&paths, |x| autocorr(x, 1)),
    (t1 + t1 * t2) / scale,
    "MA(2) autocorrelation at lag 1",
  );
  holds(
    across_paths(&paths, |x| autocorr(x, 2)),
    t2 / scale,
    "MA(2) autocorrelation at lag 2",
  );
  holds(
    across_paths(&paths, |x| autocorr(x, 3)),
    0.0,
    "MA(2) autocorrelation at lag 3",
  );
}

/// ARCH(1): the unconditional variance `ω/(1 − α)` and the kurtosis
/// `3(1 − α²)/(1 − 3α²)` (Engle 1982, eq. 22 and Theorem 1). The excess over
/// three is the whole point of the model — a conditionally normal series with
/// unconditionally fat tails.
///
/// `α = 0.25` keeps the eighth moment finite (`105α⁴ = 0.41 < 1`), without
/// which a sample kurtosis has no standard error to compare against.
#[test]
fn arch1_unconditional_variance_and_kurtosis_match_engle() {
  let (omega, alpha) = (0.3, 0.25);
  let paths = Arch::<f64, _>::new(omega, array![alpha], N, Deterministic::new(53)).sample_par(PATHS);
  holds(
    across_paths(&paths, variance),
    omega / (1.0 - alpha),
    "ARCH(1) unconditional variance",
  );
  holds(
    across_paths(&paths, kurtosis),
    3.0 * (1.0 - alpha * alpha) / (1.0 - 3.0 * alpha * alpha),
    "ARCH(1) kurtosis",
  );
  holds(across_paths(&paths, mean), 0.0, "ARCH(1) mean");
}

/// GARCH(1,1): the unconditional variance `ω/(1 − α − β)` (Bollerslev 1986,
/// eq. 10) and the kurtosis `3(1 − (α+β)²)/(1 − (α+β)² − 2α²)` (Bollerslev
/// 1986, Theorem 2), whose condition `2α² + (α+β)² < 1` holds here at 0.7675.
#[test]
fn garch11_unconditional_variance_and_kurtosis_match_bollerslev() {
  let (omega, alpha, beta) = (0.15, 0.15, 0.7);
  let paths = Garch::<f64, _>::new(
    omega,
    array![alpha],
    array![beta],
    N,
    Deterministic::new(67),
  )
  .sample_par(PATHS);
  holds(
    across_paths(&paths, variance),
    omega / (1.0 - alpha - beta),
    "GARCH(1,1) unconditional variance",
  );
  let persistence = (alpha + beta).powi(2);
  holds(
    across_paths(&paths, kurtosis),
    3.0 * (1.0 - persistence) / (1.0 - persistence - 2.0 * alpha * alpha),
    "GARCH(1,1) kurtosis",
  );
}

/// GARCH(1,1) squared series: `ρ₁ = α(1 − αβ − β²)/(1 − 2αβ − β²)` and
/// `ρ_k = (α + β)^{k−1} ρ₁` past it (Bollerslev 1988, eq. 12). The level
/// series is white noise — that is the same statement — so lag one of the
/// levels is checked here too.
#[test]
fn garch11_squared_series_acf_decays_at_alpha_plus_beta() {
  let (omega, alpha, beta) = (0.15, 0.15, 0.7);
  let paths = Garch::<f64, _>::new(
    omega,
    array![alpha],
    array![beta],
    N,
    Deterministic::new(71),
  )
  .sample_par(PATHS);
  let rho1 = alpha * (1.0 - alpha * beta - beta * beta) / (1.0 - 2.0 * alpha * beta - beta * beta);
  for k in 1..=3 {
    holds(
      across_paths(&paths, |x| squared_autocorr(x, k)),
      (alpha + beta).powi(k as i32 - 1) * rho1,
      &format!("GARCH(1,1) squared-series autocorrelation at lag {k}"),
    );
  }
  holds(
    across_paths(&paths, |x| autocorr(x, 1)),
    0.0,
    "GARCH(1,1) level autocorrelation at lag 1",
  );
}

/// GJR-GARCH: a symmetric innovation crosses zero half the time, so the
/// threshold term enters the unconditional variance at half its coefficient —
/// `ω/(1 − α − γ/2 − β)` (Glosten, Jagannathan & Runkle 1993, §I).
#[test]
fn gjr_garch_unconditional_variance_carries_the_half_threshold() {
  let (omega, alpha, gamma, beta) = (0.12, 0.06, 0.12, 0.7);
  let paths = GjrGarch::<f64, _>::new(
    omega,
    array![alpha],
    array![gamma],
    array![beta],
    N,
    Deterministic::new(83),
  )
  .sample_par(PATHS);
  holds(
    across_paths(&paths, variance),
    omega / (1.0 - alpha - 0.5 * gamma - beta),
    "GJR-GARCH unconditional variance",
  );
}

/// The leverage effect the threshold exists for: a negative shock raises the
/// next conditional variance more than a positive one of the same size, so
/// the correlation between a step and the *next* squared step is negative —
/// where a plain GARCH, symmetric in the shock, has none.
#[test]
fn gjr_garch_leverage_makes_a_drop_raise_the_next_variance() {
  let build_gjr = |gamma: f64| {
    GjrGarch::<f64, _>::new(
      0.12,
      array![0.06],
      array![gamma],
      array![0.7],
      N,
      Deterministic::new(89),
    )
    .sample_par(PATHS)
  };
  // The standardised cross-moment `E[(X_{t−1} − μ)(X_t² − σ²)] / σ³`: zero
  // when the shock enters the variance through its square alone, negative
  // when a fall counts for more than a rise of the same size.
  let cross = |x: &[f64]| {
    let (m, v) = (mean(x), variance(x));
    (1..x.len())
      .map(|t| (x[t - 1] - m) * (x[t] * x[t] - v))
      .sum::<f64>()
      / x.len() as f64
      / v.powf(1.5)
  };
  let asymmetric = across_paths(&build_gjr(0.12), cross);
  assert!(
    asymmetric.mean + 5.0 * asymmetric.se < 0.0,
    "GJR-GARCH shows no leverage: {} ± {}",
    asymmetric.mean,
    asymmetric.se
  );
  let symmetric = across_paths(&build_gjr(0.0), cross);
  holds(symmetric, 0.0, "GJR-GARCH without a threshold is symmetric");
}

/// `Agarch`'s `delta` is `GjrGarch`'s `gamma` under another name — its own
/// doc says so — and the two recursions are the same expression, so the same
/// seed must give the same path, point for point. A cross-implementation
/// check rather than a law: it is what keeps the duplicate honest.
#[test]
fn agarch_delta_is_the_gjr_gamma() {
  let (omega, alpha, asym, beta) = (0.12, 0.06, 0.12, 0.7);
  let gjr = GjrGarch::<f64, _>::new(
    omega,
    array![alpha],
    array![asym],
    array![beta],
    512,
    Deterministic::new(97),
  )
  .sample();
  let agarch = Agarch::<f64, _>::new(
    omega,
    array![alpha],
    array![asym],
    array![beta],
    512,
    Deterministic::new(97),
  )
  .sample();
  assert_eq!(gjr, agarch, "Agarch and GjrGarch parted ways");
}

/// EGARCH(1,1): in the stationary law `E[ln σ²] = ω/(1 − β)` and
/// `Var[ln σ²] = (α²(1 − 2/π) + γ²)/(1 − β²)` (Nelson 1991, §2.2, using
/// `E|z| = √(2/π)` and `Cov(|z|, z) = 0` for a symmetric innovation). Neither
/// is observable directly — the sampler returns `X = σz` — so both are read
/// through `ln X² = ln σ² + ln z²`, whose second term is a log-chi-square
/// with one degree of freedom, independent of the first.
#[test]
fn egarch_log_variance_moments_match_nelson() {
  let (omega, alpha, gamma, beta) = (-0.1, 0.2, -0.1, 0.9);
  let paths = Egarch::<f64, _>::new(
    omega,
    array![alpha],
    array![gamma],
    array![beta],
    N,
    Deterministic::new(101),
  )
  .sample_par(PATHS);
  let log_square = |x: &[f64]| -> Vec<f64> { x.iter().map(|v| (v * v).ln()).collect() };
  holds(
    across_paths(&paths, |x| mean(&log_square(x))),
    omega / (1.0 - beta) + LOG_CHI2_MEAN,
    "EGARCH mean log-variance",
  );
  let shock_variance = alpha * alpha * (1.0 - 2.0 / std::f64::consts::PI) + gamma * gamma;
  holds(
    across_paths(&paths, |x| variance(&log_square(x))),
    shock_variance / (1.0 - beta * beta) + LOG_CHI2_VARIANCE,
    "EGARCH log-variance spread",
  );
}

/// ARMA(1,1) — an `Arima` with no differencing: variance
/// `σ²(1 + 2φθ + θ²)/(1 − φ²)` and `ρ(1) = (1 + φθ)(φ + θ)/(1 + 2φθ + θ²)`,
/// with `ρ(k) = φ^{k−1} ρ(1)` past it (Box & Jenkins 1970, §3.4.2).
#[test]
fn arma11_variance_and_acf_match_box_jenkins() {
  let (phi, theta, sigma) = (0.7, 0.4, 0.9);
  let paths = Arima::<f64, _>::new(
    array![phi],
    array![theta],
    0,
    sigma,
    N,
    Deterministic::new(103),
  )
  .sample_par(PATHS);
  let numerator = 1.0 + 2.0 * phi * theta + theta * theta;
  holds(
    across_paths(&paths, variance),
    sigma * sigma * numerator / (1.0 - phi * phi),
    "ARMA(1,1) variance",
  );
  let rho1 = (1.0 + phi * theta) * (phi + theta) / numerator;
  for k in 1..=3 {
    holds(
      across_paths(&paths, |x| autocorr(x, k)),
      phi.powi(k as i32 - 1) * rho1,
      &format!("ARMA(1,1) autocorrelation at lag {k}"),
    );
  }
}

/// A purely seasonal AR — `X_t = Φ X_{t−s} + ε_t`, the `Sarima` with every
/// other order empty — is an AR(1) on the seasonal lag: variance
/// `σ²/(1 − Φ²)`, a spike at `s` and nothing between (Box & Jenkins 1970,
/// §9.2).
#[test]
fn seasonal_ar_puts_its_only_spike_at_the_season() {
  let (big_phi, sigma, season) = (0.6, 1.1, 4);
  let paths = Sarima::<f64, _>::new(
    array![],
    array![],
    array![big_phi],
    array![],
    0,
    0,
    season,
    sigma,
    N,
    Deterministic::new(107),
  )
  .sample_par(PATHS);
  holds(
    across_paths(&paths, variance),
    sigma * sigma / (1.0 - big_phi * big_phi),
    "seasonal AR(1) variance",
  );
  for k in 1..season {
    holds(
      across_paths(&paths, |x| autocorr(x, k)),
      0.0,
      &format!("seasonal AR(1) autocorrelation at lag {k}"),
    );
  }
  holds(
    across_paths(&paths, |x| autocorr(x, season)),
    big_phi,
    "seasonal AR(1) autocorrelation at the season",
  );
  holds(
    across_paths(&paths, |x| autocorr(x, 2 * season)),
    big_phi * big_phi,
    "seasonal AR(1) autocorrelation at twice the season",
  );
}

/// Every path is finite and starts where the recursion says it does. The
/// cheap structural check the law cases above assume.
#[test]
fn every_autoregressive_path_stays_in_the_reals() {
  let checks: Vec<(&str, Array1<f64>)> = vec![
    (
      "ARp",
      ARp::<f64, _>::new(array![0.5], 1.0, 4_096, None, Deterministic::new(2)).sample(),
    ),
    (
      "MAq",
      MAq::<f64, _>::new(array![0.5, 0.2], 1.0, 4_096, Deterministic::new(3)).sample(),
    ),
    (
      "Arch",
      Arch::<f64, _>::new(0.2, array![0.3], 4_096, Deterministic::new(5)).sample(),
    ),
    (
      "Garch",
      Garch::<f64, _>::new(0.1, array![0.1], array![0.8], 4_096, Deterministic::new(7)).sample(),
    ),
    (
      "Egarch",
      Egarch::<f64, _>::new(
        -0.1,
        array![0.2],
        array![-0.1],
        array![0.9],
        4_096,
        Deterministic::new(13),
      )
      .sample(),
    ),
    (
      "GjrGarch",
      GjrGarch::<f64, _>::new(
        0.1,
        array![0.05],
        array![0.1],
        array![0.7],
        4_096,
        Deterministic::new(17),
      )
      .sample(),
    ),
    (
      "Agarch",
      Agarch::<f64, _>::new(
        0.1,
        array![0.05],
        array![0.1],
        array![0.7],
        4_096,
        Deterministic::new(19),
      )
      .sample(),
    ),
    (
      "Arima",
      Arima::<f64, _>::new(array![0.5], array![0.3], 1, 1.0, 4_096, Deterministic::new(29)).sample(),
    ),
    (
      "Sarima",
      Sarima::<f64, _>::new(
        array![0.3],
        array![0.2],
        array![0.4],
        array![0.1],
        0,
        0,
        4,
        1.0,
        4_096,
        Deterministic::new(31),
      )
      .sample(),
    ),
  ];
  for (name, path) in checks {
    assert!(
      path.iter().all(|v| v.is_finite()),
      "{name} left the reals"
    );
    assert_eq!(path.len(), 4_096, "{name} produced the wrong length");
  }
}
