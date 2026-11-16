//! Phillips-Perron unit-root test: the lag-0 Dickey-Fuller regression corrected for serial
//! correlation by a Newey-West long-run variance, as the `Tau` and `Rho` statistics.
//!
//! References:
//! - Phillips P. C. B. & Perron P. (1988) — *Testing for a Unit Root in
//!   Time Series Regression*, Biometrika 75(2), 335–346,
//!   DOI: 10.1093/biomet/75.2.335; the origin of both statistics.
//! - StataCorp (2025) — *Stata Time-Series Reference Manual*,
//!   `[TS] pperron`, Methods and formulas; the finite-sample `Z_ρ` and
//!   `Z_τ` computed here.
//! - Hamilton J. D. (1994) — *Time Series Analysis*, Princeton University
//!   Press, ch. 17, DOI: 10.2307/j.ctv14jx6sm, which Stata cites for them.
//! - Sheppard K. et al. — *arch* 9.0.0, `arch.unitroot.PhillipsPerron`,
//!   DOI: 10.5281/zenodo.593254; the regression on the series less its mean.
//! - Newey W. K. & West K. D. (1987) — *A Simple, Positive
//!   Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
//!   Covariance Matrix*, Econometrica 55(3), 703–708,
//!   DOI: 10.2307/1913610.

use ndarray::ArrayView1;

use super::common::Trend;
use super::common::adf_critical_values;
use super::common::centred;
use super::common::fit_adf;
use super::common::newey_west_long_run_variance;
use super::common::schwert_max_lags;
use super::common::unit_scaled;
use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate;

/// Phillips-Perron test statistic variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpTestType {
  /// Z-tau (t-ratio style) statistic.
  Tau,
  /// Z-rho statistic.
  Rho,
}

/// Configuration for the Phillips-Perron unit-root test.
#[derive(Debug, Clone, Copy)]
pub struct PhillipsPerronConfig {
  /// Deterministic terms in the test regression.
  pub trend: Trend,
  /// PP statistic type.
  pub test_type: PpTestType,
  /// Newey-West lag length. If `None`, a Schwert-style default is used.
  pub lags: Option<usize>,
  /// Significance level for decision output (used for `Tau`).
  pub alpha: f64,
}

impl Default for PhillipsPerronConfig {
  fn default() -> Self {
    Self {
      trend: Trend::Constant,
      test_type: PpTestType::Tau,
      lags: None,
      alpha: 0.05,
    }
  }
}

/// Phillips–Perron test (null: a unit root); `Rho` has no shipped table, so it carries no decision.
#[derive(Debug, Clone)]
pub struct PhillipsPerronTest {
  /// PP statistic value.
  pub statistic: f64,
  /// No p-value table is shipped; `Tau` is decided against `critical_values`.
  pub p_value: Option<f64>,
  /// The ADF table at the 1 %, 5 % and 10 % levels for `Tau`; `None` for `Rho`.
  pub critical_values: Option<CriticalValues>,
  /// Unit-root rejection at `alpha` for `Tau`; `None` for `Rho`.
  pub rejects_null: Option<bool>,
  /// Newey-West lag length used.
  pub used_lags: usize,
  /// Variant of the PP statistic.
  pub test_type: PpTestType,
}

impl crate::traits::HypothesisTest for PhillipsPerronTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    self.rejects_null
  }
}

/// Phillips–Perron test; `Err` below 20 points, on non-finite, constant or exactly fitted data, a
/// singular fit or a zero or non-finite variance; panics unless `0 < alpha < 1` (`Tau`: `>= 0.01`).
pub fn phillips_perron_test(
  y: ArrayView1<f64>,
  cfg: PhillipsPerronConfig,
) -> Result<PhillipsPerronTest, StatsError> {
  assert_alpha(cfg.alpha);
  let critical_values = match cfg.test_type {
    PpTestType::Tau => Some(adf_critical_values(cfg.trend)),
    PpTestType::Rho => None,
  };
  let critical = critical_values.as_ref().map(|cv| cv.value_at(cfg.alpha));
  let y = as_slice_or_copy(y);
  validate(&y, 20)?;
  reject_constant(&y, "constant series")?;
  let y = unit_scaled(match cfg.trend {
    Trend::None => y.into_owned(),
    Trend::Constant | Trend::ConstantTrend => centred(&unit_scaled(y.into_owned())),
  });

  let fit = fit_adf(&y, 0, cfg.trend, &y)?;
  let u = fit.residuals;
  let n_f = u.len() as f64;
  let gamma0 = u.iter().map(|v| v * v).sum::<f64>() / n_f;

  let used_lags = cfg.lags.unwrap_or_else(|| schwert_max_lags(y.len()));
  let lam2 = newey_west_long_run_variance(&u, used_lags)?;
  let lam = lam2.sqrt();
  let se = fit.std_err_gamma;

  // Phillips & Perron (1988) as Stata's `pperron` prints them: `s2` has the regression's `n − k`
  // degrees of freedom, and the ADF regression's `gamma` is `ρ̂ − 1`.
  let s2 = fit.sigma2;
  let statistic = match cfg.test_type {
    PpTestType::Rho => n_f * fit.gamma - 0.5 * (n_f * n_f * se * se / s2) * (lam2 - gamma0),
    PpTestType::Tau => {
      (gamma0 / lam2).sqrt() * (fit.gamma / se)
        - 0.5 * ((lam2 - gamma0) / lam) * (n_f * se / s2.sqrt())
    }
  };

  Ok(PhillipsPerronTest {
    statistic,
    p_value: None,
    critical_values,
    rejects_null: critical.map(|c| statistic < c),
    used_lags,
    test_type: cfg.test_type,
  })
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::PhillipsPerronConfig;
  use super::PpTestType;
  use super::phillips_perron_test;
  use crate::stationarity::Trend;

  fn simulate_ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      (0..n).map(|_| dist.sample()).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = phi * x[t - 1] + innovations[t];
    }
    x
  }

  fn simulate_random_walk(n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      (0..n).map(|_| dist.sample()).collect::<Vec<_>>()
    };

    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = x[t - 1] + innovations[t];
    }
    x
  }

  #[test]
  fn pp_tau_rejects_stationary_ar1() {
    let x = simulate_ar1(0.75, 2500, 0xA11CE);
    let cfg = PhillipsPerronConfig {
      trend: Trend::Constant,
      test_type: PpTestType::Tau,
      lags: Some(12),
      ..PhillipsPerronConfig::default()
    };
    let res = phillips_perron_test(ndarray::ArrayView1::from(&x), cfg).unwrap();
    assert_eq!(
      res.rejects_null,
      Some(true),
      "expected rejection, got {res:?}"
    );
  }

  #[test]
  fn pp_tau_random_walk_is_not_rejected_at_one_percent() {
    let x = simulate_random_walk(2500, 0xBADC0DE);
    let cfg = PhillipsPerronConfig {
      trend: Trend::Constant,
      test_type: PpTestType::Tau,
      lags: Some(12),
      alpha: 0.01,
    };
    let res = phillips_perron_test(ndarray::ArrayView1::from(&x), cfg).unwrap();
    assert_eq!(
      res.rejects_null,
      Some(false),
      "expected no rejection, got {res:?}"
    );
  }

  /// arch 9.0.0's `PhillipsPerron(y, lags=6, trend, test_type)`, whose statistics are Stata's.
  #[test]
  fn the_statistics_match_arch() {
    let normals = |n: usize, seed: u64| {
      let mut e = vec![0.0; n];
      SimdNormal::<f64>::new(0.0, 1.0)
        .seeded(&Deterministic::new(seed))
        .fill_slice(&mut e);
      e
    };
    let mut x = 0.0;
    let ar = normals(200, 21)
      .into_iter()
      .map(|e| {
        x = 0.6 * x + e;
        x
      })
      .collect::<Vec<_>>();
    let (mut level, mut step) = (0.0, 0.0);
    let walk = normals(250, 22)
      .into_iter()
      .map(|e| {
        step = e + 0.5 * step;
        level += step;
        level
      })
      .collect::<Vec<_>>();
    let cases = [
      (&ar, Trend::None, PpTestType::Tau, -7.0868928000863605),
      (&ar, Trend::None, PpTestType::Rho, -80.46433636486303),
      (&ar, Trend::Constant, PpTestType::Tau, -7.1339841826824655),
      (&ar, Trend::Constant, PpTestType::Rho, -81.62660721468625),
      (
        &ar,
        Trend::ConstantTrend,
        PpTestType::Tau,
        -7.171622630513965,
      ),
      (
        &ar,
        Trend::ConstantTrend,
        PpTestType::Rho,
        -82.24578562649376,
      ),
      (&walk, Trend::None, PpTestType::Tau, 1.4679944742856357),
      (&walk, Trend::None, PpTestType::Rho, 2.6726474640935187),
      (&walk, Trend::Constant, PpTestType::Tau, 0.32981219666128614),
      (&walk, Trend::Constant, PpTestType::Rho, 0.9543235554751717),
      (
        &walk,
        Trend::ConstantTrend,
        PpTestType::Tau,
        -0.3490891885094801,
      ),
      (
        &walk,
        Trend::ConstantTrend,
        PpTestType::Rho,
        -1.1256717679899748,
      ),
    ];
    for (y, trend, test_type, arch) in cases {
      let cfg = PhillipsPerronConfig {
        trend,
        test_type,
        lags: Some(6),
        alpha: 0.05,
      };
      let got = phillips_perron_test(ndarray::ArrayView1::from(y), cfg)
        .unwrap()
        .statistic;
      assert!(
        (got - arch).abs() <= 1e-10 * arch.abs(),
        "{trend:?} {test_type:?}: {got} against arch's {arch}"
      );
    }
  }

  #[test]
  fn a_constant_series_is_degenerate_data() {
    let x = ndarray::Array1::<f64>::from_elem(64, 0.3);
    assert_eq!(
      phillips_perron_test(x.view(), PhillipsPerronConfig::default()).unwrap_err(),
      crate::StatsError::DegenerateData("constant series")
    );
  }
}
