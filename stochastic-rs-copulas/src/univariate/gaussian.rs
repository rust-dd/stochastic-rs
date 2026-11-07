//! # Gaussian (univariate margin)
//!
//! $$
//! F(x) = \Phi\!\left(\frac{x - \mu}{\sigma}\right), \qquad (\mu, \sigma)
//! \text{ fit from data}
//! $$
//!
//! [`GaussianUnivariate::fit`] estimates `(mean, std)` from a column and `cdf` maps raw values onto
//! `[0, 1]`, as a copula `fit` expects; unlike [`crate::empirical`] it extrapolates smoothly.

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::error::CopulaError;

/// `N(mu, sigma²)` as a marginal transform onto `[0, 1]`.
///
/// ```
/// use ndarray::array;
/// use stochastic_rs_copulas::univariate::gaussian::GaussianUnivariate;
///
/// let g = GaussianUnivariate::fit(array![1.0, 2.0, 3.0, 4.0, 5.0].view()).unwrap();
/// // The sample mean (3.0) maps to the median of the fitted normal.
/// assert!((g.cdf(3.0) - 0.5).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianUnivariate {
  mu: f64,
  sigma: f64,
}

impl GaussianUnivariate {
  /// Panics unless `sigma > 0` and both are finite.
  pub fn new(mu: f64, sigma: f64) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma > 0.0,
      "sigma must satisfy `sigma > 0.0`, got sigma = {sigma:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    Self { mu, sigma }
  }

  /// Sample mean and (n − 1) standard deviation of `column`; `InsufficientData` below two values, `InvalidStructure`
  /// for a constant or non-finite column.
  pub fn fit(column: ArrayView1<f64>) -> Result<Self, CopulaError> {
    let n = column.len();
    if n < 2 {
      return Err(CopulaError::InsufficientData { needed: 2, got: n });
    }
    if column.iter().all(|&x| x == column[0]) {
      return Err(CopulaError::InvalidStructure(format!(
        "a Gaussian margin needs a positive variance, got {n} equal values {:?}",
        column[0]
      )));
    }
    let mu = column.sum() / n as f64;
    let var = column.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    if !(var > 0.0 && var.is_finite()) {
      return Err(CopulaError::InvalidStructure(format!(
        "a Gaussian margin needs a positive variance, got {var:?}"
      )));
    }
    Ok(Self {
      mu,
      sigma: var.sqrt(),
    })
  }

  pub fn mu(&self) -> f64 {
    self.mu
  }

  pub fn sigma(&self) -> f64 {
    self.sigma
  }

  pub fn cdf(&self, x: f64) -> f64 {
    norm_cdf((x - self.mu) / self.sigma)
  }

  pub fn pdf(&self, x: f64) -> f64 {
    norm_pdf((x - self.mu) / self.sigma) / self.sigma
  }

  /// The quantile function, the inverse of `cdf`.
  #[doc(alias = "ppf")]
  pub fn percent_point(&self, p: f64) -> f64 {
    self.mu + self.sigma * ndtri(p)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::Axis;
  use ndarray::array;
  use ndarray::stack;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;
  use crate::bivariate::frank::Frank;
  use crate::traits::BivariateExt;

  /// After `fit`, `percent_point` inverts `cdf` and the median of the fitted margin is its mean.
  #[test]
  fn percent_point_inverts_cdf_after_fit() {
    let g = GaussianUnivariate::fit(array![1.0, 2.0, 3.0, 4.0, 5.0].view()).unwrap();
    for x in [-2.0, 0.5, 3.0, 4.2, 8.0] {
      let back = g.percent_point(g.cdf(x));
      assert!(
        (back - x).abs() < 1e-9,
        "x = {x}: percent_point(cdf(x)) = {back}"
      );
    }
    assert!((g.percent_point(0.5) - 3.0).abs() < 1e-12);
  }

  /// Two Gaussian columns through `fit` + `cdf` pass `check_marginal`'s uniformity diagnostic and fit a copula that
  /// reaches both signs of τ, as the independent draws need.
  #[test]
  fn gaussian_univariate_output_feeds_a_real_bivariate_copula() {
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(11));
    let mut x = Array1::<f64>::zeros(600);
    normal.fill_slice(x.as_slice_mut().unwrap());
    let mut y = Array1::<f64>::zeros(600);
    normal.fill_slice(y.as_slice_mut().unwrap());

    let gx = GaussianUnivariate::fit(x.view()).unwrap();
    let gy = GaussianUnivariate::fit(y.view()).unwrap();

    let u = x.mapv(|v| gx.cdf(v));
    let v = y.mapv(|v| gy.cdf(v));
    let mut frank = Frank::default();
    assert_eq!(frank.check_marginal(u.view()), Ok(()));
    assert_eq!(frank.check_marginal(v.view()), Ok(()));
    let data = stack![Axis(1), u, v];

    frank
      .fit(data.view())
      .expect("the transformed margins are a two-column sample in [0, 1]");
    assert!(frank.theta().is_some());
  }

  #[test]
  fn a_margin_needs_two_values_and_a_spread() {
    assert_eq!(
      GaussianUnivariate::fit(array![1.0].view()).unwrap_err(),
      CopulaError::InsufficientData { needed: 2, got: 1 }
    );
    for (value, n) in [
      (2.0, 3),
      (0.1, 3),
      (0.7, 7),
      (0.1, 50),
      (1.0 / 3.0, 10),
      (123456.789, 10),
    ] {
      let got = GaussianUnivariate::fit(Array1::from_elem(n, value).view());
      assert!(
        matches!(got, Err(CopulaError::InvalidStructure(_))),
        "{n} x {value}: {got:?}"
      );
    }
    let g = GaussianUnivariate::fit(array![1.0, 2.0, 3.0, 4.0, 5.0].view()).unwrap();
    assert_eq!((g.mu(), g.sigma()), (3.0, 2.5_f64.sqrt()));
    assert!((g.cdf(3.0) - 0.5).abs() < 1e-12 && (g.percent_point(0.5) - 3.0).abs() < 1e-12);
  }

  /// A NaN or infinite entry leaves no finite variance, so no margin.
  #[test]
  fn a_non_finite_column_has_no_margin() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let got = GaussianUnivariate::fit(array![1.0, 2.0, bad].view());
      assert!(
        matches!(got, Err(CopulaError::InvalidStructure(_))),
        "{bad}: {got:?}"
      );
    }
  }

  #[test]
  #[should_panic(expected = "sigma must satisfy `sigma > 0.0`, got sigma = 0.0")]
  fn a_zero_sigma_is_a_precondition() {
    let _ = GaussianUnivariate::new(0.0, 0.0);
  }

  #[test]
  #[should_panic(expected = "sigma must satisfy `sigma.is_finite()`, got sigma = inf")]
  fn an_infinite_sigma_is_a_precondition() {
    let _ = GaussianUnivariate::new(0.0, f64::INFINITY);
  }

  #[test]
  #[should_panic(expected = "mu must satisfy `mu.is_finite()`, got mu = NaN")]
  fn a_nan_mu_is_a_precondition() {
    let _ = GaussianUnivariate::new(f64::NAN, 1.0);
  }

  #[test]
  fn a_margin_built_from_its_parameters_evaluates_them() {
    let g = GaussianUnivariate::new(1.0, 2.0);
    assert!((g.cdf(1.0) - 0.5).abs() < 1e-15);
    assert!((g.pdf(1.0) - 1.0 / (2.0 * (2.0 * std::f64::consts::PI).sqrt())).abs() < 1e-15);
    assert!((g.percent_point(0.975) - (1.0 + 2.0 * 1.959963984540054)).abs() < 1e-9);
  }
}
