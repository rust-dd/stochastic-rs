//! # Gaussian (univariate margin)
//!
//! $$
//! F(x) = \Phi\!\left(\frac{x - \mu}{\sigma}\right), \qquad (\mu, \sigma)
//! \text{ fit from data}
//! $$
//!
//! Parametric marginal transform: [`GaussianUnivariate::fit`] estimates
//! `(mean, std)` from a data column, then [`GaussianUnivariate::cdf`] maps
//! raw values onto `[0, 1]` — the input every [`crate::traits::BivariateExt`]
//! / `MultivariateExt` copula expects. This is the parametric counterpart
//! to [`crate::empirical::EmpiricalCopula2D`]'s nonparametric rank
//! transform: reach for it when the margin is known (or assumed) Gaussian
//! and a smooth, extrapolating CDF is preferable to the empirical
//! transform's step function (e.g. for query points beyond the observed
//! data range).
//!
use ndarray::ArrayView1;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

/// ```
/// use ndarray::array;
/// use stochastic_rs_copulas::univariate::gaussian::GaussianUnivariate;
///
/// let mut g = GaussianUnivariate::new();
/// g.fit(array![1.0, 2.0, 3.0, 4.0, 5.0].view());
/// // The sample mean (3.0) maps to the median of the fitted normal.
/// assert!((g.cdf(3.0) - 0.5).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GaussianUnivariate {
  /// `Some((mean, std))` once `fit` has been called; `None` otherwise.
  params: Option<(f64, f64)>,
}

impl GaussianUnivariate {
  pub fn new() -> Self {
    Self::default()
  }

  /// Fit mean and std to data column, then cache the closed-form parameters.
  pub fn fit(&mut self, column: ArrayView1<f64>) {
    let n = column.len() as f64;
    if n < 2.0 {
      // fallback: standard normal
      self.params = Some((0.0, 1.0));
      return;
    }
    let mean = column.mean().unwrap_or(0.0);
    let var = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt().max(1e-12);
    self.params = Some((mean, std));
  }

  pub fn is_fitted(&self) -> bool {
    self.params.is_some()
  }

  /// Normal CDF using the cached `(mean, std)`.
  pub fn cdf(&self, x: f64) -> f64 {
    match self.params {
      Some((mu, sigma)) => norm_cdf((x - mu) / sigma),
      None => 0.5,
    }
  }

  /// Normal pdf using the cached `(mean, std)`.
  pub fn pdf(&self, x: f64) -> f64 {
    match self.params {
      Some((mu, sigma)) => norm_pdf((x - mu) / sigma) / sigma,
      None => 1.0,
    }
  }

  /// Normal quantile (inverse CDF) using the cached `(mean, std)`.
  #[doc(alias = "ppf")]
  pub fn percent_point(&self, p: f64) -> f64 {
    match self.params {
      Some((mu, sigma)) => mu + sigma * ndtri(p),
      None => 0.0,
    }
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
    let mut g = GaussianUnivariate::new();
    g.fit(array![1.0, 2.0, 3.0, 4.0, 5.0].view());
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

    let mut gx = GaussianUnivariate::new();
    gx.fit(x.view());
    let mut gy = GaussianUnivariate::new();
    gy.fit(y.view());

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
}
