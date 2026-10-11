//! # Multivariate Student-t copula
//!
//! $$
//! C_{\Sigma,\nu}(u) = t_{\Sigma,\nu}\!\big(t_\nu^{-1}(u_1), \dots, t_\nu^{-1}(u_d)\big),
//! $$
//!
//! where $t_{\Sigma,\nu}$ is the $d$-variate Student-t CDF with correlation
//! $\Sigma$ and degrees of freedom $\nu > 0$, and $t_\nu^{-1}$ the univariate
//! Student-t quantile. The corresponding density of the copula is
//!
//! $$
//! c_{\Sigma,\nu}(u) = \frac{f_{\Sigma,\nu}(z)}{\prod_{j=1}^{d} f_\nu(z_j)},
//! \qquad z_j = t_\nu^{-1}(u_j),
//! $$
//!
//! with
//!
//! $$
//! f_{\Sigma,\nu}(z) = \frac{\Gamma\!\big(\tfrac{\nu+d}{2}\big)}
//! {\Gamma\!\big(\tfrac{\nu}{2}\big)\,(\nu\pi)^{d/2}\,|\Sigma|^{1/2}}
//! \left(1 + \tfrac{1}{\nu}\,z^\top \Sigma^{-1} z\right)^{-\tfrac{\nu+d}{2}}.
//! $$
//!
//! **Sampling (Demarta-McNeil 2005, §2.3).** Let $Z \sim N(0,\Sigma)$ and
//! $W \sim \chi^2_\nu / \nu$ independent. Then $X = Z / \sqrt{W}$ is $d$-variate
//! Student-$t$ with $(\nu, \Sigma)$, and $U_j = t_\nu(X_j)$ are the copula
//! marginals. The $\nu \to \infty$ limit collapses to the Gaussian copula.
//!
//! **Tail dependence.** Symmetric upper/lower:
//! $\lambda = 2\,t_{\nu+1}\!\big(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big)$ for
//! every off-diagonal pair $\rho_{ij}$ — strictly positive for finite $\nu$,
//! in contrast to the Gaussian copula.
//!
//! **Calibration.** [`fit`](TMultivariate::fit) implements the two-stage
//! pseudo-MLE of Demarta-McNeil 2005, §6: (a) recover $\Sigma$ from the
//! sample Kendall-τ via $\rho_{ij} = \sin(\tfrac{\pi}{2}\tau_{ij})$
//! (rank-based, robust to ν), then (b) maximise the log-likelihood in
//! $\nu$ by 1-D Brent search on $[2.1, 50]$ holding $\Sigma$ fixed.
//!
//! References:
//! - Demarta, S., McNeil, A.J. (2005), "The t copula and related copulas",
//!   *International Statistical Review* 73(1), 111-129.
//! - Embrechts, P., Lindskog, F., McNeil, A.J. (2003), "Modelling
//!   Dependence with Copulas and Applications to Risk Management",
//!   in *Handbook of Heavy Tailed Distributions in Finance*, Elsevier,
//!   ch. 8.
//! - McNeil, Frey, Embrechts (2015), *Quantitative Risk Management*,
//!   Princeton UP, §7.5.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::beta_i;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::special::ndtri;

use super::MultivariateFamily;
use super::correlation_matrix::CorrelationMatrix;
use super::correlation_matrix::shrink_to_spd;
use crate::bivariate::t_copula::check_nu;
use crate::correlation::kendall_tau;
use crate::error::CopulaError;
use crate::traits::MultivariateExt;

/// Multivariate Student-$t$ copula with correlation $\Sigma$ and degrees of freedom $\nu$; as $\nu \to \infty$ it
/// collapses to the Gaussian copula, and for finite $\nu$ every pair has symmetric tail dependence.
#[derive(Debug, Clone)]
pub struct TMultivariate {
  /// Degrees of freedom, `0 < nu < inf`; 4 by default.
  nu: f64,
  corr: Option<CorrelationMatrix>,
}

impl Default for TMultivariate {
  fn default() -> Self {
    Self {
      nu: 4.0,
      corr: None,
    }
  }
}

impl TMultivariate {
  /// The t copula of correlation `corr` and `nu` degrees of freedom; panics unless `corr` is a correlation matrix and
  /// `0 < nu < ∞` with `nu / 2 > 0`.
  pub fn new(corr: Array2<f64>, nu: f64) -> Self {
    Self::try_new(corr, nu).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the error returned instead of a panic; `nu` is checked first.
  pub fn try_new(corr: Array2<f64>, nu: f64) -> Result<Self, CopulaError> {
    check_nu(nu)?;
    Ok(Self {
      nu,
      corr: Some(CorrelationMatrix::try_new(corr)?),
    })
  }

  /// Returns a reference to the internal correlation matrix, if fitted.
  pub fn correlation(&self) -> Option<&Array2<f64>> {
    self.corr.as_ref().map(CorrelationMatrix::corr)
  }

  /// Current degrees of freedom $\nu$.
  pub fn nu(&self) -> f64 {
    self.nu
  }

  /// Overrides $\nu$, skipping the fit's own search; panics unless `0 < nu < ∞` and `nu / 2 > 0`.
  pub fn set_nu(&mut self, nu: f64) {
    check_nu(nu).unwrap_or_else(|e| panic!("{e}"));
    self.nu = nu;
  }

  fn fitted(&self) -> Result<&CorrelationMatrix, CopulaError> {
    self.corr.as_ref().ok_or(CopulaError::NotFitted)
  }

  /// Standard Student-$t$ density $f_\nu(x)$ with the natural-log
  /// normaliser kept as a separate term for log-pdf composition.
  fn t_log_pdf(x: f64, nu: f64) -> f64 {
    let log_norm =
      ln_gamma(0.5 * (nu + 1.0)) - 0.5 * (nu * std::f64::consts::PI).ln() - ln_gamma(0.5 * nu);
    let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
    log_norm + log_kernel
  }

  /// Standard Student-$t$ CDF $F_\nu(x)$ via the regularised
  /// incomplete-beta identity.
  fn t_cdf(x: f64, nu: f64) -> f64 {
    if !x.is_finite() {
      return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let t = nu / (nu + x * x);
    let half = 0.5 * beta_i(0.5 * nu, 0.5, t);
    if x >= 0.0 { 1.0 - half } else { half }
  }

  /// Quantile $t_\nu^{-1}(p)$: Cornish-Fisher-style normal seed refined by
  /// 40 Newton steps. Same routine as
  /// [`crate::bivariate::t_copula::TCopula::t_quantile`].
  fn t_quantile(p: f64, nu: f64) -> f64 {
    if p <= 0.0 {
      return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    let z = ndtri(p);
    let mut x = z * (1.0 + (z * z + 1.0) / (4.0 * nu));
    for _ in 0..40 {
      let cdf = Self::t_cdf(x, nu);
      let f = cdf - p;
      let pdf_log = Self::t_log_pdf(x, nu);
      if !pdf_log.is_finite() {
        break;
      }
      let pdf = pdf_log.exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = x - dx;
      if (new_x - x).abs() < 1e-14 * (1.0 + x.abs()) {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  /// Map U ∈ (0,1)^{n×d} → Z ∈ ℝ^{n×d} via the univariate $t_\nu^{-1}$.
  fn transform_to_t(u: ArrayView2<f64>, nu: f64) -> Array2<f64> {
    let eps = 1e-12;
    let mut z = u.to_owned();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        let clamped = val.clamp(eps, 1.0 - eps);
        *val = Self::t_quantile(clamped, nu);
      }
    }
    z
  }

  /// Build $\Sigma$ from sample Kendall-τ via $\rho_{ij} = \sin(\pi\tau_{ij}/2)$.
  /// This is the rank-based identity that holds for **any** elliptical
  /// copula, hence is ν-agnostic and robust to the t-copula's heavy tails.
  fn estimate_corr_from_kendall(x: ArrayView2<f64>) -> Array2<f64> {
    let d = x.ncols();
    let tau = kendall_tau(x);
    let mut corr = Array2::<f64>::zeros((d, d));
    for i in 0..d {
      corr[[i, i]] = 1.0;
    }
    for i in 0..d {
      for j in (i + 1)..d {
        let rho = (0.5 * std::f64::consts::PI * tau[[i, j]])
          .sin()
          .clamp(-0.999_999, 0.999_999);
        corr[[i, j]] = rho;
        corr[[j, i]] = rho;
      }
    }
    corr
  }

  /// Log-density of the multivariate Student-$t$ at the un-transformed
  /// argument `z` (i.e. **before** dividing by the marginal $f_\nu(z_j)$ to
  /// obtain the copula density). Used internally by [`log_pdf`].
  fn mv_log_pdf(corr: &CorrelationMatrix, nu: f64, z: ArrayView2<f64>) -> Array1<f64> {
    let inv = corr.inverse();
    let log_det = corr.log_det();
    let d = corr.dim() as f64;
    let log_norm = ln_gamma(0.5 * (nu + d))
      - ln_gamma(0.5 * nu)
      - 0.5 * d * (nu * std::f64::consts::PI).ln()
      - 0.5 * log_det;
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let row_owned = row.to_owned();
      let q = inv.dot(&row_owned).dot(&row_owned);
      out[i] = log_norm - 0.5 * (nu + d) * (1.0 + q / nu).ln();
    }
    out
  }

  /// Log-likelihood of the data under `corr` and the candidate $\nu$; the 1-D
  /// Brent search in [`fit`] varies $\nu$ and never $\Sigma$.
  fn log_likelihood_for_nu(corr: &CorrelationMatrix, x: ArrayView2<f64>, nu: f64) -> f64 {
    let z = Self::transform_to_t(x, nu);
    let mv = Self::mv_log_pdf(corr, nu, z.view());
    let mut marginal_log = 0.0;
    for v in z.iter() {
      marginal_log += Self::t_log_pdf(*v, nu);
    }
    mv.iter().sum::<f64>() - marginal_log
  }

  /// Brent minimisation on the negative log-likelihood w.r.t. $\nu$, holding
  /// $\Sigma$ fixed. Bracket $[\nu_{\min}, \nu_{\max}] = [2.1, 50]$ matches
  /// Demarta-McNeil (2005, §6) where the upper bound is taken as the
  /// effective "Gaussian limit" threshold.
  fn optimise_nu(corr: &CorrelationMatrix, x: ArrayView2<f64>) -> f64 {
    let f = |nu: f64| -Self::log_likelihood_for_nu(corr, x, nu);
    // Brent on [2.1, 50] with golden-section initialisation.
    let mut a = 2.1_f64;
    let mut b = 50.0_f64;
    let golden = (5f64.sqrt() - 1.0) / 2.0;
    let mut x1 = b - golden * (b - a);
    let mut x2 = a + golden * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..60 {
      if (b - a).abs() < 1e-4 {
        break;
      }
      if f1 < f2 {
        b = x2;
        x2 = x1;
        f2 = f1;
        x1 = b - golden * (b - a);
        f1 = f(x1);
      } else {
        a = x1;
        x1 = x2;
        f1 = f2;
        x2 = a + golden * (b - a);
        f2 = f(x2);
      }
    }
    0.5 * (a + b)
  }
}

impl MultivariateExt for TMultivariate {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::TMultivariate
  }

  /// The normal and the $\chi^2$ stream both come from `seed`, which advances between them.
  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    let corr = self.fitted()?;
    let d = corr.dim();
    let l = corr.chol_lower();
    // Z ~ N(0, Σ) by L · G with G ~ N(0, I).
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(seed);
    let g = Array2::from_shape_fn((n, d), |_| normal.sample());
    let z = g.dot(&l.t());
    // W ~ χ²_ν / ν, independently per row.
    let mut chi = SimdChiSquared::<f64>::new(self.nu).seeded(seed);
    let mut u = Array2::<f64>::zeros((n, d));
    for r in 0..n {
      let w_raw = chi.sample();
      let w = (w_raw / self.nu).max(1e-300);
      let scale = 1.0 / w.sqrt();
      for c in 0..d {
        let xc = z[[r, c]] * scale;
        u[[r, c]] = Self::t_cdf(xc, self.nu).clamp(1e-12, 1.0 - 1e-12);
      }
    }
    Ok(u)
  }

  fn fit(&mut self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    if X.ncols() < 2 {
      return Err(CopulaError::InvalidStructure(format!(
        "a t-copula needs dim >= 2, got {}",
        X.ncols()
      )));
    }
    if X.nrows() < 2 {
      return Err(CopulaError::InsufficientData {
        needed: 2,
        got: X.nrows(),
      });
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    let corr = CorrelationMatrix::try_new(shrink_to_spd(Self::estimate_corr_from_kendall(X)))?;
    // Profile MLE for ν conditional on Σ.
    self.nu = Self::optimise_nu(&corr, X);
    self.corr = Some(corr);
    Ok(())
  }

  fn check_fit(&self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    let dim = self.fitted()?.dim();
    if X.ncols() != dim {
      return Err(CopulaError::DimensionMismatch {
        expected: dim,
        got: X.ncols(),
      });
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    Ok(())
  }

  fn pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let nu = self.nu;
    let z = Self::transform_to_t(X, nu);
    let mv = Self::mv_log_pdf(self.fitted()?, nu, z.view());
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut marginal_log = 0.0;
      for v in row.iter() {
        marginal_log += Self::t_log_pdf(*v, nu);
      }
      out[i] = (mv[i] - marginal_log).exp();
    }
    Ok(out)
  }

  fn log_pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let nu = self.nu;
    let z = Self::transform_to_t(X, nu);
    let mv = Self::mv_log_pdf(self.fitted()?, nu, z.view());
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut marginal_log = 0.0;
      for v in row.iter() {
        marginal_log += Self::t_log_pdf(*v, nu);
      }
      out[i] = mv[i] - marginal_log;
    }
    Ok(out)
  }

  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    // Closed forms for the multivariate Student-t CDF exist only in d ≤ 2
    // (Dunnett-Sobel 1D reduction); for d ≥ 3 we estimate via the χ²-mixer
    // representation: 1/m Σ_r 1{ Z_r/√W_r ≤ z } with Z_r ~ N(0,Σ),
    // W_r ~ χ²_ν / ν. 4000 MC samples per query match the Gaussian copula
    // CDF estimator's tolerance.
    let z = Self::transform_to_t(X, self.nu);
    let corr = self.fitted()?;
    let l = corr.chol_lower();
    let d = corr.dim();
    let n = z.nrows();
    let m = 4000usize;
    let mut out = Array1::<f64>::zeros(n);
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Unseeded);
    let mut chi = SimdChiSquared::<f64>::new(self.nu).seeded(&Unseeded);
    let g = Array2::from_shape_fn((m, d), |_| normal.sample());
    let y = g.dot(&l.t());
    let mut w_buf = vec![0.0f64; m];
    for v in w_buf.iter_mut() {
      let w = (chi.sample() / self.nu).max(1e-300);
      *v = 1.0 / w.sqrt();
    }
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut count = 0usize;
      'outer: for r in 0..m {
        for c in 0..d {
          if y[[r, c]] * w_buf[r] > row[c] {
            continue 'outer;
          }
        }
        count += 1;
      }
      out[i] = count as f64 / m as f64;
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests;
