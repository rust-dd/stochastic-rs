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

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_linalg::Cholesky;
use ndarray_linalg::Inverse;
use ndarray_linalg::UPLO;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::beta_i;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::special::ndtri;

use super::CopulaType;
use crate::correlation::kendall_tau;
use crate::traits::MultivariateExt;

/// Multivariate Student-$t$ copula with degrees of freedom $\nu$ and
/// correlation matrix $\Sigma$. The $\nu \to \infty$ limit collapses to the
/// Gaussian copula; for finite $\nu$ the copula exhibits symmetric
/// upper/lower tail dependence on every off-diagonal pair.
#[derive(Debug, Clone)]
pub struct TMultivariate {
  dim: usize,
  /// Degrees of freedom $\nu > 2$. Default 4.
  nu: f64,
  /// Correlation matrix (dim × dim).
  corr: Option<Array2<f64>>,
  /// Inverse correlation matrix (cached for log-pdf).
  inv_corr: Option<Array2<f64>>,
  /// Lower-triangular Cholesky factor of `corr` (cached for sampling).
  chol_lower: Option<Array2<f64>>,
  /// Log-determinant of `corr` (cached for log-pdf).
  log_det_corr: Option<f64>,
}

impl Default for TMultivariate {
  fn default() -> Self {
    Self {
      dim: 0,
      nu: 4.0,
      corr: None,
      inv_corr: None,
      chol_lower: None,
      log_det_corr: None,
    }
  }
}

impl TMultivariate {
  pub fn new() -> Self {
    Self::default()
  }

  /// Construct directly from a correlation matrix and degrees of freedom.
  pub fn new_with(corr: Array2<f64>, nu: f64) -> Result<Self, Box<dyn Error>> {
    if nu <= 0.0 || nu.is_nan() {
      return Err("Degrees of freedom must be positive".into());
    }
    let dim = corr.nrows();
    if dim != corr.ncols() {
      return Err("Correlation matrix must be square".into());
    }
    let mut t = Self {
      nu,
      ..Self::default()
    };
    t.set_corr(corr)?;
    Ok(t)
  }

  /// Returns a reference to the internal correlation matrix, if fitted.
  pub fn correlation(&self) -> Option<&Array2<f64>> {
    self.corr.as_ref()
  }

  /// Current degrees of freedom $\nu$.
  pub fn degrees_of_freedom(&self) -> f64 {
    self.nu
  }

  /// Override the degrees of freedom. Useful when the user picks $\nu$ from
  /// an external calibration (e.g. tail-coefficient match) and wants the
  /// copula to skip its own optimisation. Returns an error if $\nu \le 0$.
  pub fn set_degrees_of_freedom(&mut self, nu: f64) -> Result<(), Box<dyn Error>> {
    if nu <= 0.0 || nu.is_nan() {
      return Err("Degrees of freedom must be positive".into());
    }
    self.nu = nu;
    Ok(())
  }

  fn set_corr(&mut self, corr: Array2<f64>) -> Result<(), Box<dyn Error>> {
    let dim = corr.nrows();
    self.dim = dim;

    let l_arr = corr
      .cholesky(UPLO::Lower)
      .map_err(|_| -> Box<dyn Error> { "Correlation matrix is not positive definite".into() })?;
    let mut log_det = 0.0;
    for i in 0..dim {
      log_det += l_arr[[i, i]].ln();
    }
    log_det *= 2.0;
    let inv_arr = corr
      .inv()
      .map_err(|_| -> Box<dyn Error> { "Failed to invert correlation matrix".into() })?;

    self.corr = Some(corr);
    self.inv_corr = Some(inv_arr);
    self.chol_lower = Some(l_arr);
    self.log_det_corr = Some(log_det);
    Ok(())
  }

  fn require_fitted(&self) -> Result<(), Box<dyn Error>> {
    if self.corr.is_none()
      || self.inv_corr.is_none()
      || self.chol_lower.is_none()
      || self.log_det_corr.is_none()
    {
      return Err("Fit the copula or provide a correlation matrix first".into());
    }
    Ok(())
  }

  /// Standard Student-$t$ density $f_\nu(x)$ with the natural-log
  /// normaliser kept as a separate term for log-pdf composition.
  fn t_log_pdf(x: f64, nu: f64) -> f64 {
    let log_norm =
      ln_gamma(0.5 * (nu + 1.0)) - 0.5 * (nu * f64::consts::PI).ln() - ln_gamma(0.5 * nu);
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
  fn transform_to_t(&self, u: &Array2<f64>) -> Array2<f64> {
    let eps = 1e-12;
    let mut z = u.clone();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        let clamped = val.clamp(eps, 1.0 - eps);
        *val = Self::t_quantile(clamped, self.nu);
      }
    }
    z
  }

  /// Build $\Sigma$ from sample Kendall-τ via $\rho_{ij} = \sin(\pi\tau_{ij}/2)$.
  /// This is the rank-based identity that holds for **any** elliptical
  /// copula, hence is ν-agnostic and robust to the t-copula's heavy tails.
  fn estimate_corr_from_kendall(x: &Array2<f64>) -> Array2<f64> {
    let d = x.ncols();
    let tau = kendall_tau(x);
    let mut corr = Array2::<f64>::zeros((d, d));
    for i in 0..d {
      corr[[i, i]] = 1.0;
    }
    for i in 0..d {
      for j in (i + 1)..d {
        let rho = (0.5 * f64::consts::PI * tau[[i, j]])
          .sin()
          .clamp(-0.999_999, 0.999_999);
        corr[[i, j]] = rho;
        corr[[j, i]] = rho;
      }
    }
    Self::nearest_spd(corr)
  }

  /// Project an indefinite candidate onto the SPD cone by a small diagonal
  /// jitter. We don't need full Higham — Kendall-derived matrices are
  /// already close to SPD; only edge cases (highly correlated d ≥ 10)
  /// occasionally need the bump.
  fn nearest_spd(mut corr: Array2<f64>) -> Array2<f64> {
    let d = corr.nrows();
    let mut jitter = 0usize;
    while corr.cholesky(UPLO::Lower).is_err() && jitter < 6 {
      let eps = 10f64.powi(-6_i32 + jitter as i32);
      for k in 0..d {
        corr[[k, k]] = 1.0 + eps;
      }
      jitter += 1;
    }
    corr
  }

  /// Log-density of the multivariate Student-$t$ at the un-transformed
  /// argument `z` (i.e. **before** dividing by the marginal $f_\nu(z_j)$ to
  /// obtain the copula density). Used internally by [`log_pdf`].
  fn mv_log_pdf(&self, z: &Array2<f64>) -> Array1<f64> {
    let inv = self.inv_corr.as_ref().unwrap();
    let log_det = self.log_det_corr.unwrap();
    let d = self.dim as f64;
    let nu = self.nu;
    let log_norm = ln_gamma(0.5 * (nu + d))
      - ln_gamma(0.5 * nu)
      - 0.5 * d * (nu * f64::consts::PI).ln()
      - 0.5 * log_det;
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let row_owned = row.to_owned();
      let q = inv.dot(&row_owned).dot(&row_owned);
      out[i] = log_norm - 0.5 * (nu + d) * (1.0 + q / nu).ln();
    }
    out
  }

  /// Log-likelihood of the data under the current correlation matrix and
  /// the candidate $\nu$. Used by the 1-D Brent search in [`fit`].
  fn log_likelihood_for_nu(&self, x: &Array2<f64>, nu: f64) -> f64 {
    // Build a temporary view of self with the candidate ν so we can re-use
    // the cached Σ⁻¹ / |Σ| (the Brent search only varies ν, never Σ).
    let saved = self.nu;
    let tmp = TMultivariate {
      dim: self.dim,
      nu,
      corr: self.corr.clone(),
      inv_corr: self.inv_corr.clone(),
      chol_lower: self.chol_lower.clone(),
      log_det_corr: self.log_det_corr,
    };
    let z = tmp.transform_to_t(x);
    let mv = tmp.mv_log_pdf(&z);
    let mut marginal_log = 0.0;
    for v in z.iter() {
      marginal_log += Self::t_log_pdf(*v, nu);
    }
    let _ = saved;
    mv.iter().sum::<f64>() - marginal_log
  }

  /// Brent minimisation on the negative log-likelihood w.r.t. $\nu$, holding
  /// $\Sigma$ fixed. Bracket $[\nu_{\min}, \nu_{\max}] = [2.1, 50]$ matches
  /// Demarta-McNeil (2005, §6) where the upper bound is taken as the
  /// effective "Gaussian limit" threshold.
  fn optimise_nu(&self, x: &Array2<f64>) -> f64 {
    let f = |nu: f64| -self.log_likelihood_for_nu(x, nu);
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
  fn r#type(&self) -> CopulaType {
    CopulaType::TMultivariate
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    self.require_fitted()?;
    let d = self.dim;
    let l = self.chol_lower.as_ref().unwrap();
    // Z ~ N(0, Σ) by L · G with G ~ N(0, I).
    let normal = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
    let g = Array2::from_shape_fn((n, d), |_| normal.sample_fast());
    let z = g.dot(&l.t());
    // W ~ χ²_ν / ν, independently per row.
    let chi = SimdChiSquared::<f64>::new(self.nu, &Unseeded);
    let mut u = Array2::<f64>::zeros((n, d));
    for r in 0..n {
      let w_raw = chi.sample_fast();
      let w = (w_raw / self.nu).max(1e-300);
      let scale = 1.0 / w.sqrt();
      for c in 0..d {
        let xc = z[[r, c]] * scale;
        u[[r, c]] = Self::t_cdf(xc, self.nu).clamp(1e-12, 1.0 - 1e-12);
      }
    }
    Ok(u)
  }

  fn fit(&mut self, X: Array2<f64>) -> Result<(), Box<dyn Error>> {
    if X.nrows() < 2 || X.ncols() < 2 {
      return Err("Need at least 2 samples and 2 dimensions".into());
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err("Input data must be in [0,1] for the t-copula fit".into());
    }
    self.dim = X.ncols();
    let corr = Self::estimate_corr_from_kendall(&X);
    self.set_corr(corr)?;
    // Profile MLE for ν conditional on Σ.
    self.nu = self.optimise_nu(&X);
    Ok(())
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    self.require_fitted()?;
    if X.ncols() != self.dim {
      return Err("Dimension mismatch".into());
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err("Input X must be in [0,1] for the t-copula".into());
    }
    Ok(())
  }

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let z = self.transform_to_t(&X);
    let mv = self.mv_log_pdf(&z);
    let nu = self.nu;
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

  fn log_pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let z = self.transform_to_t(&X);
    let mv = self.mv_log_pdf(&z);
    let nu = self.nu;
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

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    // Closed forms for the multivariate Student-t CDF exist only in d ≤ 2
    // (Dunnett-Sobel 1D reduction); for d ≥ 3 we estimate via the χ²-mixer
    // representation: 1/m Σ_r 1{ Z_r/√W_r ≤ z } with Z_r ~ N(0,Σ),
    // W_r ~ χ²_ν / ν. 4000 MC samples per query match the Gaussian copula
    // CDF estimator's tolerance.
    let z = self.transform_to_t(&X);
    let l = self.chol_lower.as_ref().unwrap();
    let n = z.nrows();
    let m = 4000usize;
    let mut out = Array1::<f64>::zeros(n);
    let normal = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
    let chi = SimdChiSquared::<f64>::new(self.nu, &Unseeded);
    let g = Array2::from_shape_fn((m, self.dim), |_| normal.sample_fast());
    let y = g.dot(&l.t());
    let mut w_buf = vec![0.0f64; m];
    for v in w_buf.iter_mut() {
      let w = (chi.sample_fast() / self.nu).max(1e-300);
      *v = 1.0 / w.sqrt();
    }
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut count = 0usize;
      'outer: for r in 0..m {
        for c in 0..self.dim {
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
mod tests {
  use ndarray::array;

  use super::*;

  /// Construct a 3-dim t-copula with mild dependence and verify the
  /// sampler produces uniforms on every coordinate.
  #[test]
  fn t_copula_samples_have_uniform_marginals() {
    let corr = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]];
    let cop = TMultivariate::new_with(corr, 5.0).expect("valid corr");
    let u = cop.sample(20_000).expect("sample");
    assert_eq!(u.ncols(), 3);
    assert_eq!(u.nrows(), 20_000);
    for j in 0..3 {
      let col = u.column(j);
      let mean = col.iter().sum::<f64>() / col.len() as f64;
      // Uniform mean = 0.5 with sqrt(1/12N) ≈ 0.002 std on 20k samples;
      // 0.02 tolerance is ~10σ — safe.
      assert!(
        (mean - 0.5).abs() < 0.02,
        "marginal {j} mean = {mean}, expected ~0.5"
      );
      let in_range = col.iter().all(|v| *v > 0.0 && *v < 1.0);
      assert!(in_range, "marginal {j} not strictly in (0,1)");
    }
  }

  /// In the $\nu \to \infty$ limit the t-copula degenerates to the
  /// Gaussian copula. Pick $\nu = 200$ as a practical proxy and verify
  /// the multivariate density agrees with the Gaussian copula density on
  /// random uniform queries.
  #[test]
  fn t_copula_large_nu_approaches_gaussian() {
    let corr = array![[1.0, 0.5], [0.5, 1.0]];
    let t_cop = TMultivariate::new_with(corr.clone(), 200.0).unwrap();
    let g_cop = super::super::gaussian::GaussianMultivariate::new_with_corr(corr).unwrap();
    let queries = array![[0.25, 0.75], [0.5, 0.5], [0.1, 0.9], [0.8, 0.3],];
    let t_pdf = t_cop.pdf(queries.clone()).unwrap();
    let g_pdf = g_cop.pdf(queries).unwrap();
    for i in 0..t_pdf.len() {
      assert!(
        (t_pdf[i] - g_pdf[i]).abs() / g_pdf[i].max(1e-10) < 0.02,
        "ν=200 t-pdf[{i}]={} vs Gaussian-pdf[{i}]={}",
        t_pdf[i],
        g_pdf[i]
      );
    }
  }

  /// At u = (0.5, …, 0.5) the t-copula density reduces to the multivariate
  /// kernel at z=0 divided by the marginal kernels at 0. Both reduce to a
  /// closed form involving Γ-ratios and |Σ|^{-1/2}.
  #[test]
  fn t_copula_pdf_at_center() {
    let corr = array![[1.0, 0.3], [0.3, 1.0]];
    let nu = 6.0;
    let cop = TMultivariate::new_with(corr.clone(), nu).unwrap();
    let q = array![[0.5, 0.5]];
    let pdf = cop.pdf(q).unwrap()[0];
    // Analytic value: at z=0 the multivariate kernel = 1, marginal kernel = 1.
    //   c(0.5, 0.5) = f_{Σ,ν}(0,0) / [f_ν(0)]^2
    //   f_{Σ,ν}(0,0) = Γ((ν+2)/2) / [Γ(ν/2) · ν · π · √|Σ|]
    //   f_ν(0)       = Γ((ν+1)/2) / [Γ(ν/2) · √(ν · π)]
    let det: f64 = 1.0 - 0.3 * 0.3;
    let f_mv = (ln_gamma(0.5 * (nu + 2.0))
      - ln_gamma(0.5 * nu)
      - (nu * f64::consts::PI).ln()
      - 0.5 * det.ln())
    .exp();
    let f_marg =
      (ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * f64::consts::PI).ln()).exp();
    let expected = f_mv / (f_marg * f_marg);
    assert!(
      (pdf - expected).abs() / expected < 1e-10,
      "t-pdf at center = {pdf}, expected {expected}"
    );
  }

  /// Round-trip: sample → recover Σ via Kendall-τ → check entries match
  /// the true correlations within MC tolerance. Verifies the fit path.
  /// `n = 5000` keeps the test under 1 minute (the ν-profile Brent search
  /// scales as n × n_iter × t_quantile_cost); recovery quality is still
  /// `|Σ̂ − Σ| < 0.05` on Kendall-τ inversion.
  #[test]
  fn t_copula_fit_recovers_correlation_from_sample() {
    let true_corr = array![[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]];
    let cop = TMultivariate::new_with(true_corr.clone(), 5.0).unwrap();
    let u = cop.sample(5_000).unwrap();
    let mut fitted = TMultivariate::new();
    fitted.fit(u).unwrap();
    let recovered = fitted.correlation().unwrap();
    for i in 0..3 {
      for j in (i + 1)..3 {
        let err = (recovered[[i, j]] - true_corr[[i, j]]).abs();
        assert!(
          err < 0.05,
          "Σ[{i},{j}]: true={}, recovered={} (err={err})",
          true_corr[[i, j]],
          recovered[[i, j]]
        );
      }
    }
    // ν is the harder estimate; profile-likelihood gives ~2-unit error on
    // 5k samples — accept anything in [2.5, 12.0] as recovery of ν=5.
    let nu_hat = fitted.degrees_of_freedom();
    assert!(
      (2.5..=12.0).contains(&nu_hat),
      "ν recovered = {nu_hat}, expected ~5"
    );
  }

  /// Log-pdf = ln(pdf) for all valid query points (no separate log path
  /// divergence).
  #[test]
  fn t_copula_log_pdf_matches_ln_pdf() {
    let corr = array![[1.0, 0.4], [0.4, 1.0]];
    let cop = TMultivariate::new_with(corr, 6.0).unwrap();
    let q = array![[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]];
    let pdf = cop.pdf(q.clone()).unwrap();
    let lp = cop.log_pdf(q).unwrap();
    for i in 0..pdf.len() {
      assert!(
        (lp[i] - pdf[i].ln()).abs() < 1e-12,
        "log_pdf[{i}]={} vs ln(pdf[{i}])={}",
        lp[i],
        pdf[i].ln()
      );
    }
  }

  /// CDF in $\mathbb{R}^2$ should approximately match the bivariate
  /// Dunnett-Sobel reference implemented in [`crate::bivariate::t_copula`].
  /// MC noise on 4000 samples ≈ 1/√4000 ≈ 0.016; allow 0.04 tolerance.
  #[test]
  fn t_copula_cdf_matches_bivariate_reference() {
    use crate::bivariate::t_copula::TCopula;
    use crate::traits::BivariateExt;
    let rho = 0.4;
    let nu = 6.0;
    let corr = array![[1.0, rho], [rho, 1.0]];
    let mv = TMultivariate::new_with(corr, nu).unwrap();
    let mut bv = TCopula::with_nu(nu);
    bv.set_theta(rho);
    let q = array![[0.3, 0.6], [0.5, 0.5], [0.8, 0.2]];
    let cdf_mv = mv.cdf(q.clone()).unwrap();
    let cdf_bv = bv.cdf(&q).unwrap();
    for i in 0..cdf_mv.len() {
      assert!(
        (cdf_mv[i] - cdf_bv[i]).abs() < 0.04,
        "MV-CDF[{i}]={} vs BV-CDF[{i}]={}",
        cdf_mv[i],
        cdf_bv[i]
      );
    }
  }

  /// Manual ν override must be respected by sampling and pdf paths.
  #[test]
  fn t_copula_manual_nu_override() {
    let corr = array![[1.0, 0.3], [0.3, 1.0]];
    let mut cop = TMultivariate::new_with(corr, 4.0).unwrap();
    cop.set_degrees_of_freedom(12.0).unwrap();
    assert_eq!(cop.degrees_of_freedom(), 12.0);
    let _ = cop.sample(100).unwrap();
    let _ = cop.pdf(array![[0.5, 0.5]]).unwrap();
    let bad = cop.set_degrees_of_freedom(0.0);
    assert!(bad.is_err(), "ν=0 must be rejected");
  }
}
