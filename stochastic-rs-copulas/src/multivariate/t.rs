//! # Multivariate Student-t copula
//!
//! $$
//! C_{\Sigma,\nu}(u) = t_{\Sigma,\nu}\!\big(t_\nu^{-1}(u_1), \dots, t_\nu^{-1}(u_d)\big),
//! $$
//!
//! where $t_{\Sigma,\nu}$ is the $d$-variate t cdf of correlation $\Sigma$ and $t_\nu^{-1}$ the
//! univariate quantile, $z_j = t_\nu^{-1}(u_j)$; the density is
//!
//! $$
//! c_{\Sigma,\nu}(u) = \frac{f_{\Sigma,\nu}(z)}{\prod_{j=1}^{d} f_\nu(z_j)},
//! \qquad
//! f_{\Sigma,\nu}(z) = \frac{\Gamma\!\big(\tfrac{\nu+d}{2}\big)}
//! {\Gamma\!\big(\tfrac{\nu}{2}\big)\,(\nu\pi)^{d/2}\,|\Sigma|^{1/2}}
//! \left(1 + \tfrac{1}{\nu}\,z^\top \Sigma^{-1} z\right)^{-\tfrac{\nu+d}{2}}.
//! $$
//!
//! Each pair's tail dependence, upper and lower, is positive for finite $\nu$:
//!
//! $$
//! \lambda = 2\,t_{\nu+1}\big(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big).
//! $$
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
//! - Brent, R.P. (1973), *Algorithms for Minimization without Derivatives*, Prentice-Hall, ch. 5 (`localmin`).

use std::sync::OnceLock;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;

use super::MultivariateFamily;
use super::correlation_matrix::CorrelationMatrix;
use super::correlation_matrix::shrink_to_spd;
use super::genz::chi_scale;
use super::genz::mvt_upper;
use super::genz::open_coordinates;
use super::genz::prioritize;
use super::qmc::sobol_points;
use crate::bivariate::conditional::check_open;
use crate::bivariate::conditional::check_unit;
use crate::bivariate::conditional::softplus;
use crate::bivariate::t_copula::cdf::BivariateT;
use crate::correlation::check_defined;
use crate::correlation::kendall_tau;
use crate::error::CopulaError;
use crate::multivariate::inside;
use crate::multivariate::on_open_cube;
use crate::optim::localmin;
use crate::student_t;
use crate::student_t::Law;
use crate::student_t::Score;
use crate::student_t::check_nu;
use crate::student_t::ln_gamma_constant;
use crate::traits::MultivariateExt;

/// Multivariate Student-$t$ copula with correlation $\Sigma$ and degrees of freedom $\nu$; as $\nu \to \infty$ it
/// collapses to the Gaussian copula, and for finite $\nu$ every pair has symmetric tail dependence.
#[derive(Debug, Clone)]
pub struct TMultivariate {
  /// Degrees of freedom, `0 < nu < inf`; 4 by default.
  nu: f64,
  corr: Option<CorrelationMatrix>,
  /// The cdf's χ scales, a function of `nu` alone, built by the first query that integrates.
  chi: OnceLock<Array1<f64>>,
}

impl Default for TMultivariate {
  fn default() -> Self {
    Self {
      nu: 4.0,
      corr: None,
      chi: OnceLock::new(),
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
      chi: OnceLock::new(),
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
    self.chi = OnceLock::new();
  }

  fn fitted(&self) -> Result<&CorrelationMatrix, CopulaError> {
    self.corr.as_ref().ok_or(CopulaError::NotFitted)
  }

  /// $\Sigma$ from the sample Kendall τ-b matrix through $\rho_{ij} = \sin(\pi\tau_{ij}/2)$, which holds for any elliptical
  /// copula and so does not depend on $\nu$.
  fn estimate_corr_from_kendall(x: ArrayView2<f64>) -> Result<Array2<f64>, CopulaError> {
    let d = x.ncols();
    let tau = kendall_tau(x)?;
    check_defined(tau.view())?;
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
    Ok(corr)
  }

  /// Log-likelihood of the data under `corr` at `nu`, the objective `optimise_nu` maximises with Σ held fixed.
  fn log_likelihood_for_nu(corr: &CorrelationMatrix, x: ArrayView2<f64>, nu: f64) -> f64 {
    LogDensity::new(corr, nu).rows(x).sum()
  }

  /// Brent's `localmin` on [2.1, 50] to 1e-6 (`eps = t = 1e-6`), the resolution the likelihood's rounding leaves in ν; a
  /// likelihood that is not finite scores `+∞`, which the search steps away from.
  fn optimise_nu(corr: &CorrelationMatrix, x: ArrayView2<f64>) -> f64 {
    let nll = |nu: f64| {
      let ll = Self::log_likelihood_for_nu(corr, x, nu);
      if ll.is_finite() { -ll } else { f64::INFINITY }
    };
    localmin(2.1, 50.0, 1e-6, 1e-6, nll).0
  }
}

impl MultivariateExt for TMultivariate {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::TMultivariate
  }

  /// Demarta & McNeil (2005, §2.3): $X = Z/\sqrt{W}$, $U_j = t_\nu(X_j)$; the normal and $\chi^2$
  /// streams both come from `seed`, which advances between them.
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
        u[[r, c]] = inside(student_t::cdf(xc, self.nu));
      }
    }
    Ok(u)
  }

  /// Σ from Kendall's τ, shrunk until every eigenvalue is at least `1e-3`; ν by profile likelihood, Brent's `localmin`
  /// on `[2.1, 50]` to a relative `1e-6`.
  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    if x.ncols() < 2 {
      return Err(CopulaError::InvalidStructure(format!(
        "a t-copula needs dim >= 2, got {}",
        x.ncols()
      )));
    }
    if x.nrows() < 2 {
      return Err(CopulaError::InsufficientData {
        needed: 2,
        got: x.nrows(),
      });
    }
    check_unit(x)?;
    check_open(x)?;
    let corr = CorrelationMatrix::try_new(shrink_to_spd(Self::estimate_corr_from_kendall(x)?))?;
    // Profile MLE for ν conditional on Σ.
    self.nu = Self::optimise_nu(&corr, x);
    self.chi = OnceLock::new();
    self.corr = Some(corr);
    Ok(())
  }

  fn check_fit(&self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    let dim = self.fitted()?.dim();
    if x.ncols() != dim {
      return Err(CopulaError::DimensionMismatch {
        expected: dim,
        got: x.ncols(),
      });
    }
    check_unit(x)
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.log_pdf(x)?.mapv(f64::exp))
  }

  /// `LogDensity::rows` on the open cube.
  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(x)?;
    let out = LogDensity::new(self.fitted()?, self.nu).rows(x);
    Ok(on_open_cube(x, out))
  }

  /// `T_{R,ν}(t_ν⁻¹(u))` on the closed cube: a zero coordinate gives 0 and a unit one drops out; two left are the
  /// bivariate closed form, more are Genz–Bretz over the Sobol points in that query's prioritized order.
  fn cdf(&self, uv: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(uv)?;
    let corr = self.fitted()?;
    let mut points = None;
    let d = corr.dim();
    // One bivariate law per pair of free coordinates, built on the first row that needs it.
    let mut pairs: Vec<Option<BivariateT>> = vec![None; d * d];
    Ok(
      uv.rows()
        .into_iter()
        .map(|u| match open_coordinates(u).as_deref() {
          None => 0.0,
          Some([]) => 1.0,
          Some(&[i]) => u[i],
          Some(&[i, j]) => pairs[i * d + j]
            .get_or_insert_with(|| BivariateT::new(corr.corr()[[i, j]], self.nu))
            .copula_cdf(u[i], u[j]),
          Some(open) => {
            let points = points.get_or_insert_with(|| sobol_points(corr.dim()));
            let scale = self.chi.get_or_init(|| chi_scale(self.nu));
            let b: Vec<f64> = open
              .iter()
              .map(|&i| student_t::quantile(u[i], self.nu))
              .collect();
            match prioritize(corr.corr(), open, &b) {
              Some((l, b)) => mvt_upper(&l, &b, points.view(), scale.view()),
              None => {
                let b: Vec<f64> = u.iter().map(|&x| student_t::quantile(x, self.nu)).collect();
                mvt_upper(corr.chol_lower(), &b, points.view(), scale.view())
              }
            }
          }
        })
        .collect(),
    )
  }
}

/// The t copula's log-density at one `(R, ν)`, with what no point changes formed once:
/// `ln c = C_d − ½ ln|R| − ((ν + d)/2) ln(1 + zᵀR⁻¹z/ν) + ((ν + 1)/2) Σ ln(1 + z_j²/ν)` at the law's scores `z_j`.
struct LogDensity<'a> {
  law: Law,
  inverse: &'a Array2<f64>,
  nu: f64,
  ln_nu: f64,
  /// `C_d − ½ ln|R|`, `C_d = ⌊d/2⌋G + Σ_{k<⌊d/2⌋} ln(1 + (2k + d mod 2)/ν)` from `G = student_t::ln_gamma_constant(ν)`,
  /// the gammas' logarithms cancelled in closed form.
  constant: f64,
}

impl<'a> LogDensity<'a> {
  fn new(corr: &'a CorrelationMatrix, nu: f64) -> Self {
    let d = corr.dim();
    let (pairs, odd) = (d / 2, (d % 2) as f64);
    let rising: f64 = (0..pairs)
      .map(|k| ((2 * k) as f64 + odd) / nu)
      .map(f64::ln_1p)
      .sum();
    Self {
      law: Law::new(nu),
      inverse: corr.inverse(),
      nu,
      ln_nu: nu.ln(),
      constant: pairs as f64 * ln_gamma_constant(nu) + rising - 0.5 * corr.log_det(),
    }
  }

  /// `ln c` of every row of `x`, at the scores the law forms from its body quantiles, with one buffer for all rows.
  fn rows(&self, x: ArrayView2<f64>) -> Array1<f64> {
    let body = self.law.body_quantiles(x);
    let scores: Vec<Score> = x
      .iter()
      .zip(body.iter())
      .map(|(&p, &t)| self.law.score(p, t))
      .collect();
    let mut w = vec![0.0; x.ncols()];
    scores
      .chunks(x.ncols())
      .map(|row| self.at(row, &mut w))
      .collect()
  }

  /// `ln c` at one row's scores: in the scores where they and `zᵀR⁻¹z` stay finite, else in `ln|z_j|`, scaled by the
  /// largest, so a score past `f64::MAX` keeps its terms.
  fn at(&self, scores: &[Score], w: &mut [f64]) -> f64 {
    for (w, s) in w.iter_mut().zip(scores) {
      *w = s.value;
    }
    let quadratic = self.quadratic(w);
    let (joint, margins) = if quadratic.is_finite() && w.iter().all(|x| (x * x).is_finite()) {
      let margins: f64 = w.iter().map(|&x| (x * x / self.nu).ln_1p()).sum();
      ((quadratic / self.nu).ln_1p(), margins)
    } else {
      let top = scores
        .iter()
        .fold(f64::NEG_INFINITY, |a, s| a.max_or_nan(s.ln_abs()));
      for (w, s) in w.iter_mut().zip(scores.iter()) {
        *w = s.sign() * (s.ln_abs() - top).exp();
      }
      let ln_quadratic = 2.0 * top + self.quadratic(w).ln();
      let margins: f64 = scores
        .iter()
        .map(|s| softplus(2.0 * s.ln_abs() - self.ln_nu))
        .sum();
      (softplus(ln_quadratic - self.ln_nu), margins)
    };
    let d = scores.len() as f64;
    self.constant - 0.5 * (self.nu + d) * joint + 0.5 * (self.nu + 1.0) * margins
  }

  /// `wᵀR⁻¹w`.
  fn quadratic(&self, w: &[f64]) -> f64 {
    w.iter()
      .enumerate()
      .map(|(i, &a)| {
        a * w
          .iter()
          .enumerate()
          .map(|(j, &b)| self.inverse[[i, j]] * b)
          .sum::<f64>()
      })
      .sum()
  }
}

#[cfg(test)]
mod tests;
