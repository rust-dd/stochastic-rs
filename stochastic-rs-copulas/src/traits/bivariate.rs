//! `BivariateExt` — bivariate copula trait.

use std::cell::RefCell;
use std::cmp::Ordering;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::stack;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::ABSOLUTE_TOLERANCE;
use crate::bivariate::conditional::check_lengths;
use crate::bivariate::conditional::check_open;
use crate::bivariate::conditional::check_pairs;
use crate::bivariate::conditional::check_unit;
use crate::bivariate::conditional::inside;
use crate::correlation::kendall_tau_b;
use crate::error::CopulaError;
use crate::optim::zero;

/// Upper- and lower-tail dependence $\lambda_L = \lim_{u\to0^+} C(u,u)/u$ and
/// $\lambda_U = \lim_{u\to1^-} (1 - 2u + C(u,u))/(1 - u)$; every [`BivariateExt`] impl in this crate is `f64`-based.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailDependence<T> {
  pub lower: T,
  pub upper: T,
}

/// Every batch method, `generator` too, answers `DimensionMismatch` off two columns, then `MarginalOutOfRange` outside
/// `[0, 1]` or for NaN; densities are NaN on the edges, and `fit` answers `MarginalOnBoundary` for a 0 or 1.
pub trait BivariateExt {
  /// Which family this copula is; the label its error messages carry.
  fn family(&self) -> BivariateFamily;

  fn theta(&self) -> Option<f64>;

  /// Stores `theta`; panics unless `validate_theta` admits it.
  fn set_theta(&mut self, theta: f64);

  fn theta_bounds(&self) -> (f64, f64);

  fn invalid_thetas(&self) -> Vec<f64>;

  /// A finite `theta` inside the family's bounds and off its exclusions: the one check every entry point runs.
  fn validate_theta(&self, theta: f64) -> Result<(), CopulaError> {
    let (lower, upper) = self.theta_bounds();
    let invalid = self.invalid_thetas();
    if !(theta.is_finite() && lower <= theta && theta <= upper) || invalid.contains(&theta) {
      let below = if lower.is_finite() { "<=" } else { "<" };
      let above = if upper.is_finite() { "<=" } else { "<" };
      let bounds = format!("{lower:?} {below} theta {above} {upper:?}");
      let constraint = if invalid.is_empty() {
        bounds
      } else {
        format!("{bounds}, theta not in {invalid:?}")
      };
      return Err(CopulaError::InvalidParameter {
        name: "theta",
        value: theta,
        constraint,
      });
    }
    Ok(())
  }

  fn check_theta(&self) -> Result<(), CopulaError> {
    let theta = self.theta().ok_or(CopulaError::NotFitted)?;
    self.validate_theta(theta)
  }

  /// Kendall's τ of the family at `theta`: a closed form, or quadrature where there is none.
  fn tau_from_theta(&self, theta: f64) -> f64;

  /// The `theta` whose Kendall's τ is `tau`, always one `validate_theta` admits; `InvalidParameter { name: "tau", .. }`
  /// outside the reachable range.
  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError>;

  /// Kendall's τ implied by the current `theta`.
  fn tau(&self) -> Option<f64> {
    self.theta().map(|theta| self.tau_from_theta(theta))
  }

  /// Upper and lower tail-dependence coefficients at the current parameters; `NotFitted` before they are set,
  /// `InvalidParameter` outside the family's range.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError>;

  /// Archimedean generator $\varphi$, $C(u,v) = \varphi^{-1}(\varphi(u) + \varphi(v))$: `MarginalOutOfRange` outside
  /// `[0, 1]` or for NaN, then the family's own, or `Unsupported`, named by `family()`, for a family with none.
  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    check_unit(t)?;
    self.generator_unchecked(t).unwrap_or_else(|| {
      Err(CopulaError::Unsupported(format!(
        "{:?} is not Archimedean: no generator",
        self.family()
      )))
    })
  }

  /// A family's own generator on inputs in `[0, 1]`; `None` by default, a family with none.
  #[doc(hidden)]
  fn generator_unchecked(&self, _t: ArrayView1<f64>) -> Option<Result<Array1<f64>, CopulaError>> {
    None
  }

  /// `n` pairs by conditional inversion from `seed`, each coordinate in `(0, 1)` (one rounded to 0 or 1 moves one ulp
  /// inside): a fresh `Deterministic` replays them, a shared one advances; generic, so not on a `dyn BivariateExt`.
  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<ndarray::Array2<f64>, CopulaError>
  where
    Self: Sized,
  {
    self.sample_with_uniform(SimdUniform::<f64>::new(0.0, 1.0).seeded(seed), n)
  }

  #[doc(hidden)]
  fn sample_with_uniform(
    &self,
    mut ud: Seeded<SimdUniform<f64>>,
    n: usize,
  ) -> Result<ndarray::Array2<f64>, CopulaError> {
    // The family's `check_fit` is the gate, so a copula parameterised outside `theta` passes it.
    self.check_fit()?;

    let mut v = Array1::<f64>::zeros(n);
    ud.fill_slice(v.as_slice_mut().unwrap());
    let mut c = Array1::<f64>::zeros(n);
    ud.fill_slice(c.as_slice_mut().unwrap());
    // The uniforms are its own, in `[0, 1]` and of equal length: nothing for a family's own inverse to check again.
    let u = match self.percent_point_unchecked(c.view(), v.view()) {
      Some(u) => u?,
      None => self.percent_point(c.view(), v.view())?,
    };

    Ok(stack![Axis(1), u.mapv(inside), v.mapv(inside)])
  }

  /// Kendall's τ-b of a two-column sample of pseudo-observations, inverted through `theta_from_tau`; nothing is stored
  /// when the family cannot represent it (`InvalidParameter { name: "tau", .. }`) or the inverse fails `validate_theta`.
  fn fit(&mut self, x: ndarray::ArrayView2<f64>) -> Result<(), CopulaError> {
    check_pairs(x)?;
    check_open(x)?;
    let tau = kendall_tau_b(x.column(0), x.column(1))?;
    let theta = self.theta_from_tau(tau)?;
    self.validate_theta(theta)?;
    self.set_theta(theta);
    Ok(())
  }

  /// Sampling, evaluation and `tail_dependence` gate on this, so an override must validate the parameters it holds.
  fn check_fit(&self) -> Result<(), CopulaError> {
    if self.theta().is_none() {
      return Err(CopulaError::NotFitted);
    }

    self.check_theta()?;
    Ok(())
  }

  /// Kolmogorov–Smirnov uniformity test of one margin at the bound `1.627/√n` (nominally 1 %), so a valid sample fails it
  /// by chance: a diagnostic, which `fit` does not run.
  fn check_marginal(&self, u: ArrayView1<f64>) -> Result<(), CopulaError> {
    check_unit(u)?;

    let mut empirical_cdf = u.to_vec();
    empirical_cdf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let empirical_cdf = Array1::from(empirical_cdf);
    let uniform = Array1::linspace(0.0, 1.0, u.len());
    let ks = (empirical_cdf - uniform).fold(0.0_f64, |acc, &d| acc.max_or_nan(d.abs()));

    if ks > 1.627 / (u.len() as f64).sqrt() {
      return Err(CopulaError::MarginalNotUniform);
    }

    Ok(())
  }

  fn pdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;

  fn log_pdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.pdf(x)?.ln())
  }

  fn cdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;

  /// Inverse conditional: the `u` with `P(U ≤ u | V = v) = y`, its parameters and inputs checked once;
  /// `percent_point_numerical` by default, which a family with its own inverse overrides.
  #[doc(alias = "ppf")]
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.percent_point_numerical(y, v)
  }

  /// A family's own inverse on inputs already checked, as the sampler's uniforms are; `None` by default, so the sampler
  /// takes `percent_point` and an override of `percent_point` that calls this cannot recurse.
  #[doc(hidden)]
  fn percent_point_unchecked(
    &self,
    _y: ArrayView1<f64>,
    _v: ArrayView1<f64>,
  ) -> Option<Result<Array1<f64>, CopulaError>> {
    None
  }

  /// `percent_point`'s default, callable from an override: Brent's `zero` in `u` of `partial_derivative` held in
  /// `[0, 1]`, on `[0, 1]`, saturating at its ends; `Err` for a NaN `partial_derivative` or its error.
  fn percent_point_numerical(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    check_lengths(y, v)?;
    check_unit(y.iter().chain(v))?;
    numerical_inverse(self, y, v)
  }

  /// `h(u | v) = ∂_v C(u, v)`, the conditional cdf of `U` given `V = v`, over the rows `(u, v)` of `x`: what
  /// `percent_point` inverts in `u` and the samplers draw through; required, as no difference of `cdf` is exact.
  fn partial_derivative(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;

  #[doc(hidden)]
  fn partial_derivative_scalar(&self, u: f64, v: f64) -> Result<f64, CopulaError> {
    self.check_fit()?;
    let x = stack![Axis(1), Array1::from(vec![u]), Array1::from(vec![v])];
    let out = self.partial_derivative(x.view());

    Ok(*out?.get(0).unwrap())
  }
}

/// Brent's `zero` on `residual(θ) = τ(θ) − τ` over `[lo, hi]` to the absolute `t` the family's τ rule resolves;
/// `Numerical` without a sign change or past its bound.
pub(crate) fn invert_tau(
  residual: impl FnMut(f64) -> f64,
  lo: f64,
  hi: f64,
  t: f64,
) -> Result<f64, CopulaError> {
  zero(lo, hi, t, residual).ok_or_else(|| {
    CopulaError::Numerical(format!("theta from tau has no root on [{lo:?}, {hi:?}]"))
  })
}

/// `percent_point` of a family with its own `inverse`: its parameters and inputs checked once, then that inverse.
pub(crate) fn checked_percent_point<C: BivariateExt + ?Sized>(
  copula: &C,
  y: ArrayView1<f64>,
  v: ArrayView1<f64>,
  inverse: impl FnOnce(&C, ArrayView1<f64>, ArrayView1<f64>) -> Result<Array1<f64>, CopulaError>,
) -> Result<Array1<f64>, CopulaError> {
  copula.check_fit()?;
  check_lengths(y, v)?;
  check_unit(y.iter().chain(v))?;
  inverse(copula, y, v)
}

/// Brent's `zero` in `u` of `partial_derivative` held in `[0, 1]`, on `[0, 1]`, for checked `(y, v)`: the body of
/// `percent_point_numerical`.
fn numerical_inverse<C: BivariateExt + ?Sized>(
  copula: &C,
  y: ArrayView1<f64>,
  v: ArrayView1<f64>,
) -> Result<Array1<f64>, CopulaError> {
  let n = y.len();
  let mut results = Array1::zeros(n);
  let lo = 0.0;

  for i in 0..n {
    let y_i = y[i];
    let v_i = v[i];
    if y_i == 0.0 {
      results[i] = lo;
      continue;
    }

    // The solver's closure cannot return a `Result`: the first inner failure or NaN is parked, the closure answers
    // 0 so the solve stops there, and the parked error is re-raised after.
    let inner_err: RefCell<Option<CopulaError>> = RefCell::new(None);
    let gap = |u: f64| match copula.partial_derivative_scalar(u, v_i) {
      Ok(d) if !d.is_nan() => d.clamp(0.0, 1.0) - y_i,
      outcome => {
        let mut slot = inner_err.borrow_mut();
        if slot.is_none() {
          *slot = Some(outcome.err().unwrap_or_else(|| {
            CopulaError::Numerical(format!(
              "{:?} partial_derivative is NaN at (u={u:?}, v={v_i:?})",
              copula.family()
            ))
          }));
        }
        0.0
      }
    };

    let f_lo = gap(lo);
    if let Some(e) = inner_err.borrow_mut().take() {
      return Err(e);
    }
    if f_lo >= 0.0 {
      results[i] = lo;
      continue;
    }
    let f_hi = gap(1.0);
    if let Some(e) = inner_err.borrow_mut().take() {
      return Err(e);
    }
    if f_hi <= 0.0 {
      results[i] = 1.0;
      continue;
    }

    let root = zero(lo, 1.0, ABSOLUTE_TOLERANCE, &gap);
    if let Some(e) = inner_err.borrow_mut().take() {
      return Err(e);
    }
    results[i] = root.ok_or_else(|| {
      CopulaError::Numerical(format!(
        "{:?} h-inverse did not converge at (y={y_i:?}, v={v_i:?})",
        copula.family()
      ))
    })?;
  }

  Ok(results)
}

/// `theta` when the family admits it; an inverse that rounds onto an excluded value is the `tau` error, which names it.
pub(crate) fn admitted<C: BivariateExt + ?Sized>(
  copula: &C,
  tau: f64,
  theta: f64,
) -> Result<f64, CopulaError> {
  match copula.validate_theta(theta) {
    Ok(()) => Ok(theta),
    Err(CopulaError::InvalidParameter { constraint, .. }) => Err(tau_out_of_range(
      tau,
      format!("an inverse theta (here {theta:?}) with {constraint}"),
    )),
    Err(e) => Err(e),
  }
}

pub(crate) fn tau_out_of_range(tau: f64, constraint: String) -> CopulaError {
  CopulaError::InvalidParameter {
    name: "tau",
    value: tau,
    constraint,
  }
}

#[cfg(test)]
mod tests;
