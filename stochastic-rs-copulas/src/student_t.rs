//! The Student-t law every t copula and the t pair copula evaluate through: `SimdStudentT`'s cdf and quantile, the
//! log-density its `pdf` exponentiates, the t copula's log-density at t scores, and the one rule for `nu`.

use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::studentt::SimdStudentT;

use crate::error::CopulaError;

/// `0 < nu < ∞` and `nu / 2 > 0`, which only `5e-324` fails: what `SimdStudentT::new` and `SimdChiSquared::new` assert, so
/// the t copulas and the vines' t pairs refuse up front a `nu` those laws would panic on.
pub(crate) fn check_nu(nu: f64) -> Result<(), CopulaError> {
  let constraint = if !nu.is_finite() || nu <= 0.0 {
    "0.0 < nu < inf"
  } else if nu * 0.5 == 0.0 {
    "nu / 2 > 0"
  } else {
    return Ok(());
  };
  Err(CopulaError::InvalidParameter {
    name: "nu",
    value: nu,
    constraint: constraint.into(),
  })
}

pub(crate) fn cdf(x: f64, nu: f64) -> f64 {
  SimdStudentT::new(nu)
    .cdf(x)
    .expect("the Student-t law has a cdf")
}

/// `∓∞` at `p = 0` and `p = 1`; inlined, so a loop over `p` at one `nu` hoists the work that depends on `nu` alone.
#[inline(always)]
pub(crate) fn quantile(p: f64, nu: f64) -> f64 {
  SimdStudentT::new(nu)
    .quantile(p)
    .expect("the Student-t law has a quantile")
}

/// The exponent `SimdStudentT::pdf` takes, so `ln_pdf(x, nu).exp()` is that pdf bit for bit.
pub(crate) fn ln_pdf(x: f64, nu: f64) -> f64 {
  let log_norm =
    ln_gamma(0.5 * (nu + 1.0)) - 0.5 * (nu * std::f64::consts::PI).ln() - ln_gamma(0.5 * nu);
  let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
  log_norm + log_kernel
}

/// `log c` of the t copula at the scores `x = t_ν⁻¹(u)`, `y = t_ν⁻¹(v)`: `c = Γ((ν+2)/2) Γ(ν/2) / (Γ((ν+1)/2)² √(1−ρ²))`
/// `· (1 + (x² − 2ρxy + y²)/(ν(1−ρ²)))^{−(ν+2)/2} · ((1 + x²/ν)(1 + y²/ν))^{(ν+1)/2}`.
pub(crate) fn ln_copula_density(x: f64, y: f64, rho: f64, nu: f64) -> f64 {
  let one_minus_rho2 = 1.0 - rho * rho;
  let log_norm = ln_gamma(0.5 * (nu + 2.0)) + ln_gamma(0.5 * nu)
    - 2.0 * ln_gamma(0.5 * (nu + 1.0))
    - 0.5 * one_minus_rho2.ln();
  let log_joint =
    -0.5 * (nu + 2.0) * (1.0 + (x * x - 2.0 * rho * x * y + y * y) / (nu * one_minus_rho2)).ln();
  let log_marg = 0.5 * (nu + 1.0) * ((1.0 + x * x / nu).ln() + (1.0 + y * y / nu).ln());
  log_norm + log_joint + log_marg
}
