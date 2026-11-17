use stochastic_rs_distributions::brent;
use stochastic_rs_distributions::special::norm_cdf;

use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::sabr::SabrParams;

/// Forward FX F = S * exp((r_d - r_f) T)
pub fn forward_fx(s: f64, tau: f64, r_d: f64, r_f: f64) -> f64 {
  s * ((r_d - r_f) * tau).exp()
}

/// Hagan et al. (2002), "Managing Smile Risk", Wilmott Magazine, 84–108, eq. A.69a (general β).
///
/// $$
/// \sigma(K)\approx
/// \frac{\alpha}{(FK)^{(1-\beta)/2}\!\bigl[1+\tfrac{(1-\beta)^2}{24}\ln^2\!\tfrac FK
/// +\tfrac{(1-\beta)^4}{1920}\ln^4\!\tfrac FK\bigr]}
/// \;\frac{z}{x(z)}
/// \;\Bigl[1+\Bigl(\tfrac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}}
/// +\tfrac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}}
/// +\tfrac{(2-3\rho^2)\nu^2}{24}\Bigr)\tau\Bigr]
/// $$
///
/// NaN for `k ≤ 0`, `f ≤ 0` or a negative or non-finite `tau` (`tau = 0` is the leading order);
/// panics unless `alpha` is finite and positive and `rho ∈ (−1, 1)` (`x(z)` divides by `1 − ρ`).
pub fn hagan_implied_vol(k: f64, f: f64, tau: f64, params: &SabrParams) -> f64 {
  let SabrParams {
    alpha,
    beta,
    rho,
    nu,
  } = *params;
  assert!(
    alpha.is_finite() && alpha > 0.0,
    "alpha must satisfy `alpha.is_finite() && alpha > 0.0`, got alpha = {alpha:?}"
  );
  assert!(
    rho > -1.0 && rho < 1.0,
    "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = {rho:?}"
  );
  if !(k > 0.0 && f > 0.0 && tau.is_finite() && tau >= 0.0) {
    return f64::NAN;
  }

  let eps = 1e-07;
  let logfk = (f / k).ln();
  let fkbeta = (f * k).powf(1.0 - beta);
  let a = (1.0 - beta).powi(2) * alpha * alpha / (24.0 * fkbeta);
  let b = 0.25 * rho * beta * nu * alpha / fkbeta.sqrt();
  let c = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
  let d = fkbeta.sqrt();
  let v = (1.0 - beta).powi(2) * logfk * logfk / 24.0;
  let w = (1.0 - beta).powi(4) * logfk.powi(4) / 1920.0;
  let z = nu * fkbeta.sqrt() * logfk / alpha;

  if z.abs() > eps {
    let arg = (1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho;
    let xz = (arg / (1.0 - rho)).ln();
    if xz.abs() < 1e-14 {
      return alpha * (1.0 + (a + b + c) * tau) / (d * (1.0 + v + w));
    }
    alpha * z * (1.0 + (a + b + c) * tau) / (d * (1.0 + v + w) * xz)
  } else {
    alpha * (1.0 + (a + b + c) * tau) / (d * (1.0 + v + w))
  }
}

/// Hagan et al. (2002) eq. A.69b: α, the cubic's smallest positive root if it reprices `v_atm` to a
/// relative `√ε`, else NaN as at `f ≤ 0`, `v_atm ≤ 0` or `tau ∉ [0, ∞)`; panics unless `|rho| < 1`.
///
/// $$
/// \frac{(1-\beta)^2 f_*^3}{24}\,\tau\;\alpha^3
/// +\frac{\rho\beta\nu\,f_*^2}{4}\,\tau\;\alpha^2
/// +\bigl(1+\tfrac{(2-3\rho^2)\nu^2}{24}\,\tau\bigr)\,f_*\;\alpha
/// -\sigma_{ATM}=0, \quad f_*=F^{\beta-1}
/// $$
pub fn alpha_from_atm_vol(v_atm: f64, f: f64, tau: f64, beta: f64, rho: f64, nu: f64) -> f64 {
  assert!(
    rho > -1.0 && rho < 1.0,
    "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = {rho:?}"
  );
  if !(f > 0.0 && v_atm > 0.0 && tau.is_finite() && tau >= 0.0) {
    return f64::NAN;
  }

  let f_ = f.powf(beta - 1.0);
  let p3 = tau * f_.powi(3) * (1.0 - beta).powi(2) / 24.0;
  let p2 = tau * f_.powi(2) * rho * beta * nu / 4.0;
  let p1 = (1.0 + tau * nu * nu * (2.0 - 3.0 * rho * rho) / 24.0) * f_;
  let alpha = smallest_positive_root([p3, p2, p1, -v_atm], v_atm * f.powf(1.0 - beta));
  if !(alpha.is_finite() && alpha > 0.0) {
    return f64::NAN;
  }
  // Far above α₀ the cubic's terms cancel to σ_ATM only within rounding, so its root may miss it.
  let params = SabrParams {
    alpha,
    beta,
    rho,
    nu,
  };
  let atm = hagan_implied_vol(f, f, tau, &params);
  if (atm - v_atm).abs() <= f64::EPSILON.sqrt() * v_atm {
    alpha
  } else {
    f64::NAN
  }
}

/// The smallest positive root of `p3 x³ + p2 x² + p1 x + p0` (`p3 ≥ 0 > p0`), Newton's from `seed`
/// where it reaches that root and Brent's on a bracket from 0 where not; NaN with none.
fn smallest_positive_root([p3, p2, p1, p0]: [f64; 4], seed: f64) -> f64 {
  let cubic = |x: f64| p3 * x * x * x + p2 * x * x + p1 * x + p0;
  // Rising from `p0 < 0` at 0, the cubic crosses 0 below its first peak when that peak is ≥ 0.
  let disc = p2 * p2 - 3.0 * p3 * p1;
  let hump = (p2 < 0.0 && p1 > 0.0 && disc >= 0.0)
    .then(|| p1 / (disc.sqrt() - p2))
    .filter(|&peak| cubic(peak) >= 0.0);

  let mut x = seed;
  let mut converged = false;
  for _ in 0..100 {
    let fx = cubic(x);
    let dfx = 3.0 * p3 * x * x + 2.0 * p2 * x + p1;
    if dfx.abs() < 1e-15 {
      break;
    }
    let dx = fx / dfx;
    x -= dx;
    if dx.abs() < 1e-12 * x.abs().max(1e-12) {
      converged = true;
      break;
    }
  }
  if converged && x > 0.0 && hump.is_none_or(|peak| x <= peak) {
    return x;
  }

  let (lo, hi) = match hump {
    Some(peak) => (0.0, peak),
    // Past the hump at most one positive root is left, and a positive lead coefficient has one.
    None if p3 > 0.0 || p2 > 0.0 || (p2 == 0.0 && p1 > 0.0) => {
      let (mut lo, mut hi) = (0.0, seed.max(f64::MIN_POSITIVE));
      loop {
        if !hi.is_finite() {
          return f64::NAN;
        }
        if cubic(hi) > 0.0 {
          break (lo, hi);
        }
        (lo, hi) = (hi, 2.0 * hi);
      }
    }
    None => return f64::NAN,
  };
  brent::zero(lo, hi, f64::MIN_POSITIVE, cubic).unwrap_or(f64::NAN)
}

/// Black-Scholes(-Garman-Kohlhagen) price for FX under GK using BsmPricer.
pub fn bs_price_fx(s: f64, k: f64, r_d: f64, r_f: f64, tau: f64, sigma: f64) -> (f64, f64) {
  // Garman-Kohlhagen puts the foreign rate in the model's dividend slot,
  // so the query's `(r, q)` pair is `(r_d, r_f)`.
  BsmPricer::new(sigma, BsmCoc::GarmanKohlhagen1983).call_put(s, k, r_d, r_f, tau)
}

/// Delta on forward with premium included
pub fn fx_delta_from_forward(k: f64, f: f64, sigma: f64, tau: f64, r_f: f64, phi: f64) -> f64 {
  let d2 = (f / k).ln() / (sigma * tau.sqrt()) - 0.5 * sigma * tau.sqrt();
  let nd2 = norm_cdf(phi * d2);
  phi * (-r_f * tau).exp() * (k / f) * nd2
}

/// Garman–Kohlhagen `(call, put)` at the Hagan vol of `params` against the forward of `s`.
pub fn model_price_hagan(
  s: f64,
  k: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  params: &SabrParams,
) -> (f64, f64) {
  let fwd = forward_fx(s, tau, r_d, r_f);
  let sigma = hagan_implied_vol(k, fwd, tau, params);
  bs_price_fx(s, k, r_d, r_f, tau, sigma)
}
