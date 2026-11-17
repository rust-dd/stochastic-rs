use std::f64::consts::PI;

use super::DiffusionModel;

/// The normal density; NaN at a variance not above zero.
pub(super) fn gaussian_pdf(x: f64, mean: f64, var: f64) -> f64 {
  if var.is_nan() || var <= 0.0 {
    return f64::NAN;
  }
  let z = (x - mean) / var.sqrt();
  (-0.5 * z * z).exp() / (2.0 * PI * var).sqrt()
}

/// The log of the normal density, finite in tails where the density underflows; NaN as
/// [`gaussian_pdf`].
pub(super) fn gaussian_ln_pdf(x: f64, mean: f64, var: f64) -> f64 {
  if var.is_nan() || var <= 0.0 {
    return f64::NAN;
  }
  let z = (x - mean) / var.sqrt();
  -0.5 * z * z - 0.5 * (2.0 * PI * var).ln()
}

/// Transition-density approximation method.
///
/// # References
/// - Euler: Maruyama, G. (1955). *Rendiconti del Circolo Matematico di Palermo*,
///   4(1), 48-90. <https://doi.org/10.1007/BF02846028>
/// - Ozaki: Ozaki, T. (1992). *Statistica Sinica*, 2(1), 113-135.
///   <https://www.jstor.org/stable/24304999>
/// - Shoji-Ozaki: Shoji, I. & Ozaki, T. (1998). *Biometrika*, 85(1), 240-243.
///   <https://doi.org/10.1093/biomet/85.1.240>
/// - Elerian: Elerian, O., Chib, S. & Shephard, N. (2001). *Econometrica*, 69(4),
///   959-993. <https://doi.org/10.1111/1468-0262.00226>
/// - Kessler: Kessler, M. (1997). *Scand. J. Statist.*, 24(2), 211-229.
///   <https://doi.org/10.1111/1467-9469.00059>
/// - Aït-Sahalia: Aït-Sahalia, Y. (2002). *Econometrica*, 70(1), 223-262.
///   <https://doi.org/10.1111/1468-0262.00274>
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum DensityApprox {
  /// Exact closed-form density (must be implemented by the model).
  Exact,
  /// Euler-Maruyama (Gaussian with Euler moments).
  Euler,
  /// Ozaki local-linearisation.
  Ozaki,
  /// Shoji-Ozaki (second-order drift correction).
  ShojiOzaki,
  /// Elerian / Milstein density.
  Elerian,
  /// Kessler second-order moment expansion.
  Kessler,
  /// Aït-Sahalia Hermite expansion (must be implemented by the model).
  AitSahalia,
}

const NO_EXACT: &str = "Exact density not implemented for this model";

impl DensityApprox {
  /// The transition density `p(xt | x0)` over `dt` from `t0`: `0.0` outside the method's support,
  /// NaN where it has none, as at a variance not above zero; panics if `model` lacks the method.
  pub fn density(&self, model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    if let Some((mean, var)) = self.normal_moments(model, x0, t0, dt) {
      return gaussian_pdf(xt, mean, var);
    }
    match self {
      DensityApprox::Exact => model.exact_density(x0, xt, t0, dt).expect(NO_EXACT),
      DensityApprox::AitSahalia => Self::density_ait_sahalia(model, x0, xt, t0, dt),
      DensityApprox::Elerian => match Self::ln_density_milstein(model, x0, xt, t0, dt) {
        Some(ln_density) => ln_density.exp(),
        None => DensityApprox::Euler.density(model, x0, xt, t0, dt),
      },
      DensityApprox::Euler
      | DensityApprox::Ozaki
      | DensityApprox::ShojiOzaki
      | DensityApprox::Kessler => unreachable!("a normal method returned above"),
    }
  }

  /// The log of [`density`](Self::density), formed in logs for a normal law and for Elerian's, so
  /// finite where the density underflows; `−∞` outside the support, NaN and panics as `density`.
  pub fn ln_density(&self, model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    if let Some((mean, var)) = self.normal_moments(model, x0, t0, dt) {
      return gaussian_ln_pdf(xt, mean, var);
    }
    match self {
      DensityApprox::Exact => model.exact_ln_density(x0, xt, t0, dt).expect(NO_EXACT),
      DensityApprox::AitSahalia => Self::density_ait_sahalia(model, x0, xt, t0, dt).ln(),
      DensityApprox::Elerian => Self::ln_density_milstein(model, x0, xt, t0, dt)
        .unwrap_or_else(|| DensityApprox::Euler.ln_density(model, x0, xt, t0, dt)),
      DensityApprox::Euler
      | DensityApprox::Ozaki
      | DensityApprox::ShojiOzaki
      | DensityApprox::Kessler => unreachable!("a normal method returned above"),
    }
  }

  /// The mean and variance of the four methods whose transition law is a normal.
  fn normal_moments(
    &self,
    model: &dyn DiffusionModel,
    x0: f64,
    t0: f64,
    dt: f64,
  ) -> Option<(f64, f64)> {
    match self {
      DensityApprox::Euler => Some(Self::moments_euler(model, x0, t0, dt)),
      DensityApprox::Ozaki => Some(Self::moments_ozaki(model, x0, t0, dt)),
      DensityApprox::ShojiOzaki => Some(Self::moments_shoji_ozaki(model, x0, t0, dt)),
      DensityApprox::Kessler => Some(Self::moments_kessler(model, x0, t0, dt)),
      DensityApprox::Exact | DensityApprox::Elerian | DensityApprox::AitSahalia => None,
    }
  }

  /// Ref: Maruyama (1955)
  fn moments_euler(model: &dyn DiffusionModel, x0: f64, t0: f64, dt: f64) -> (f64, f64) {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    (x0 + mu * dt, sig * sig * dt)
  }

  /// Ref: Ozaki (1992)
  fn moments_ozaki(model: &dyn DiffusionModel, x0: f64, t0: f64, dt: f64) -> (f64, f64) {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let mu_x = model.drift_x(x0, t0);

    if mu_x.abs() < 1e-10 {
      return Self::moments_euler(model, x0, t0, dt);
    }

    let mt = x0 + mu * ((mu_x * dt).exp() - 1.0) / mu_x;

    let diff = mt - x0;
    let kt = if x0.abs() > 1e-10 {
      let ratio = 1.0 + diff / x0;
      if ratio > 1e-30 {
        (2.0 / dt) * ratio.ln()
      } else {
        mu_x
      }
    } else {
      mu_x
    };

    let exp_kt_dt = (kt * dt).exp();
    let vt_sq = if kt.abs() > 1e-10 && exp_kt_dt > 1.0 {
      sig * sig * (exp_kt_dt - 1.0) / kt
    } else {
      sig * sig * dt
    };

    (mt, vt_sq)
  }

  /// Ref: Shoji & Ozaki (1998)
  fn moments_shoji_ozaki(model: &dyn DiffusionModel, x0: f64, t0: f64, dt: f64) -> (f64, f64) {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let mu_x = model.drift_x(x0, t0);
    let mu_xx = model.drift_xx(x0, t0);
    let mu_t = model.drift_t(x0, t0);

    let mt = 0.5 * sig * sig * mu_xx + mu_t;
    let lt = mu_x;

    if lt.abs() > 1e-10 {
      let exp_lt = (lt * dt).exp();
      let b2 = sig * sig * ((2.0 * lt * dt).exp() - 1.0) / (2.0 * lt);
      let a = x0 + mu / lt * (exp_lt - 1.0) + mt / (lt * lt) * (exp_lt - 1.0 - lt * dt);
      (a, b2)
    } else {
      let b2 = sig * sig * dt;
      let a = x0 + mu * dt + mt * dt * dt / 2.0;
      (a, b2)
    }
  }

  /// The Milstein step's log-density, `−∞` below its support, and `None` where the diffusion is
  /// flat in `x` and the step Euler's; in logs, as its `cosh` overflows from `√(cz) ≈ 710`.
  fn ln_density_milstein(
    model: &dyn DiffusionModel,
    x0: f64,
    xt: f64,
    t0: f64,
    dt: f64,
  ) -> Option<f64> {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let sig_x = model.diffusion_x(x0, t0);

    if sig_x.abs() < 1e-10 {
      return None;
    }

    let a_coeff = sig * sig_x * dt * 0.5;
    let b_coeff = -0.5 * sig / sig_x + x0 + mu * dt - a_coeff;

    if a_coeff.abs() < 1e-30 {
      return None;
    }

    let z = (xt - b_coeff) / a_coeff;
    let c = 1.0 / (sig_x * sig_x * dt);

    if z <= 0.0 {
      return Some(f64::NEG_INFINITY);
    }

    // `cosh(√(cz)) e^{−(c+z)/2} = ½ e^{−(√c − √z)²/2} (1 + e^{−2√(cz)})`.
    let gap = c.sqrt() - z.sqrt();
    let ln_kernel = -0.5 * gap * gap + (-2.0 * (c * z).sqrt()).exp().ln_1p();
    Some(-0.5 * z.ln() + ln_kernel - (4.0 * a_coeff.abs() * (2.0 * PI).sqrt()).ln())
  }

  /// Ref: Kessler (1997)
  fn moments_kessler(model: &dyn DiffusionModel, x0: f64, t0: f64, dt: f64) -> (f64, f64) {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let mu_x = model.drift_x(x0, t0);
    let sig_x = model.diffusion_x(x0, t0);
    let sig_xx = model.diffusion_xx(x0, t0);

    let sig2 = sig * sig;
    let d = dt * dt / 2.0;

    let e_x = x0 + mu * dt + (mu * mu_x + 0.5 * sig2 * sig_xx) * d;

    let term = 2.0 * sig * sig_x;
    let e_x2 = x0 * x0
      + (2.0 * mu * x0 + sig2) * dt
      + (2.0 * mu * (mu_x * x0 + mu + sig * sig_x)
        + sig2 * (sig_xx * x0 + 2.0 * sig_x + term + sig * sig_xx))
        * d;

    (e_x, (e_x2 - e_x * e_x).abs())
  }

  /// Ref: Aït-Sahalia (2002)
  fn density_ait_sahalia(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    model
      .ait_sahalia_density(x0, xt, t0, dt)
      .expect("Ait-Sahalia density not implemented for this model")
  }
}
