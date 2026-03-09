use std::f64::consts::PI;

use super::DiffusionModel;

/// Gaussian probability density function.
pub(super) fn gaussian_pdf(x: f64, mean: f64, var: f64) -> f64 {
  if var <= 0.0 {
    return 1e-30;
  }
  let z = (x - mean) / var.sqrt();
  (-0.5 * z * z).exp() / (2.0 * PI * var).sqrt()
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
  /// Ait-Sahalia Hermite expansion (must be implemented by the model).
  AitSahalia,
}

impl DensityApprox {
  /// Evaluate the transition density p(x0 -> xt | t0, dt) under the given model.
  pub fn density(&self, model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    match self {
      DensityApprox::Exact => Self::density_exact(model, x0, xt, t0, dt),
      DensityApprox::Euler => Self::density_euler(model, x0, xt, t0, dt),
      DensityApprox::Ozaki => Self::density_ozaki(model, x0, xt, t0, dt),
      DensityApprox::ShojiOzaki => Self::density_shoji_ozaki(model, x0, xt, t0, dt),
      DensityApprox::Elerian => Self::density_elerian(model, x0, xt, t0, dt),
      DensityApprox::Kessler => Self::density_kessler(model, x0, xt, t0, dt),
      DensityApprox::AitSahalia => Self::density_ait_sahalia(model, x0, xt, t0, dt),
    }
  }

  fn density_exact(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    model
      .exact_density(x0, xt, t0, dt)
      .expect("Exact density not implemented for this model")
  }

  /// Ref: Maruyama (1955)
  fn density_euler(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let mean = x0 + mu * dt;
    let var = sig * sig * dt;
    gaussian_pdf(xt, mean, var)
  }

  /// Ref: Ozaki (1992)
  fn density_ozaki(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let mu_x = model.drift_x(x0, t0);

    if mu_x.abs() < 1e-10 {
      return Self::density_euler(model, x0, xt, t0, dt);
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

    gaussian_pdf(xt, mt, vt_sq)
  }

  /// Ref: Shoji & Ozaki (1998)
  fn density_shoji_ozaki(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
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
      gaussian_pdf(xt, a, b2)
    } else {
      let b2 = sig * sig * dt;
      let a = x0 + mu * dt + mt * dt * dt / 2.0;
      gaussian_pdf(xt, a, b2)
    }
  }

  /// Ref: Elerian, Chib & Shephard (2001)
  fn density_elerian(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    let mu = model.drift(x0, t0);
    let sig = model.diffusion(x0, t0);
    let sig_x = model.diffusion_x(x0, t0);

    if sig_x.abs() < 1e-10 {
      return Self::density_euler(model, x0, xt, t0, dt);
    }

    let a_coeff = sig * sig_x * dt * 0.5;
    let b_coeff = -0.5 * sig / sig_x + x0 + mu * dt - a_coeff;

    if a_coeff.abs() < 1e-30 {
      return Self::density_euler(model, x0, xt, t0, dt);
    }

    let z = (xt - b_coeff) / a_coeff;
    let c = 1.0 / (sig_x * sig_x * dt);

    if z <= 0.0 {
      return 1e-30;
    }

    let val = z.powf(-0.5) * (c * z).sqrt().cosh() * (-0.5 * (c + z)).exp()
      / (2.0 * a_coeff.abs() * (2.0 * PI).sqrt());

    if val.is_finite() && val > 0.0 {
      val
    } else {
      1e-30
    }
  }

  /// Ref: Kessler (1997)
  fn density_kessler(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
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

    let v = (e_x2 - e_x * e_x).abs();
    let std = v.sqrt();
    if std < 1e-30 {
      return 1e-30;
    }

    let z = (xt - e_x) / std;
    (-0.5 * z * z).exp() / ((2.0 * PI).sqrt() * std)
  }

  fn density_ait_sahalia(model: &dyn DiffusionModel, x0: f64, xt: f64, t0: f64, dt: f64) -> f64 {
    model
      .ait_sahalia_density(x0, xt, t0, dt)
      .expect("Ait-Sahalia density not implemented for this model")
  }
}
