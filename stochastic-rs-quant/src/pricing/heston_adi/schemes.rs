//! The four ADI time-stepping schemes of in 't Hout & Foulon (2008), §2.4,
//! eqs. (2.17)–(2.20), on the split operator `F = F_0 + F_1 + F_2`, plus the
//! Rannacher start-up damping of §2.5.

use super::operators::Operators;

/// ADI splitting scheme; the recommended `θ` of each is
/// [`AdiScheme::default_theta`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AdiScheme {
  /// Douglas (2.17): first order when `ρ ≠ 0`, unconditionally stable for `θ ≥ ½`.
  Douglas,
  /// Craig–Sneyd (2.18): second order only at `θ = ½`.
  CraigSneyd,
  /// Modified Craig–Sneyd (2.19): second order for any `θ`; the paper's
  /// recommendation at `θ = ⅓` with damping.
  #[default]
  ModifiedCraigSneyd,
  /// Hundsdorfer–Verwer (2.20): second order for any `θ`; `θ = ½ + √3/6`.
  HundsdorferVerwer,
}

impl AdiScheme {
  /// The `θ` the paper singles out for each scheme (§2.5, §3).
  pub fn default_theta(self) -> f64 {
    match self {
      Self::Douglas | Self::CraigSneyd => 0.5,
      Self::ModifiedCraigSneyd => 1.0 / 3.0,
      Self::HundsdorferVerwer => 0.5 + 3.0_f64.sqrt() / 6.0,
    }
  }
}

fn axpy(y: &[f64], a: f64, x: &[f64]) -> Vec<f64> {
  y.iter().zip(x).map(|(yi, xi)| yi + a * xi).collect()
}

fn diff(a: &[f64], b: &[f64]) -> Vec<f64> {
  a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The two implicit unidirectional corrector stages shared by every scheme:
/// `Y_j = Y_{j−1} + θΔt (F_j(t_n, Y_j) − F_j(t_ref, U_ref))`, `j = 1, 2`.
fn correct(
  ops: &Operators,
  theta_dt: f64,
  t_new: f64,
  y0: &[f64],
  f1_ref: &[f64],
  f2_ref: &[f64],
) -> Vec<f64> {
  let rhs1 = axpy(y0, theta_dt, &diff(&ops.b1_at(t_new), f1_ref));
  let y1 = ops.solve_a1(theta_dt, &rhs1);
  let rhs2 = axpy(&y1, theta_dt, &diff(&ops.b2_at(t_new), f2_ref));
  ops.solve_a2(theta_dt, &rhs2)
}

/// One step `U_{n−1} → U_n` of the chosen scheme from `t_prev` over `dt`.
pub(super) fn step(
  ops: &Operators,
  scheme: AdiScheme,
  theta: f64,
  t_prev: f64,
  dt: f64,
  u: &[f64],
) -> Vec<f64> {
  let t_new = t_prev + dt;
  let theta_dt = theta * dt;
  let f0_u = ops.f0(t_prev, u);
  let f1_u = ops.f1(t_prev, u);
  let f2_u = ops.f2(t_prev, u);
  let f_u: Vec<f64> = f0_u
    .iter()
    .zip(&f1_u)
    .zip(&f2_u)
    .map(|((a, b), c)| a + b + c)
    .collect();
  let y0 = axpy(u, dt, &f_u);
  let y2 = correct(ops, theta_dt, t_new, &y0, &f1_u, &f2_u);
  match scheme {
    AdiScheme::Douglas => y2,
    AdiScheme::CraigSneyd => {
      let y0_tilde = axpy(&y0, 0.5 * dt, &diff(&ops.f0(t_new, &y2), &f0_u));
      correct(ops, theta_dt, t_new, &y0_tilde, &f1_u, &f2_u)
    }
    AdiScheme::ModifiedCraigSneyd => {
      let y0_hat = axpy(&y0, theta_dt, &diff(&ops.f0(t_new, &y2), &f0_u));
      let y0_tilde = axpy(&y0_hat, (0.5 - theta) * dt, &diff(&ops.f(t_new, &y2), &f_u));
      correct(ops, theta_dt, t_new, &y0_tilde, &f1_u, &f2_u)
    }
    AdiScheme::HundsdorferVerwer => {
      let y0_tilde = axpy(&y0, 0.5 * dt, &diff(&ops.f(t_new, &y2), &f_u));
      let f1_y2 = ops.f1(t_new, &y2);
      let f2_y2 = ops.f2(t_new, &y2);
      correct(ops, theta_dt, t_new, &y0_tilde, &f1_y2, &f2_y2)
    }
  }
}

/// Marches `u0` from `t = 0` to `tau` in `steps` steps. With `damping` the
/// first step is replaced by two Douglas steps of `Δt/2` at `θ = 1`, the
/// practical form of the Rannacher backward-Euler start-up the paper applies
/// to damp the non-smooth payoff (§2.5).
pub(super) fn march(
  ops: &Operators,
  scheme: AdiScheme,
  theta: f64,
  damping: bool,
  steps: usize,
  tau: f64,
  u0: Vec<f64>,
) -> Vec<f64> {
  let dt = tau / steps as f64;
  let mut u = u0;
  let mut t = 0.0;
  let mut remaining = steps;
  if damping && steps > 0 {
    u = step(ops, AdiScheme::Douglas, 1.0, t, 0.5 * dt, &u);
    t += 0.5 * dt;
    u = step(ops, AdiScheme::Douglas, 1.0, t, 0.5 * dt, &u);
    t += 0.5 * dt;
    remaining -= 1;
  }
  for _ in 0..remaining {
    u = step(ops, scheme, theta, t, dt, &u);
    t += dt;
  }
  u
}
