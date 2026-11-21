//! The variance steps of [`HestonHullWhite2F`]: [`Euler`], floored at zero as `Heston`'s is, and
//! [`AndersenQe`], whose spot step is Andersen's eq. 33 with the rate's trapezoid as its drift.

use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonHullWhite2F;
use super::HestonHullWhite2FPaths;
use crate::linalg::cholesky_lower;
use crate::traits::FloatExt;
use crate::volatility::heston::AndersenQe;
use crate::volatility::heston::Euler;

mod sealed {
  #[diagnostic::on_unimplemented(
    message = "`{Self}` cannot implement the sealed trait `HybridScheme`",
    note = "the variance schemes are `Euler` and `AndersenQe`"
  )]
  pub trait Sealed {}
}

impl sealed::Sealed for Euler {}

impl sealed::Sealed for AndersenQe {}

/// Compile-time selector of [`HestonHullWhite2F`]'s variance step, sealed to [`Euler`] and
/// [`AndersenQe`].
pub trait HybridScheme: sealed::Sealed + Send + Sync + 'static {
  /// Lower Cholesky factor of `rho`, permuted to the driver order this scheme steps.
  fn factor<T: FloatExt>(rho: &Array2<T>) -> Array2<T>;

  /// Panics unless the variance parameters and `rho` meet this scheme's own preconditions.
  fn check_parameters<T: FloatExt>(kappa: T, theta: T, sigma: T, rho: &Array2<T>);

  /// One path into `out` from `seed`, the sampler's basis; `curve[i]` is `theta_r` at point `i`.
  fn simulate_into<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, Self, S, B>,
    curve: &[T],
    seed: &S,
    out: &mut HestonHullWhite2FPaths<T>,
  ) where
    Self: Sized;
}

impl<T: FloatExt, Sch: HybridScheme, S: SeedExt, B> HestonHullWhite2F<T, Sch, S, B> {
  /// One Euler step of the rate leg; `target` is `theta_r` at the step's end, as in `HullWhite2F`.
  fn rate_step(&self, target: T, dt: T, r: T, u: T, dw_r: T, dw_u: T) -> (T, T) {
    let r_next = r + (target + u - self.a * r) * dt + self.sigma1 * dw_r;
    let u_next = u - self.b * u * dt + self.sigma2 * dw_u;
    (r_next, u_next)
  }

  fn start(&self) -> (T, T, T) {
    (self.s0, self.v0, self.r0.unwrap_or(T::zero()))
  }
}

fn write_point<T: FloatExt>(out: &mut HestonHullWhite2FPaths<T>, i: usize, state: [T; 5]) {
  let [s, v, r, u, integrated_rate] = state;
  out.s[i] = s;
  out.v[i] = v;
  out.r[i] = r;
  out.u[i] = u;
  out.integrated_rate[i] = integrated_rate;
}

impl HybridScheme for Euler {
  fn factor<T: FloatExt>(rho: &Array2<T>) -> Array2<T> {
    cholesky_lower(rho)
  }

  fn check_parameters<T: FloatExt>(_kappa: T, _theta: T, _sigma: T, _rho: &Array2<T>) {}

  fn simulate_into<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, Euler, S, B>,
    curve: &[T],
    seed: &S,
    out: &mut HestonHullWhite2FPaths<T>,
  ) {
    let n = model.n;
    out.assert_points(n);
    let dt = model.dt();
    let half = T::from_f64_fast(0.5);
    let mut z = Array2::<T>::zeros((4, n - 1));
    let mut normal = SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(seed);
    for mut row in z.rows_mut() {
      normal.fill_slice(
        row
          .as_slice_mut()
          .expect("HestonHullWhite2F noise rows must be contiguous"),
      );
    }
    let l = &model.chol;
    let (s0, v0, r0) = model.start();
    let (mut log_s, mut v, mut r, mut u, mut int_r) = (s0.ln(), v0, r0, T::zero(), T::zero());
    write_point(out, 0, [s0, v0, r0, u, int_r]);
    for i in 1..n {
      let j = i - 1;
      let dw_x = l[(0, 0)] * z[(0, j)];
      let dw_v = l[(1, 0)] * z[(0, j)] + l[(1, 1)] * z[(1, j)];
      let dw_r = l[(2, 0)] * z[(0, j)] + l[(2, 1)] * z[(1, j)] + l[(2, 2)] * z[(2, j)];
      let dw_u = l[(3, 0)] * z[(0, j)]
        + l[(3, 1)] * z[(1, j)]
        + l[(3, 2)] * z[(2, j)]
        + l[(3, 3)] * z[(3, j)];
      let v_next =
        (v + model.kappa * (model.theta - v) * dt + model.sigma * v.sqrt() * dw_v).max(T::zero());
      let (r_next, u_next) = model.rate_step(curve[i], dt, r, u, dw_r, dw_u);
      let r_dt = half * (r + r_next) * dt;
      log_s = log_s + r_dt - half * v * dt + v.sqrt() * dw_x;
      int_r += r_dt;
      write_point(out, i, [log_s.exp(), v_next, r_next, u_next, int_r]);
      v = v_next;
      r = r_next;
      u = u_next;
    }
  }
}

impl HybridScheme for AndersenQe {
  /// `(v, r, u, x)` order: with the rate-variance correlations zero the rates draw from their own
  /// normals and the spot's remaining components carry variance `1 − ρ_xv²`.
  fn factor<T: FloatExt>(rho: &Array2<T>) -> Array2<T> {
    let order = [1, 2, 3, 0];
    let permuted = Array2::from_shape_fn((4, 4), |(i, j)| rho[(order[i], order[j])]);
    cholesky_lower(&permuted)
  }

  /// The moments need `kappa`, `theta`, `sigma > 0`; eq. 33's shock is independent of the variance
  /// only with the rate-variance correlations zero.
  fn check_parameters<T: FloatExt>(kappa: T, theta: T, sigma: T, rho: &Array2<T>) {
    require!(kappa > T::zero(), kappa);
    require!(theta > T::zero(), theta);
    require!(sigma > T::zero(), sigma);
    for (what, index) in [("rho[(1, 2)]", (1, 2)), ("rho[(1, 3)]", (1, 3))] {
      let value = rho[index];
      assert!(
        value == T::zero(),
        "{what} must satisfy `{what} == T::zero()`, got {what} = {value:?}"
      );
    }
  }

  /// Andersen (2008) QE variance step, then eq. 33 for the log spot with the rate's trapezoidal
  /// sum in place of the constant drift and the orthogonal shock correlated with the rate shocks.
  fn simulate_into<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, AndersenQe, S, B>,
    curve: &[T],
    seed: &S,
    out: &mut HestonHullWhite2FPaths<T>,
  ) {
    Self::check_parameters(model.kappa, model.theta, model.sigma, &model.rho);
    let n = model.n;
    out.assert_points(n);
    let dt = model.dt();
    let (kappa, theta, eps) = (model.kappa, model.theta, model.sigma);
    let rho_xv = model.rho[(0, 1)];
    let one = T::one();
    let two = T::from_f64_fast(2.0);
    let half = T::from_f64_fast(0.5);
    let psi_c = T::from_f64_fast(1.5);
    let e_kd = (-kappa * dt).exp();
    let c1 = eps * eps * e_kd * (one - e_kd) / kappa;
    let c2 = theta * eps * eps * (one - e_kd) * (one - e_kd) / (two * kappa);
    let krho_eps = kappa * rho_xv / eps;
    let k0 = -rho_xv * kappa * theta * dt / eps;
    let k1 = half * dt * (krho_eps - half) - rho_xv / eps;
    let k2 = half * dt * (krho_eps - half) + rho_xv / eps;
    let sqrt_dt = dt.sqrt();
    let lq = &model.chol;
    // The uniform stream derives from the basis, which keeps it independent of the normals.
    let mut normal = SimdNormal::<T>::new(T::zero(), one).seeded(seed);
    let mut urng = seed.derive().rng();
    let (s0, v0, r0) = model.start();
    let (mut log_s, mut v, mut r, mut u, mut int_r) = (s0.ln(), v0, r0, T::zero(), T::zero());
    write_point(out, 0, [s0, v0, r0, u, int_r]);
    for i in 1..n {
      let m = theta + (v - theta) * e_kd;
      let s2 = v * c1 + c2;
      let psi = s2 / (m * m);
      let v_next = if psi <= psi_c {
        let inv = two / psi;
        let b2 = inv - one + (inv * (inv - one)).sqrt();
        let a = m / (one + b2);
        let b = b2.sqrt();
        let zv = normal.sample();
        a * (b + zv) * (b + zv)
      } else {
        let p = (psi - one) / (psi + one);
        let beta = (one - p) / m;
        let uniform = T::sample_uniform_simd(&mut urng);
        if uniform <= p {
          T::zero()
        } else {
          ((one - p) / (one - uniform)).ln() / beta
        }
      };
      let z_r = normal.sample();
      let z_3 = normal.sample();
      let z_4 = normal.sample();
      let dw_r = sqrt_dt * lq[(1, 1)] * z_r;
      let dw_u = sqrt_dt * (lq[(2, 1)] * z_r + lq[(2, 2)] * z_3);
      let z_x = lq[(3, 1)] * z_r + lq[(3, 2)] * z_3 + lq[(3, 3)] * z_4;
      let (r_next, u_next) = model.rate_step(curve[i], dt, r, u, dw_r, dw_u);
      let r_dt = half * (r + r_next) * dt;
      let vol = (half * dt * (v + v_next)).max(T::zero()).sqrt();
      log_s = log_s + r_dt + k0 + k1 * v + k2 * v_next + vol * z_x;
      int_r += r_dt;
      write_point(out, i, [log_s.exp(), v_next, r_next, u_next, int_r]);
      v = v_next;
      r = r_next;
      u = u_next;
    }
  }
}
