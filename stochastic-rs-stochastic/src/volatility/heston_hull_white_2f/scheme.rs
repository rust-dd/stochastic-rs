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

  /// The device family of this step, its constants folded once.
  #[doc(hidden)]
  fn euler_spec<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, Self, S, B>,
  ) -> crate::euler::EulerSpec<T>
  where
    Self: Sized;

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

  fn euler_spec<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, Euler, S, B>,
  ) -> crate::euler::EulerSpec<T> {
    let l = &model.chol;
    crate::euler::EulerSpec::HestonHullWhite2F {
      kappa: model.kappa,
      theta: model.theta,
      sigma: model.sigma,
      a: model.a,
      b: model.b,
      sigma1: model.sigma1,
      sigma2: model.sigma2,
      l: [
        l[(0, 0)],
        l[(1, 0)],
        l[(1, 1)],
        l[(2, 0)],
        l[(2, 1)],
        l[(2, 2)],
        l[(3, 0)],
        l[(3, 1)],
        l[(3, 2)],
        l[(3, 3)],
      ],
    }
  }

  fn simulate_into<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, Euler, S, B>,
    curve: &[T],
    seed: &S,
    out: &mut HestonHullWhite2FPaths<T>,
  ) {
    let n = model.n;
    out.assert_points(n);
    let dt = model.dt();
    let mut z = Array2::<T>::zeros((4, n - 1));
    let mut normal = SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(seed);
    for mut row in z.rows_mut() {
      normal.fill_slice(
        row
          .as_slice_mut()
          .expect("HestonHullWhite2F noise rows must be contiguous"),
      );
    }
    let (s0, v0, r0) = model.start();
    let (mut state, mut int_r) = ([s0.ln(), v0, r0, T::zero()], T::zero());
    write_point(out, 0, [s0, v0, r0, T::zero(), int_r]);
    for i in 1..n {
      let j = i - 1;
      let shocks = [z[(0, j)], z[(1, j)], z[(2, j)], z[(3, j)]];
      let (next, r_dt) = model.euler_step(curve[i], dt, state, shocks);
      int_r += r_dt;
      write_point(out, i, [next[0].exp(), next[1], next[2], next[3], int_r]);
      state = next;
    }
  }
}

/// The variance after one Euler step from `v >= 0`, floored at zero as `Heston`'s `Floor` step.
fn floored_variance<T: FloatExt>(v: T, kappa: T, theta: T, sigma: T, dt: T, dw_v: T) -> T {
  (v + kappa * (theta - v) * dt + sigma * v.sqrt() * dw_v).max(T::zero())
}

impl<T: FloatExt, S: SeedExt, B> HestonHullWhite2F<T, Euler, S, B> {
  /// One Euler step of `(log S, v, r, u)` from four independent increments, with `½(r + r′)dt`,
  /// the trapezoid the spot and the integrated rate share.
  pub(super) fn euler_step(&self, target: T, dt: T, state: [T; 4], z: [T; 4]) -> ([T; 4], T) {
    let half = T::from_f64_fast(0.5);
    let l = &self.chol;
    let [log_s, v, r, u] = state;
    let dw_x = l[(0, 0)] * z[0];
    let dw_v = l[(1, 0)] * z[0] + l[(1, 1)] * z[1];
    let dw_r = l[(2, 0)] * z[0] + l[(2, 1)] * z[1] + l[(2, 2)] * z[2];
    let dw_u = l[(3, 0)] * z[0] + l[(3, 1)] * z[1] + l[(3, 2)] * z[2] + l[(3, 3)] * z[3];
    let v_next = floored_variance(v, self.kappa, self.theta, self.sigma, dt, dw_v);
    let (r_next, u_next) = self.rate_step(target, dt, r, u, dw_r, dw_u);
    let r_dt = half * (r + r_next) * dt;
    let log_s_next = log_s + r_dt - half * v * dt + v.sqrt() * dw_x;
    ([log_s_next, v_next, r_next, u_next], r_dt)
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

  fn euler_spec<T: FloatExt, S: SeedExt, B>(
    model: &HestonHullWhite2F<T, AndersenQe, S, B>,
  ) -> crate::euler::EulerSpec<T> {
    Self::check_parameters(model.kappa, model.theta, model.sigma, &model.rho);
    let dt = model.dt();
    let (two, half) = (T::from_f64_fast(2.0), T::from_f64_fast(0.5));
    let (kappa, theta, eps, rho_xv) = (model.kappa, model.theta, model.sigma, model.rho[(0, 1)]);
    let e_kd = (-kappa * dt).exp();
    let krho_eps = kappa * rho_xv / eps;
    let lq = &model.chol;
    crate::euler::EulerSpec::HestonHullWhite2FQe {
      theta,
      e_kd,
      c1: eps * eps * e_kd / kappa,
      c2: theta * eps * eps / (two * kappa),
      k0: -rho_xv * kappa * theta * dt / eps,
      k1: half * dt * (krho_eps - half) - rho_xv / eps,
      k2: half * dt * (krho_eps - half) + rho_xv / eps,
      a: model.a,
      b: model.b,
      sigma1: model.sigma1,
      sigma2: model.sigma2,
      l: [
        lq[(1, 1)],
        lq[(2, 1)],
        lq[(2, 2)],
        lq[(3, 1)],
        lq[(3, 2)],
        lq[(3, 3)],
      ],
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
