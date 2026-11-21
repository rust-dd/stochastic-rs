use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::mc::McEstimate;

use super::HestonMalliavinGreeks;
use super::estimate;
use super::undefined;

pub(super) struct HestonElKhatibPath {
  pub(super) s: Array1<f64>,
  pub(super) v: Array1<f64>,
  pub(super) dw_s: Array1<f64>,
  pub(super) dw_v: Array1<f64>,
}

impl HestonMalliavinGreeks {
  fn sample_el_khatib_path(
    &self,
    normal_s: &mut Seeded<SimdNormal<f64>>,
    normal_perp: &mut Seeded<SimdNormal<f64>>,
  ) -> HestonElKhatibPath {
    let n_increments = self.n_steps - 1;
    let corr_scale = (1.0 - self.params.rho * self.params.rho).max(0.0).sqrt();
    let mut dw_s = Array1::<f64>::zeros(n_increments);
    let mut dw_v = Array1::<f64>::zeros(n_increments);

    for k in 0..n_increments {
      let dws = normal_s.sample();
      let dwp = normal_perp.sample();
      dw_s[k] = dws;
      dw_v[k] = self.params.rho * dws + corr_scale * dwp;
    }

    let (s, v) = self.simulate_heston_from_increments(&dw_s, &dw_v);
    HestonElKhatibPath { s, v, dw_s, dw_v }
  }

  fn simulate_heston_from_increments(
    &self,
    dw_s: &Array1<f64>,
    dw_v: &Array1<f64>,
  ) -> (Array1<f64>, Array1<f64>) {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let mut s = Array1::<f64>::zeros(self.n_steps);
    let v = self.variance_path_from_shifted_increments(dw_v, 0.0);

    s[0] = self.s;
    for k in 0..(self.n_steps - 1) {
      let s_prev = s[k];
      let sqrt_v = v[k].max(0.0).sqrt();
      s[k + 1] = s_prev + self.r * s_prev * dt + s_prev * sqrt_v * dw_s[k];
    }

    (s, v)
  }

  fn variance_path_from_shifted_increments(&self, dw_v: &Array1<f64>, shift: f64) -> Array1<f64> {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let mut v = Array1::<f64>::zeros(self.n_steps);
    v[0] = self.params.v0.max(0.0);

    for k in 0..(self.n_steps - 1) {
      let v_prev = v[k].max(0.0);
      let dv = self.params.kappa * (self.params.theta - v_prev) * dt
        + self.params.sigma * v_prev.sqrt() * (dw_v[k] + shift);
      v[k + 1] = (v[k] + dv).max(0.0);
    }

    v
  }

  fn el_khatib_a(&self, v: &Array1<f64>, dw_s: &Array1<f64>, dw_v: &Array1<f64>) -> f64 {
    self.el_khatib_a_with_shifts(v, dw_s, dw_v, 0.0, 0.0)
  }

  fn el_khatib_a_with_shifts(
    &self,
    v: &Array1<f64>,
    dw_s: &Array1<f64>,
    dw_v: &Array1<f64>,
    dw_s_shift: f64,
    dw_v_shift: f64,
  ) -> f64 {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let n_increments = self.n_steps - 1;

    if self.params.sigma.abs() <= f64::EPSILON || self.params.rho.abs() <= f64::EPSILON {
      let mut a = 0.0;
      for i in 0..n_increments {
        a += v[i].max(0.0).sqrt() * dt;
      }
      return a;
    }

    let mut a = 0.0;
    for i in 0..n_increments {
      let sqrt_v_i = v[i].max(0.0).sqrt();
      let mut g = sqrt_v_i;
      let mut d_v = self.params.rho * self.params.sigma * sqrt_v_i;

      for j in i..n_increments {
        let sqrt_v_j = v[j].max(Self::EL_KHATIB_VAR_FLOOR).sqrt();
        let d_sigma = 0.5 / sqrt_v_j;
        g += d_sigma * d_v * (dw_s[j] + dw_s_shift - sqrt_v_j * dt);

        let raw_next = v[j]
          + self.params.kappa * (self.params.theta - v[j].max(0.0)) * dt
          + self.params.sigma * v[j].max(0.0).sqrt() * (dw_v[j] + dw_v_shift);
        let tangent = 1.0 - self.params.kappa * dt
          + self.params.sigma * 0.5 / sqrt_v_j * (dw_v[j] + dw_v_shift);
        d_v = if raw_next > 0.0 { d_v * tangent } else { 0.0 };
      }

      a += g * dt;
    }

    a
  }

  fn shifted_el_khatib_a(
    &self,
    dw_s: &Array1<f64>,
    dw_v: &Array1<f64>,
    dw_s_shift: f64,
    dw_v_shift: f64,
  ) -> f64 {
    let v = self.variance_path_from_shifted_increments(dw_v, dw_v_shift);
    self.el_khatib_a_with_shifts(&v, dw_s, dw_v, dw_s_shift, dw_v_shift)
  }

  fn el_khatib_du_a(&self, dw_s: &Array1<f64>, dw_v: &Array1<f64>) -> f64 {
    if self.params.sigma.abs() <= f64::EPSILON || self.params.rho.abs() <= f64::EPSILON {
      return 0.0;
    }

    let dt = self.tau / (self.n_steps - 1) as f64;
    let eps = Self::EL_KHATIB_DU_EPS;
    let dw_s_shift = eps * dt;
    let dw_v_shift = self.params.rho * dw_s_shift;
    let a_up = self.shifted_el_khatib_a(dw_s, dw_v, dw_s_shift, dw_v_shift);
    let a_dn = self.shifted_el_khatib_a(dw_s, dw_v, -dw_s_shift, -dw_v_shift);

    (a_up - a_dn) / (2.0 * eps)
  }

  fn regularize_el_khatib_a(a: f64) -> f64 {
    if !a.is_finite() {
      return Self::EL_KHATIB_A_FLOOR;
    }

    if a.abs() >= Self::EL_KHATIB_A_FLOOR {
      a
    } else if a.is_sign_negative() {
      -Self::EL_KHATIB_A_FLOOR
    } else {
      Self::EL_KHATIB_A_FLOOR
    }
  }

  /// El-Khatib's (2009, arXiv:0904.3247, Prop. 5) delta with the Skorohod correction `D_u A`, a
  /// central Cameron–Martin shift of the increments, in O(n_steps²).
  ///
  /// $$
  /// \pi_\Delta=\frac{1}{S_0}\Big(\frac{W_T}{A}+\frac{D_uA}{A^2}\Big),\ A=\int_0^T G(t,T)\,dt
  /// $$
  pub fn delta_el_khatib(&self) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let dt = self.tau / (self.n_steps - 1) as f64;
    let sqrt_dt = dt.sqrt();
    let mut normal_s = SimdNormal::new(0.0, sqrt_dt).seeded(&Unseeded);
    let mut normal_perp = SimdNormal::new(0.0, sqrt_dt).seeded(&Unseeded);

    self.delta_el_khatib_from_normals(&mut normal_s, &mut normal_perp)
  }

  /// Seeded variant of [`delta_el_khatib`](Self::delta_el_khatib) for reproducible
  /// tests and benchmarks.
  pub fn delta_el_khatib_with_seed(&self, seed: u64) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let dt = self.tau / (self.n_steps - 1) as f64;
    let sqrt_dt = dt.sqrt();
    let mut normal_s = SimdNormal::new(0.0, sqrt_dt).seeded(&Deterministic::new(seed));
    let mut normal_perp =
      SimdNormal::new(0.0, sqrt_dt).seeded(&Deterministic::new(seed ^ 0x9E37_79B9_7F4A_7C15));

    self.delta_el_khatib_from_normals(&mut normal_s, &mut normal_perp)
  }

  fn delta_el_khatib_from_normals(
    &self,
    normal_s: &mut Seeded<SimdNormal<f64>>,
    normal_perp: &mut Seeded<SimdNormal<f64>>,
  ) -> McEstimate<f64> {
    let discount = (-self.r * self.tau).exp();
    let (mut sum, mut sum_sq) = (0.0, 0.0);

    for _ in 0..self.n_paths {
      let path = self.sample_el_khatib_path(normal_s, normal_perp);
      let payoff = (path.s[self.n_steps - 1] - self.k).max(0.0);
      let w_t: f64 = path.dw_s.iter().sum();
      let a = Self::regularize_el_khatib_a(self.el_khatib_a(&path.v, &path.dw_s, &path.dw_v));
      let du_a = self.el_khatib_du_a(&path.dw_s, &path.dw_v);
      let weight = (w_t / a + du_a / (a * a)) / self.s;

      if weight.is_finite() {
        let contribution = discount * payoff * weight;
        sum += contribution;
        sum_sq += contribution * contribution;
      }
    }

    estimate(sum, sum_sq, self.n_paths)
  }
}
