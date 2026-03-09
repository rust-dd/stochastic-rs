//! # Regime-Switching European Option Pricing
//!
//! Implements the regime-switching diffusion model as a [`FourierModelExt`],
//! enabling pricing via any Fourier pricer (Carr-Madan, Gil-Pelaez, Lewis).
//!
//! $$
//! dS_t = (r - q - \tfrac12\sigma_{Z_t}^2)S_t\,dt + \sigma_{Z_t}S_t\,dW_t
//! $$
//!
//! where $Z_t$ is a continuous-time Markov chain with generator matrix $Q$.
//!
//! The characteristic function is obtained via the matrix exponential of a
//! state-dependent drift/diffusion matrix:
//!
//! $$
//! \phi_T(\xi) = \mathbf{e}_{z_0}^\top \exp\!\bigl(T\bigl(Q + \mathrm{diag}(
//! i\xi\mu_k - \tfrac12\sigma_k^2\xi^2)\bigr)\bigr) \mathbf{1}
//! $$
//!
//! Source:
//! - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::fourier::Cumulants;
use super::fourier::FourierModelExt;

fn mat_mul_complex(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
  let n = a.nrows();
  let mut c = Array2::<Complex64>::zeros((n, n));
  for i in 0..n {
    for j in 0..n {
      let mut s = Complex64::new(0.0, 0.0);
      for k in 0..n {
        s += a[[i, k]] * b[[k, j]];
      }
      c[[i, j]] = s;
    }
  }
  c
}

fn mat_add_complex(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
  let n = a.nrows();
  let mut c = Array2::<Complex64>::zeros((n, n));
  for i in 0..n {
    for j in 0..n {
      c[[i, j]] = a[[i, j]] + b[[i, j]];
    }
  }
  c
}

fn mat_identity_complex(n: usize) -> Array2<Complex64> {
  let mut m = Array2::<Complex64>::zeros((n, n));
  for i in 0..n {
    m[[i, i]] = Complex64::new(1.0, 0.0);
  }
  m
}

fn mat_scale_complex(a: &Array2<Complex64>, s: Complex64) -> Array2<Complex64> {
  let n = a.nrows();
  let mut c = Array2::<Complex64>::zeros((n, n));
  for i in 0..n {
    for j in 0..n {
      c[[i, j]] = a[[i, j]] * s;
    }
  }
  c
}

fn mat_inf_norm(a: &Array2<Complex64>) -> f64 {
  let mut norm = 0.0_f64;
  let n = a.nrows();
  for i in 0..n {
    let mut row_sum = 0.0;
    for j in 0..n {
      row_sum += a[[i, j]].norm();
    }
    norm = norm.max(row_sum);
  }
  norm
}

fn matrix_exp_complex(a: &Array2<Complex64>) -> Array2<Complex64> {
  let n = a.nrows();
  assert!(n > 0);

  let norm = mat_inf_norm(a);

  let s = if norm > 0.5 {
    (norm * 2.0).log2().ceil() as u32
  } else {
    0
  };

  let scale = Complex64::new((2.0_f64).powi(s as i32), 0.0);
  let scaled = mat_scale_complex(a, Complex64::new(1.0, 0.0) / scale);

  let mut result = mat_identity_complex(n);
  let mut term = mat_identity_complex(n);

  for k in 1..=20 {
    term = mat_mul_complex(&term, &scaled);
    let factor = Complex64::new(1.0 / k as f64, 0.0);
    term = mat_scale_complex(&term, factor);
    result = mat_add_complex(&result, &term);
  }

  for _ in 0..s {
    result = mat_mul_complex(&result, &result);
  }

  result
}

/// Regime-switching diffusion model for Fourier pricing.
///
/// The stock follows GBM with regime-dependent volatility. Regime transitions
/// are governed by a continuous-time Markov chain with generator matrix `q_matrix`.
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use stochastic_rs::quant::pricing::regime_switching::RegimeSwitchingModel;
/// use stochastic_rs::quant::pricing::fourier::{FourierModelExt, LewisPricer};
///
/// let model = RegimeSwitchingModel {
///   q_matrix: array![[-0.5, 0.5], [1.0, -1.0]],
///   vols: array![0.15, 0.35],
///   r: 0.05,
///   q: 0.01,
///   initial_state: 0,
/// };
/// let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
/// assert!(price > 0.0);
/// ```
pub struct RegimeSwitchingModel {
  /// M x M generator matrix for the continuous-time Markov chain.
  /// Each row must sum to zero. Off-diagonal entries are non-negative.
  pub q_matrix: Array2<f64>,
  /// Per-regime volatilities (length M).
  pub vols: Array1<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Initial regime index (0-based).
  pub initial_state: usize,
}

impl FourierModelExt for RegimeSwitchingModel {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let m = self.vols.len();
    let i = Complex64::i();

    let mut a = Array2::<Complex64>::zeros((m, m));
    for row in 0..m {
      for col in 0..m {
        a[[row, col]] = Complex64::new(self.q_matrix[[row, col]] * t, 0.0);
      }
      let sig2 = self.vols[row] * self.vols[row];
      let drift = self.r - self.q - 0.5 * sig2;
      a[[row, row]] += i * xi * drift * t - 0.5 * sig2 * xi * xi * t;
    }

    let exp_a = matrix_exp_complex(&a);

    let mut chf_val = Complex64::new(0.0, 0.0);
    for col in 0..m {
      chf_val += exp_a[[self.initial_state, col]];
    }
    chf_val
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let m = self.vols.len();
    let sigma_eff = self.vols[self.initial_state];

    let (mean_sig2, mean_sig4) = if m == 2 {
      let q01 = self.q_matrix[[0, 1]];
      let q10 = self.q_matrix[[1, 0]];
      let total = q01 + q10;
      if total > 1e-12 {
        let pi0 = q10 / total;
        let pi1 = q01 / total;
        let s2 = pi0 * self.vols[0].powi(2) + pi1 * self.vols[1].powi(2);
        let s4 = pi0 * self.vols[0].powi(4) + pi1 * self.vols[1].powi(4);
        (s2, s4)
      } else {
        (sigma_eff.powi(2), sigma_eff.powi(4))
      }
    } else {
      (sigma_eff.powi(2), sigma_eff.powi(4))
    };

    Cumulants {
      c1: (self.r - self.q - 0.5 * mean_sig2) * t,
      c2: mean_sig2 * t,
      c4: 3.0 * mean_sig4 * t,
    }
  }
}

/// COS (Fourier-cosine expansion) pricer for European options under
/// regime-switching diffusion.
///
/// Source:
/// - Fang, F. & Oosterlee, C.W. (2008), "A Novel Pricing Method for
///   European Options Based on Fourier-Cosine Series Expansions"
/// - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)
pub struct CosPricer {
  /// Number of cosine expansion terms.
  pub n: usize,
  /// Grid half-width parameter (L). Controls the truncation of the density.
  pub l: f64,
}

impl Default for CosPricer {
  fn default() -> Self {
    Self { n: 256, l: 12.0 }
  }
}

impl CosPricer {
  /// Price a European call option under regime-switching diffusion.
  pub fn price_call(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q_div: f64,
    t: f64,
    q_matrix: &Array2<f64>,
    vols: &Array1<f64>,
    initial_state: usize,
  ) -> f64 {
    let model = RegimeSwitchingModel {
      q_matrix: q_matrix.clone(),
      vols: vols.clone(),
      r,
      q: q_div,
      initial_state,
    };

    let cum = model.cumulants(t);
    let width = self.l * (cum.c2.abs().sqrt() + cum.c4.abs().powf(0.25)).max(0.1);
    let a = cum.c1 - width;
    let b = cum.c1 + width;
    let bma = b - a;
    let pi = std::f64::consts::PI;

    let disc = (-r * t).exp();
    let i_unit = Complex64::i();

    let mut price = 0.0;
    for k_idx in 0..self.n {
      let kf = k_idx as f64;
      let xi = kf * pi / bma;

      let chf_val = model.chf(t, Complex64::new(xi, 0.0));

      let phase = (-i_unit * xi * a).exp();
      let f_k = (chf_val * phase).re;

      let v_k = self.call_payoff_coeff(s, k, a, b, kf);

      let weight = if k_idx == 0 { 0.5 } else { 1.0 };
      price += weight * f_k * v_k;
    }

    (disc * price).max(0.0)
  }

  /// Price a European put option under regime-switching diffusion.
  pub fn price_put(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q_div: f64,
    t: f64,
    q_matrix: &Array2<f64>,
    vols: &Array1<f64>,
    initial_state: usize,
  ) -> f64 {
    let call = self.price_call(s, k, r, q_div, t, q_matrix, vols, initial_state);
    (call - s * (-q_div * t).exp() + k * (-r * t).exp()).max(0.0)
  }

  fn call_payoff_coeff(&self, s: f64, k: f64, a: f64, b: f64, kf: f64) -> f64 {
    let bma = b - a;
    let x_star = (k / s).ln();
    let c = x_star.max(a);
    let d = b;

    if c >= d {
      return 0.0;
    }

    let (chi, psi) = self.chi_psi(kf, a, bma, c, d);

    (2.0 / bma) * (s * chi - k * psi)
  }

  fn chi_psi(&self, kf: f64, a: f64, bma: f64, c: f64, d: f64) -> (f64, f64) {
    let pi = std::f64::consts::PI;

    let psi = if kf.abs() < 1e-14 {
      d - c
    } else {
      let w = kf * pi / bma;
      (w * (d - a)).sin() / w - (w * (c - a)).sin() / w
    };

    let chi = if kf.abs() < 1e-14 {
      d.exp() - c.exp()
    } else {
      let w = kf * pi / bma;
      let denom = 1.0 + w * w;
      let term_d = d.exp() * ((w * (d - a)).cos() + w * (w * (d - a)).sin());
      let term_c = c.exp() * ((w * (c - a)).cos() + w * (w * (c - a)).sin());
      (term_d - term_c) / denom
    };

    (chi, psi)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  const TOL_FOURIER: f64 = 0.15;

  #[test]
  fn rs_single_regime_matches_bsm() {
    use super::super::fourier::BSMFourier;
    use super::super::fourier::LewisPricer;

    let sigma = 0.2;
    let r = 0.05;
    let q = 0.01;

    let rs_model = RegimeSwitchingModel {
      q_matrix: array![[0.0]],
      vols: array![sigma],
      r,
      q,
      initial_state: 0,
    };

    let bsm_model = BSMFourier { sigma, r, q };

    let rs_price = LewisPricer::price_call(&rs_model, 100.0, 100.0, r, q, 1.0);
    let bsm_price = LewisPricer::price_call(&bsm_model, 100.0, 100.0, r, q, 1.0);

    assert!(
      (rs_price - bsm_price).abs() < TOL_FOURIER,
      "Single-regime RS should match BSM: rs={rs_price}, bsm={bsm_price}"
    );
  }

  #[test]
  fn rs_two_regime_prices_positive() {
    use super::super::fourier::LewisPricer;

    let model = RegimeSwitchingModel {
      q_matrix: array![[-0.5, 0.5], [1.0, -1.0]],
      vols: array![0.15, 0.35],
      r: 0.05,
      q: 0.01,
      initial_state: 0,
    };

    let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    assert!(
      price > 0.0,
      "Two-regime RS call price should be positive, got={price}"
    );
  }

  #[test]
  fn rs_call_decreases_with_strike() {
    use super::super::fourier::LewisPricer;

    let model = RegimeSwitchingModel {
      q_matrix: array![[-0.5, 0.5], [1.0, -1.0]],
      vols: array![0.15, 0.35],
      r: 0.05,
      q: 0.01,
      initial_state: 0,
    };

    let p1 = LewisPricer::price_call(&model, 100.0, 90.0, 0.05, 0.01, 1.0);
    let p2 = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let p3 = LewisPricer::price_call(&model, 100.0, 110.0, 0.05, 0.01, 1.0);

    assert!(
      p1 > p2 && p2 > p3,
      "RS call should decrease with strike: p(90)={p1}, p(100)={p2}, p(110)={p3}"
    );
  }

  #[test]
  fn rs_put_call_parity() {
    use super::super::fourier::CarrMadanPricer;
    use super::super::fourier::LewisPricer;

    let model = RegimeSwitchingModel {
      q_matrix: array![[-0.5, 0.5], [1.0, -1.0]],
      vols: array![0.15, 0.35],
      r: 0.05,
      q: 0.01,
      initial_state: 0,
    };

    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let q_div = 0.01;
    let t = 1.0;

    let call = LewisPricer::price_call(&model, s, k, r, q_div, t);
    let pricer = CarrMadanPricer::default();
    let put = pricer.price_put(&model, s, k, r, q_div, t);

    let lhs = call - put;
    let rhs = s * (-q_div * t).exp() - k * (-r * t).exp();
    assert!(
      (lhs - rhs).abs() < 0.5,
      "Put-call parity: C-P={lhs}, S*e^{{-qT}} - K*e^{{-rT}}={rhs}"
    );
  }

  #[test]
  fn rs_three_regime_prices_positive() {
    use super::super::fourier::LewisPricer;

    let model = RegimeSwitchingModel {
      q_matrix: array![
        [-1.0, 0.5, 0.5],
        [0.3, -0.6, 0.3],
        [0.4, 0.4, -0.8],
      ],
      vols: array![0.10, 0.25, 0.50],
      r: 0.05,
      q: 0.01,
      initial_state: 0,
    };

    let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    assert!(
      price > 0.0,
      "Three-regime RS call price should be positive, got={price}"
    );
  }

  #[test]
  fn rs_lewis_vs_carr_madan() {
    use super::super::fourier::CarrMadanPricer;
    use super::super::fourier::LewisPricer;

    let model = RegimeSwitchingModel {
      q_matrix: array![[-0.5, 0.5], [1.0, -1.0]],
      vols: array![0.15, 0.35],
      r: 0.05,
      q: 0.01,
      initial_state: 0,
    };

    let lewis_price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let cm_price = CarrMadanPricer::default().price_call(&model, 100.0, 100.0, 0.05, 1.0);

    assert!(
      (lewis_price - cm_price).abs() < TOL_FOURIER,
      "RS Lewis vs Carr-Madan: lewis={lewis_price}, cm={cm_price}"
    );
  }

  #[test]
  fn matrix_exp_identity() {
    let zero = Array2::<Complex64>::zeros((2, 2));
    let result = matrix_exp_complex(&zero);
    for i in 0..2 {
      for j in 0..2 {
        let expected = if i == j { 1.0 } else { 0.0 };
        assert!(
          (result[[i, j]].re - expected).abs() < 1e-12 && result[[i, j]].im.abs() < 1e-12,
          "exp(0)[{i}][{j}] = {:?}, expected {expected}",
          result[[i, j]]
        );
      }
    }
  }

  #[test]
  fn matrix_exp_diagonal() {
    let a = Complex64::new(1.5, 0.0);
    let b = Complex64::new(-0.5, 0.0);
    let mut mat = Array2::<Complex64>::zeros((2, 2));
    mat[[0, 0]] = a;
    mat[[1, 1]] = b;
    let result = matrix_exp_complex(&mat);
    assert!(
      (result[[0, 0]] - a.exp()).norm() < 1e-10,
      "exp(diag)[0][0] = {:?}, expected {:?}",
      result[[0, 0]],
      a.exp()
    );
    assert!(
      (result[[1, 1]] - b.exp()).norm() < 1e-10,
      "exp(diag)[1][1] = {:?}, expected {:?}",
      result[[1, 1]],
      b.exp()
    );
  }

  #[test]
  fn matrix_exp_generator() {
    let t = 1.0;
    let mut q_mat = Array2::<Complex64>::zeros((2, 2));
    q_mat[[0, 0]] = Complex64::new(-0.5 * t, 0.0);
    q_mat[[0, 1]] = Complex64::new(0.5 * t, 0.0);
    q_mat[[1, 0]] = Complex64::new(1.0 * t, 0.0);
    q_mat[[1, 1]] = Complex64::new(-1.0 * t, 0.0);
    let result = matrix_exp_complex(&q_mat);

    for i in 0..2 {
      let row_sum: f64 = (0..2).map(|j| result[[i, j]].re).sum();
      assert!(
        (row_sum - 1.0).abs() < 1e-10,
        "Row {i} sum = {row_sum}, expected 1.0"
      );
    }

    for i in 0..2 {
      for j in 0..2 {
        assert!(
          result[[i, j]].re >= -1e-12,
          "exp(tQ)[{i}][{j}] = {}, should be non-negative",
          result[[i, j]].re
        );
      }
    }
  }

  #[test]
  fn cos_single_regime_call_positive() {
    let pricer = CosPricer::default();
    let q_mat = array![[0.0]];
    let vols = array![0.2];
    let price = pricer.price_call(100.0, 100.0, 0.05, 0.01, 1.0, &q_mat, &vols, 0);
    assert!(
      price > 0.0,
      "COS single-regime call price should be positive, got={price}"
    );
  }

  #[test]
  fn cos_two_regime_call_positive() {
    let pricer = CosPricer::default();
    let q_mat = array![[-0.5, 0.5], [1.0, -1.0]];
    let vols = array![0.15, 0.35];
    let price = pricer.price_call(100.0, 100.0, 0.05, 0.01, 1.0, &q_mat, &vols, 0);
    assert!(
      price > 0.0,
      "COS two-regime call price should be positive, got={price}"
    );
  }

  #[test]
  fn cos_call_decreases_with_strike() {
    let pricer = CosPricer::default();
    let q_mat = array![[-0.5, 0.5], [1.0, -1.0]];
    let vols = array![0.15, 0.35];

    let p1 = pricer.price_call(100.0, 90.0, 0.05, 0.01, 1.0, &q_mat, &vols, 0);
    let p2 = pricer.price_call(100.0, 100.0, 0.05, 0.01, 1.0, &q_mat, &vols, 0);
    let p3 = pricer.price_call(100.0, 110.0, 0.05, 0.01, 1.0, &q_mat, &vols, 0);

    assert!(
      p1 > p2 && p2 > p3,
      "COS call should decrease with strike: p(90)={p1}, p(100)={p2}, p(110)={p3}"
    );
  }

  #[test]
  fn cos_agrees_with_lewis() {
    use super::super::fourier::LewisPricer;

    let pricer = CosPricer { n: 512, l: 14.0 };

    let q_mat = array![[0.0]];
    let vols = array![0.2];

    let cos_call = pricer.price_call(100.0, 100.0, 0.05, 0.01, 1.0, &q_mat, &vols, 0);

    let model = RegimeSwitchingModel {
      q_matrix: q_mat,
      vols,
      r: 0.05,
      q: 0.01,
      initial_state: 0,
    };
    let lewis_call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);

    assert!(
      (cos_call - lewis_call).abs() < 0.5,
      "COS vs Lewis single-regime: cos={cos_call}, lewis={lewis_call}"
    );
  }

  #[test]
  fn cos_put_call_parity() {
    let pricer = CosPricer::default();

    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let q_div = 0.01;
    let t = 1.0;
    let q_mat = array![[-0.5, 0.5], [1.0, -1.0]];
    let vols = array![0.15, 0.35];

    let call = pricer.price_call(s, k, r, q_div, t, &q_mat, &vols, 0);
    let put = pricer.price_put(s, k, r, q_div, t, &q_mat, &vols, 0);

    let lhs = call - put;
    let rhs = s * (-q_div * t).exp() - k * (-r * t).exp();
    assert!(
      (lhs - rhs).abs() < 0.5,
      "COS put-call parity: C-P={lhs}, expected={rhs}"
    );
  }
}
