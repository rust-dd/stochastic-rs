//! # Regime-Switching European Option Pricing
//!
//! Implements the regime-switching diffusion model as a [`FourierModelExt`],
//! enabling pricing via any Fourier pricer (Carr-Madan, Gil-Pelaez, Lewis).
//!
//! $$
//! d\ln S_t = (r - q - \tfrac12\sigma_{Z_t}^2)\,dt + \sigma_{Z_t}\,dW_t
//! $$
//!
//! where $Z_t$ is a continuous-time Markov chain with generator matrix $Q$.
//!
//! The characteristic function of $\ln(S_T/F_T)$, with $\mu_k=-\tfrac12\sigma_k^2$, is the
//! matrix exponential of a state-dependent drift/diffusion matrix:
//!
//! $$
//! \phi_T(\xi) = \mathbf{e}_{z_0}^\top \exp\!\bigl(T\bigl(Q + \mathrm{diag}(
//! i\xi\mu_k - \tfrac12\sigma_k^2\xi^2)\bigr)\bigr) \mathbf{1}
//! $$
//!
//! Source:
//! - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)

use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex64;

use super::fourier::CosEngine;
use super::fourier::Cumulants;
use super::fourier::FourierModelExt;
use crate::OptionType;
use crate::traits::RealExt;

fn mat_inf_norm(a: &Array2<Complex64>) -> f64 {
  let abs = a.mapv(|z| z.norm());
  f64::max_or_nan_fold(abs.rows().into_iter().map(|row| row.sum())).max_or_nan(0.0)
}

fn matrix_exp_complex(a: &Array2<Complex64>) -> Array2<Complex64> {
  let n = a.nrows();
  assert!(n > 0, "n must satisfy `n > 0`, got n = {n:?}");

  let norm = mat_inf_norm(a);

  let s = if norm > 0.5 {
    (norm * 2.0).log2().ceil() as u32
  } else {
    0
  };

  let scale_factor = Complex64::new(1.0, 0.0) / Complex64::new((2.0_f64).powi(s as i32), 0.0);
  let scaled = a.mapv(|z| z * scale_factor);

  let identity: Array2<Complex64> = Array2::eye(n);
  let mut result = identity.clone();
  let mut term = identity;

  for k in 1..=20 {
    term = term.dot(&scaled);
    let factor = Complex64::new(1.0 / k as f64, 0.0);
    term.mapv_inplace(|z| z * factor);
    result += &term;
  }

  for _ in 0..s {
    result = result.dot(&result);
  }

  result
}

/// Regime-switching diffusion for Fourier pricing: a GBM whose volatility is driven by a
/// continuous-time Markov chain with generator `q_matrix`.
///
/// ```
/// use ndarray::{array, Array2};
/// use stochastic_rs_quant::pricing::regime_switching::RegimeSwitchingModel;
/// use stochastic_rs_quant::pricing::fourier::{FourierModelExt, LewisPricer};
///
/// let model = RegimeSwitchingModel {
///   q_matrix: array![[-0.5, 0.5], [1.0, -1.0]],
///   vols: array![0.15, 0.35],
///   initial_state: 0,
/// };
/// let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
/// assert!(price > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct RegimeSwitchingModel {
  /// M x M generator matrix for the continuous-time Markov chain.
  /// Each row must sum to zero. Off-diagonal entries are non-negative.
  pub q_matrix: Array2<f64>,
  /// Per-regime volatilities (length M).
  pub vols: Array1<f64>,
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
      let drift = -0.5 * sig2;
      a[[row, row]] += i * xi * drift * t - 0.5 * sig2 * xi * xi * t;
    }

    let exp_a = matrix_exp_complex(&a);

    let mut chf_val = Complex64::new(0.0, 0.0);
    for col in 0..m {
      chf_val += exp_a[[self.initial_state, col]];
    }
    chf_val
  }

  /// Exact, from `I = ∫₀ᵗ σ²(Z_s) ds`: given `I` the log-return is `N(−I/2, I)`, so its cumulant
  /// function is `K_I((s² − s)/2)` and `c4 = 3κ₂(I) + (3/2)κ₃(I) + κ₄(I)/16`.
  fn cumulants(&self, t: f64) -> Cumulants {
    let [centre, ..] = self.integrated_variance_moments(t, 0.0, 1);
    let [m1, m2, m3, m4] = self.integrated_variance_moments(t, centre, 4);
    let k2 = m2 - m1 * m1;
    let k3 = m3 - 3.0 * m2 * m1 + 2.0 * m1 * m1 * m1;
    let k4 = m4 - 4.0 * m3 * m1 - 3.0 * m2 * m2 + 12.0 * m2 * m1 * m1 - 6.0 * m1.powi(4);
    let mean = centre + m1;
    Cumulants {
      c1: -0.5 * mean,
      c2: mean + 0.25 * k2,
      c4: Some(3.0 * k2 + 1.5 * k3 + k4 / 16.0),
    }
  }
}

impl RegimeSwitchingModel {
  /// `E[(I − c)ᵏ] = k!·B₁,ₖ₊₁1` at the initial state for `k ≤ order`, `B` the first block row of
  /// `exp(A)`, `A` with `tQ` on and `diag(σᵢ²t − c)` above its block diagonal; `c = E[I]` centres.
  /// Carbonell, Jimenez & Pedroso (2008), J. Comput. Appl. Math. 213(1), 300–305.
  fn integrated_variance_moments(&self, t: f64, c: f64, order: usize) -> [f64; 4] {
    let (m, blocks) = (self.vols.len(), order + 1);
    let mut block = Array2::<Complex64>::zeros((blocks * m, blocks * m));
    for i in 0..m {
      let above = self.vols[i] * self.vols[i] * t - c;
      for d in 0..blocks {
        for j in 0..m {
          block[[d * m + i, d * m + j]] = Complex64::new(self.q_matrix[[i, j]] * t, 0.0);
        }
        if d < order {
          block[[d * m + i, (d + 1) * m + i]] = Complex64::new(above, 0.0);
        }
      }
    }
    let exp = matrix_exp_complex(&block);
    let row = self.initial_state;
    let mut moments = [0.0; 4];
    let mut factorial = 1.0;
    for k in 1..=order {
      factorial *= k as f64;
      moments[k - 1] = factorial * (0..m).map(|j| exp[[row, k * m + j]].re).sum::<f64>();
    }
    moments
  }
}

/// COS (Fourier-cosine expansion) pricer for European options under
/// regime-switching diffusion.
///
/// Source:
/// - Fang, F. & Oosterlee, C.W. (2008), "A Novel Pricing Method for
///   European Options Based on Fourier-Cosine Series Expansions"
/// - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)
#[derive(Debug, Clone)]
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
    let query = (s, k, r, q_div, t);
    self.price(query, q_matrix, vols, initial_state, OptionType::Call)
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
    let query = (s, k, r, q_div, t);
    self.price(query, q_matrix, vols, initial_state, OptionType::Put)
  }

  fn price(
    &self,
    (s, k, r, q_div, t): (f64, f64, f64, f64, f64),
    q_matrix: &Array2<f64>,
    vols: &Array1<f64>,
    initial_state: usize,
    option_type: OptionType,
  ) -> f64 {
    let model = RegimeSwitchingModel {
      q_matrix: q_matrix.clone(),
      vols: vols.clone(),
      initial_state,
    };
    CosEngine {
      n: self.n,
      l: self.l,
    }
    .price(&model, s, k, r, q_div, t, option_type)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  const TOL_FOURIER: f64 = 0.15;

  #[test]
  fn rs_single_regime_matches_bsm() {
    use super::super::fourier::BsmFourier;
    use super::super::fourier::LewisPricer;

    let sigma = 0.2;
    let r = 0.05;
    let q = 0.01;

    let rs_model = RegimeSwitchingModel {
      q_matrix: array![[0.0]],
      vols: array![sigma],
      initial_state: 0,
    };

    let bsm_model = BsmFourier { sigma };

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
      q_matrix: array![[-1.0, 0.5, 0.5], [0.3, -0.6, 0.3], [0.4, 0.4, -0.8],],
      vols: array![0.10, 0.25, 0.50],
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
      initial_state: 0,
    };

    let lewis_price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let cm_price = CarrMadanPricer::default().price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);

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
    q_mat[[1, 1]] = Complex64::new(-t, 0.0);
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
      initial_state: 0,
    };
    let lewis_call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);

    assert!(
      (cos_call - lewis_call).abs() < 0.5,
      "COS vs Lewis single-regime: cos={cos_call}, lewis={lewis_call}"
    );
  }

  /// At `t = 0` the density degenerates: call and put are NaN, the put not floored to `0.0`.
  #[test]
  fn cos_degenerate_maturity_prices_nan() {
    let pricer = CosPricer::default();
    let q_mat = array![[-0.5, 0.5], [1.0, -1.0]];
    let vols = array![0.15, 0.35];
    let call = pricer.price_call(100.0, 100.0, 0.05, 0.01, 0.0, &q_mat, &vols, 0);
    let put = pricer.price_put(100.0, 100.0, 0.05, 0.01, 0.0, &q_mat, &vols, 0);
    assert!(call.is_nan() && put.is_nan(), "call {call}, put {put}");
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
