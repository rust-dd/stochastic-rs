//! # Regime-Switching Diffusion
//!
//! $$
//! dS_t = (r-q)S_t\,dt + \sigma_{Z_t}\,S_t\,dW_t
//! $$
//!
//! where $Z_t$ is a continuous-time Markov chain with generator matrix $Q$
//! governing transitions between $M$ regimes, each with its own volatility $\sigma_i$.
//!
//! Source:
//! - Kirkby, J.L. (PROJ_Option_Pricing_Matlab)
//! - Hamilton, J.D. (1989), "A New Approach to the Economic Analysis
//!   of Nonstationary Time Series and the Business Cycle"
//!
use ndarray::{Array1, Array2};

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Regime-switching GBM process.
///
/// The stock follows GBM in each regime with a different volatility.
/// Regime transitions are governed by a CTMC generator matrix Q.
pub struct RegimeSwitchingDiffusion<T: FloatExt, S: SeedExt = Unseeded> {
  /// Drift rate (e.g. r - q).
  pub mu: T,
  /// CTMC generator matrix (M×M, rows sum to 0).
  pub q_matrix: Array2<T>,
  /// Per-regime volatilities.
  pub vols: Array1<T>,
  /// Initial regime state (0-indexed).
  pub initial_state: usize,
  /// Number of time steps.
  pub n: usize,
  /// Initial stock price.
  pub s0: Option<T>,
  /// Total simulation horizon.
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt> RegimeSwitchingDiffusion<T> {
  pub fn new(
    mu: T,
    q_matrix: Array2<T>,
    vols: Array1<T>,
    initial_state: usize,
    n: usize,
    s0: Option<T>,
    t: Option<T>,
  ) -> Self {
    let m = vols.len();
    assert_eq!(q_matrix.nrows(), m, "Q must be M×M");
    assert_eq!(q_matrix.ncols(), m, "Q must be M×M");
    assert!(initial_state < m, "initial_state must be < M");

    Self {
      mu,
      q_matrix,
      vols,
      initial_state,
      n,
      s0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> RegimeSwitchingDiffusion<T, Deterministic> {
  pub fn seeded(
    mu: T,
    q_matrix: Array2<T>,
    vols: Array1<T>,
    initial_state: usize,
    n: usize,
    s0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    let m = vols.len();
    assert_eq!(q_matrix.nrows(), m);
    assert_eq!(q_matrix.ncols(), m);
    assert!(initial_state < m);

    Self {
      mu,
      q_matrix,
      vols,
      initial_state,
      n,
      s0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> RegimeSwitchingDiffusion<T, S> {
  fn transition_prob_matrix(&self, dt: T) -> Array2<T> {
    let m = self.vols.len();
    let mut a = Array2::<T>::zeros((m, m));
    for i in 0..m {
      for j in 0..m {
        a[[i, j]] = self.q_matrix[[i, j]] * dt;
      }
    }
    matrix_exp_real(&a)
  }

  fn sample_next_regime<R: rand::Rng + ?Sized>(
    &self,
    current: usize,
    p_matrix: &Array2<T>,
    rng: &mut R,
  ) -> usize {
    let u: f64 = rng.random();
    let m = self.vols.len();
    let mut cum = T::zero();
    for j in 0..m {
      cum = cum + p_matrix[[current, j]];
      if T::from_f64_fast(u) <= cum {
        return j;
      }
    }
    m - 1
  }
}

fn matrix_exp_real<T: FloatExt>(a: &Array2<T>) -> Array2<T> {
  let m = a.nrows();

  let mut norm = T::zero();
  for i in 0..m {
    let mut row_sum = T::zero();
    for j in 0..m {
      row_sum = row_sum + a[[i, j]].abs();
    }
    if row_sum > norm {
      norm = row_sum;
    }
  }

  let s = if norm > T::zero() {
    (norm.to_f64().unwrap().log2().ceil() as usize).saturating_add(1)
  } else {
    0
  };
  let scale = T::from_f64_fast((2.0_f64).powi(s as i32));

  let mut scaled = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..m {
      scaled[[i, j]] = a[[i, j]] / scale;
    }
  }

  let mut result = Array2::<T>::zeros((m, m));
  for i in 0..m {
    result[[i, i]] = T::one();
  }

  let mut term = Array2::<T>::zeros((m, m));
  for i in 0..m {
    term[[i, i]] = T::one();
  }

  for k in 1..=20 {
    term = mat_mul(&term, &scaled);
    let divisor = T::from_usize_(k);
    for i in 0..m {
      for j in 0..m {
        term[[i, j]] = term[[i, j]] / divisor;
        result[[i, j]] = result[[i, j]] + term[[i, j]];
      }
    }
  }

  for _ in 0..s {
    result = mat_mul(&result, &result);
  }

  result
}

fn mat_mul<T: FloatExt>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
  let m = a.nrows();
  let mut c = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..m {
      let mut sum = T::zero();
      for k in 0..m {
        sum = sum + a[[i, k]] * b[[k, j]];
      }
      c[[i, j]] = sum;
    }
  }
  c
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for RegimeSwitchingDiffusion<T, S> {
  /// Returns [stock_prices, regime_states] where regime_states are cast to T.
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let mut s_path = Array1::<T>::zeros(self.n);
    let mut z_path = Array1::<T>::zeros(self.n);

    if self.n == 0 {
      return [s_path, z_path];
    }

    let s0 = self.s0.unwrap_or(T::one());
    s_path[0] = s0;
    z_path[0] = T::from_usize_(self.initial_state);

    if self.n <= 1 {
      return [s_path, z_path];
    }

    let n_inc = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_inc);
    let sqrt_dt = dt.sqrt();

    let p_matrix = self.transition_prob_matrix(dt);

    let mut dw = Array1::<T>::zeros(n_inc);
    let dw_slice = dw.as_slice_mut().unwrap();
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(dw_slice);

    let mut rng = seed.rng();
    let mut state = self.initial_state;

    for i in 1..self.n {
      let sigma = self.vols[state];
      let half = T::from_f64_fast(0.5);

      let log_inc = (self.mu - half * sigma * sigma) * dt + sigma * dw[i - 1];
      s_path[i] = s_path[i - 1] * log_inc.exp();

      state = self.sample_next_regime(state, &p_matrix, &mut rng);
      z_path[i] = T::from_usize_(state);
    }

    [s_path, z_path]
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn q3() -> Array2<f64> {
    array![
      [-1.0, 0.5, 0.5],
      [0.5, -1.0, 0.5],
      [0.5, 0.5, -1.0],
    ]
  }

  #[test]
  fn price_stays_positive() {
    let p = RegimeSwitchingDiffusion::new(
      0.05,
      q3(),
      array![0.15, 0.25, 0.35],
      0,
      1000,
      Some(100.0),
      Some(1.0),
    );
    let [s, _z] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  #[test]
  fn regime_states_valid() {
    let p = RegimeSwitchingDiffusion::new(
      0.05,
      q3(),
      array![0.15, 0.25, 0.35],
      0,
      1000,
      Some(100.0),
      Some(1.0),
    );
    let [_s, z] = p.sample();
    for &state in z.iter() {
      assert!(state >= 0.0 && state < 3.0, "invalid state={state}");
    }
  }

  #[test]
  fn single_regime_like_gbm() {
    let q1 = array![[0.0]];
    let p = RegimeSwitchingDiffusion::seeded(
      0.05,
      q1,
      array![0.2],
      0,
      1000,
      Some(100.0),
      Some(1.0),
      42,
    );
    let [s, z] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
    assert!(z.iter().all(|x| (*x - 0.0_f64).abs() < 1e-10));
  }

  #[test]
  fn seeded_is_deterministic() {
    let p1 = RegimeSwitchingDiffusion::seeded(
      0.05,
      q3(),
      array![0.15, 0.25, 0.35],
      0,
      100,
      Some(100.0),
      Some(1.0),
      42,
    );
    let p2 = RegimeSwitchingDiffusion::seeded(
      0.05,
      q3(),
      array![0.15, 0.25, 0.35],
      0,
      100,
      Some(100.0),
      Some(1.0),
      42,
    );
    let [s1, z1] = p1.sample();
    let [s2, z2] = p2.sample();
    assert_eq!(s1, s2);
    assert_eq!(z1, z2);
  }

  #[test]
  fn matrix_exp_identity() {
    let zero = Array2::<f64>::zeros((3, 3));
    let result = matrix_exp_real(&zero);
    for i in 0..3 {
      for j in 0..3 {
        let expected = if i == j { 1.0 } else { 0.0 };
        assert!(
          (result[[i, j]] - expected).abs() < 1e-10,
          "exp(0)[{i}][{j}]={}, expected {expected}",
          result[[i, j]]
        );
      }
    }
  }

  #[test]
  fn transition_probs_sum_to_one() {
    let p = RegimeSwitchingDiffusion::new(
      0.05,
      q3(),
      array![0.15, 0.25, 0.35],
      0,
      10,
      Some(100.0),
      Some(1.0),
    );
    let pm = p.transition_prob_matrix(0.01_f64);
    for i in 0..3 {
      let sum: f64 = (0..3).map(|j| pm[[i, j]]).sum();
      assert!((sum - 1.0).abs() < 1e-10, "row sum={sum}");
    }
  }
}
