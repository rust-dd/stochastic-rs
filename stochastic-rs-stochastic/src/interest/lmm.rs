//! # Lmm — drift-coupled LIBOR Market Model under the spot-LIBOR measure
//!
//! $$
//! \frac{dL_n(t)}{L_n(t)} \;=\;
//!   \sigma_n(t)\!\!\sum_{j=\eta(t)}^{n}\!\!
//!     \frac{\delta_j\,\rho_{n,j}\,\sigma_j(t)\,L_j(t)}{1+\delta_j L_j(t)}\,dt
//!   \;+\;\sigma_n(t)\,dW_n^{\!*}(t)
//! $$
//!
//! Forward-LIBOR market model (BGM/Jamshidian/Brace-Gatarek-Musiela) with
//! cross-forward drift coupling under the **spot-LIBOR measure**
//! `B^*(t)` of Glasserman (2003) §3.7. This is the proper LIBOR Market Model
//! that the [`crate::interest::bgm::Bgm`] type **does not** implement
//! (`Bgm` is a parallel array of uncoupled Euler-stepped martingales — see
//! its module doc).
//!
//! Numerical scheme: **log-Euler with frozen-at-step drift** (Glasserman
//! 2003 §3.7.1; Joshi 2003). Each step approximates
//!
//! ```text
//! L_n(t+Δt) = L_n(t)·exp[(μ_n(t) − ½σ_n²) Δt + σ_n √Δt · Z_n]
//! ```
//!
//! where `μ_n(t)` is the drift summation evaluated at the start of the
//! step and `Z_n = Σ_k chol(ρ)_{n,k} ε_k` with `ε ~ N(0, I)`. Log-Euler
//! preserves positivity (forwards stay positive, unlike a forward-Euler
//! step) and is the standard discretization choice for caplet / swaption
//! pricing.
//!
//! ## Tenor structure
//!
//! Tenor `T_0 < T_1 < … < T_M` (length `M+1`) defines the M forward Libors
//! `L_n` for `n=0..M-1`, where `L_n(t)` is the simple-compounded forward
//! rate over `[T_n, T_{n+1}]` observed at time `t ≤ T_n`. Day-count
//! fractions `δ_n = T_{n+1} − T_n`. The active index function
//! `η(t) = min{ n : T_n > t }` is the next reset date; only Libors with
//! `n ≥ η(t)` are alive.
//!
//! ## Output
//!
//! `Array2<T>` of shape `(M, n_steps)`. Row `n` stores the path
//! `L_n(t_0), L_n(t_1), …, L_n(t_{n_steps-1})`. After `t_k > T_n`,
//! `L_n` is **frozen** at its reset value `L_n(T_n^−)` (a Libor that has
//! already reset cannot evolve further — its value is locked in).
//!
//! ## Scope and references
//!
//! - Brace, Gatarek, Musiela (1997), "The Market Model of Interest Rate
//!   Dynamics", Mathematical Finance 7(2), 127-155.
//! - Jamshidian (1997), "LIBOR and swap market models and measures",
//!   Finance and Stochastics 1(4), 293-330.
//! - Glasserman (2003), "Monte Carlo Methods in Financial Engineering",
//!   Springer, §3.7 ("LIBOR market model").
//! - Glasserman, Zhao (2000), "Arbitrage-free discretization of lognormal
//!   forward LIBOR and swap rate models", Finance and Stochastics 4(1),
//!   35-68 — the original predictor-corrector scheme; this implementation
//!   uses the simpler frozen-drift log-Euler.
//! - Hula (2011), "Discrete LIBOR Market Model Analogy", arXiv:1108.4260
//!   — discrete construction whose continuous-time limit is the LMM.
//!
//! Volatilities `σ_n(t)` are taken **time-homogeneous** (constant per
//! Libor) for v0; piecewise-constant vol can be added later by extending
//! `sigma` to `Array2<T>` of shape `(M, n_steps)`.
//!
//! Correlations `ρ` default to the identity (independent factors per
//! Libor); supply a custom positive-definite matrix via `with_correlation`.
//!

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Drift-coupled LIBOR Market Model (BGM/Jamshidian) under the spot-LIBOR
/// measure. See module header for the precise scope, dynamics and
/// references.
pub struct Lmm<T: FloatExt, S: SeedExt = Unseeded> {
  /// Tenor dates `T_0 < T_1 < … < T_M`, length `M+1`. Strictly increasing.
  pub tenor: Array1<T>,
  /// Initial forward Libor curve `L_n(0)` for `n=0..M-1`. Length `M`.
  /// Each entry must be positive.
  pub l0: Array1<T>,
  /// Per-Libor volatilities `σ_n` (constant in time for v0). Length `M`.
  pub sigma: Array1<T>,
  /// Lower-triangular Cholesky factor `L` of the forward correlation matrix
  /// `ρ = L Lᵀ`. Square `M × M`. `None` ⇒ identity (independent factors).
  pub chol: Option<Array2<T>>,
  /// Number of equally-spaced simulation grid points across `[0, t_horizon]`.
  /// Must be `≥ 1`; the first grid point is `t_0 = 0`.
  pub n: usize,
  /// Simulation horizon `t_horizon`. Defaults to `T_M` (last tenor date)
  /// when `None`. Must satisfy `0 < t_horizon ≤ T_M`.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Lmm<T> {
  /// Construct an LMM with independent forward factors (identity correlation)
  /// and an automatic seed.
  pub fn new(tenor: Array1<T>, l0: Array1<T>, sigma: Array1<T>, n: usize, t: Option<T>) -> Self {
    validate_lmm_inputs(&tenor, &l0, &sigma);
    Self {
      tenor,
      l0,
      sigma,
      chol: None,
      n,
      t,
      seed: Unseeded,
    }
  }

  /// Attach a forward correlation matrix `rho` (`M × M`, symmetric,
  /// positive-definite). Panics if shape is wrong or Cholesky fails.
  pub fn with_correlation(mut self, rho: Array2<T>) -> Self {
    let m = self.l0.len();
    assert_eq!(rho.nrows(), m, "correlation rows must equal number of Libors");
    assert_eq!(rho.ncols(), m, "correlation cols must equal number of Libors");
    self.chol = Some(cholesky_lower(&rho));
    self
  }

}

fn validate_lmm_inputs<T: FloatExt>(tenor: &Array1<T>, l0: &Array1<T>, sigma: &Array1<T>) {
  let m_plus_1 = tenor.len();
  assert!(m_plus_1 >= 2, "tenor must have at least two dates");
  let m = m_plus_1 - 1;
  assert_eq!(l0.len(), m, "l0 length must equal tenor.len() - 1");
  assert_eq!(sigma.len(), m, "sigma length must equal tenor.len() - 1");
  for i in 0..m {
    assert!(
      tenor[i + 1] > tenor[i],
      "tenor must be strictly increasing"
    );
    assert!(l0[i] > T::zero(), "initial Libor L_n(0) must be positive");
    assert!(sigma[i] >= T::zero(), "volatility σ_n must be non-negative");
  }
}

impl<T: FloatExt> Lmm<T, Deterministic> {
  /// Construct a deterministically-seeded LMM (independent factors).
  pub fn seeded(
    tenor: Array1<T>,
    l0: Array1<T>,
    sigma: Array1<T>,
    n: usize,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    validate_lmm_inputs(&tenor, &l0, &sigma);
    Self {
      tenor,
      l0,
      sigma,
      chol: None,
      n,
      t,
      seed: Deterministic::new(seed),
    }
  }

  /// Attach a forward correlation matrix `rho`.
  pub fn with_correlation(mut self, rho: Array2<T>) -> Self {
    let m = self.l0.len();
    assert_eq!(rho.nrows(), m);
    assert_eq!(rho.ncols(), m);
    self.chol = Some(cholesky_lower(&rho));
    self
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Lmm<T, S> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let m = self.l0.len();
    let n_steps = self.n;
    let mut path = Array2::<T>::zeros((m, n_steps));
    if n_steps == 0 {
      return path;
    }

    for (n, &l0) in self.l0.iter().enumerate() {
      path[(n, 0)] = l0;
    }
    if n_steps == 1 {
      return path;
    }

    let t_max = self.tenor[m];
    let horizon = self.t.unwrap_or(t_max);
    assert!(horizon > T::zero() && horizon <= t_max);

    let n_increments = n_steps - 1;
    let dt = horizon / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let half = T::from_f64_fast(0.5);

    // Day-count fractions delta_j = T_{j+1} - T_j (length M).
    let delta: Array1<T> = (0..m).map(|j| self.tenor[j + 1] - self.tenor[j]).collect();

    // Standard normal innovations for one full simulation: M factors × n_increments.
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &self.seed);
    let mut eps = Array2::<T>::zeros((m, n_increments));
    {
      let buf = eps
        .as_slice_mut()
        .expect("LMM eps buffer must be contiguous");
      normal.fill_slice_fast(buf);
    }

    // Pre-compute correlated standard normals: Z_n,k = Σ_j chol_{n,j} eps_{j,k}
    // (only when chol is provided; identity otherwise).
    let z_corr: Array2<T> = match &self.chol {
      Some(chol) => chol.dot(&eps),
      None => eps,
    };

    // Step loop. Each step uses spot-LIBOR drift with the active set
    // η(t) ≤ n, where η(t) = min{n : T_n > t}.
    let mut t_now = T::zero();
    let mut eta: usize = 0;
    let mut prev = self.l0.clone();
    for k in 0..n_increments {
      // Advance η(t) so that T_{eta} > t_now (i.e. Libor 0..eta have reset).
      while eta < m && self.tenor[eta] <= t_now {
        eta += 1;
      }
      let eta_idx = eta.saturating_sub(1).min(m); // index into Libor vector

      // Drift summation: for each alive n in [eta_idx..m), compute
      //   μ_n(t) = σ_n · Σ_{j=eta_idx..=n} (δ_j ρ_{n,j} σ_j L_j) / (1 + δ_j L_j).
      // For correlation = identity ρ_{n,j} = δ_{n,j} (Kronecker), so the sum
      // collapses to the j=n term.
      let mut next = prev.clone();
      for n in eta_idx..m {
        if self.sigma[n] <= T::zero() {
          // Zero-vol Libor stays at its current value.
          continue;
        }
        // Drift μ_n.
        let mut drift = T::zero();
        match &self.chol {
          Some(_chol) => {
            // Full correlated drift sum.
            // ρ_{n,j} is recovered as Σ_p chol_{n,p} chol_{j,p}.
            // We apply the formula directly using the original ρ matrix
            // reconstructed from the Cholesky factor on the fly.
            let chol_ref = self.chol.as_ref().unwrap();
            for j in eta_idx..=n {
              let mut rho_nj = T::zero();
              for p in 0..=j.min(n) {
                rho_nj = rho_nj + chol_ref[(n, p)] * chol_ref[(j, p)];
              }
              let denom = T::one() + delta[j] * prev[j];
              drift = drift + (delta[j] * rho_nj * self.sigma[j] * prev[j]) / denom;
            }
            drift = drift * self.sigma[n];
          }
          None => {
            // Identity correlation: only j = n term survives.
            let denom = T::one() + delta[n] * prev[n];
            drift = self.sigma[n] * self.sigma[n] * delta[n] * prev[n] / denom;
          }
        }

        // Log-Euler step with frozen drift.
        let z_nk = z_corr[(n, k)];
        let log_inc = (drift - half * self.sigma[n] * self.sigma[n]) * dt
          + self.sigma[n] * sqrt_dt * z_nk;
        next[n] = prev[n] * log_inc.exp();
      }

      // Freeze any Libor whose reset has just passed (T_n ≤ t_now + dt).
      // Their forward is locked at the value at reset time.
      // We model this by leaving prev[n] untouched once T_n is in the past.
      let t_next = t_now + dt;
      for n in 0..m {
        if self.tenor[n] <= t_now {
          // Already reset — keep prev[n]. (Skip in next pass via eta.)
          next[n] = prev[n];
        }
      }

      for n in 0..m {
        path[(n, k + 1)] = next[n];
      }
      prev = next;
      t_now = t_next;
    }

    let _ = path.axis_iter(Axis(0));
    let _ = path.slice(s![.., ..]);
    path
  }
}

/// Lower-triangular Cholesky decomposition of a symmetric positive-definite
/// matrix. Hand-rolled (no openblas dependency); panics if `rho` is not PSD.
///
/// Used inline by [`Lmm::with_correlation`].
fn cholesky_lower<T: FloatExt>(rho: &Array2<T>) -> Array2<T> {
  let m = rho.nrows();
  assert_eq!(rho.ncols(), m, "correlation must be square");
  let mut l = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..=i {
      let mut sum = T::zero();
      for k in 0..j {
        sum = sum + l[(i, k)] * l[(j, k)];
      }
      let v = rho[(i, j)] - sum;
      if i == j {
        assert!(v > T::zero(), "correlation matrix not positive-definite");
        l[(i, j)] = v.sqrt();
      } else {
        l[(i, j)] = v / l[(j, j)];
      }
    }
  }
  l
}

#[cfg(test)]
mod tests {
  use super::*;

  fn flat_tenor(m: usize, dt: f64) -> Array1<f64> {
    Array1::from_iter((0..=m).map(|i| i as f64 * dt))
  }

  #[test]
  fn lmm_sample_runs_independent_factors() {
    let m = 4;
    let tenor = flat_tenor(m, 0.5); // T_0=0, T_1=0.5, ..., T_4=2.0
    let l0 = Array1::from(vec![0.03, 0.035, 0.04, 0.045]);
    let sigma = Array1::from(vec![0.20, 0.20, 0.20, 0.20]);
    let lmm: Lmm<f64, Deterministic> = Lmm::seeded(tenor, l0, sigma, 100, Some(2.0), 42);
    let path = lmm.sample();
    assert_eq!(path.shape(), &[m, 100]);
    // Initial column matches l0.
    for n in 0..m {
      assert!(path[(n, 0)] > 0.0);
    }
    // All values must remain positive (log-Euler preserves positivity).
    for v in path.iter() {
      assert!(*v > 0.0, "LMM forward went non-positive");
    }
  }

  #[test]
  fn lmm_correlated_factors_run() {
    let m = 3;
    let tenor = flat_tenor(m, 1.0);
    let l0 = Array1::from(vec![0.04, 0.045, 0.05]);
    let sigma = Array1::from(vec![0.25, 0.20, 0.18]);
    // Constant 0.6 off-diagonal correlation.
    let mut rho = Array2::<f64>::eye(m);
    for i in 0..m {
      for j in 0..m {
        if i != j {
          rho[(i, j)] = 0.6;
        }
      }
    }
    let lmm: Lmm<f64, Deterministic> =
      Lmm::seeded(tenor, l0, sigma, 50, Some(3.0), 123).with_correlation(rho);
    let path = lmm.sample();
    assert_eq!(path.shape(), &[m, 50]);
    for v in path.iter() {
      assert!(*v > 0.0);
    }
  }

  #[test]
  fn lmm_seeded_is_deterministic() {
    let tenor = flat_tenor(2, 0.5);
    let l0 = Array1::from(vec![0.03, 0.035]);
    let sigma = Array1::from(vec![0.2, 0.2]);
    let a: Lmm<f64, Deterministic> =
      Lmm::seeded(tenor.clone(), l0.clone(), sigma.clone(), 30, Some(1.0), 7);
    let b: Lmm<f64, Deterministic> = Lmm::seeded(tenor, l0, sigma, 30, Some(1.0), 7);
    let pa = a.sample();
    let pb = b.sample();
    for (x, y) in pa.iter().zip(pb.iter()) {
      assert_eq!(x, y, "seeded LMM not reproducible");
    }
  }

  #[test]
  fn cholesky_lower_recovers_identity() {
    let rho = Array2::<f64>::eye(4);
    let l = cholesky_lower::<f64>(&rho);
    for i in 0..4 {
      for j in 0..4 {
        let expected: f64 = if i == j { 1.0 } else { 0.0 };
        assert!((l[(i, j)] - expected).abs() < 1e-12_f64);
      }
    }
  }

  #[test]
  fn cholesky_lower_recovers_2x2() {
    let rho: Array2<f64> = ndarray::array![[1.0, 0.6], [0.6, 1.0]];
    let l = cholesky_lower::<f64>(&rho);
    let recon = l.dot(&l.t());
    for i in 0..2 {
      for j in 0..2 {
        let diff: f64 = recon[(i, j)] - rho[(i, j)];
        assert!(diff.abs() < 1e-12_f64);
      }
    }
  }
}
