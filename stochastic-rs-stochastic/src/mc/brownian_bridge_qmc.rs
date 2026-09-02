//! # Brownian-bridge quasi-Monte Carlo paths
//!
//! Brownian motion sampled from a Sobol sequence through the Brownian-bridge
//! construction: the first, best-equidistributed Sobol coordinate sets the
//! terminal value $W_T$, the next one the midpoint $W_{T/2}$ conditional on
//! both ends, and so on, filling the path by recursive bisection. Given the
//! known values at $t_l < t_r$, the bridge point at $t_j$ is
//!
//! $$
//! W_{t_j} = \frac{(t_r - t_j)\,W_{t_l} + (t_j - t_l)\,W_{t_r}}{t_r - t_l} +
//!   \sqrt{\frac{(t_j - t_l)(t_r - t_j)}{t_r - t_l}}\;\Phi^{-1}(u_k),
//! $$
//!
//! so the coarse, high-variance features of the path absorb the sequence's
//! leading dimensions and the fine increments its trailing ones — the
//! effective-dimension reduction that makes QMC pay off for path-dependent
//! payoffs (Caflisch–Morokoff–Owen 1997; Glasserman 2003, §3.1).
//!
//! References:
//! - Caflisch, R.E., Morokoff, W., Owen, A.B. (1997), "Valuation of
//!   mortgage-backed securities using Brownian bridges to reduce effective
//!   dimension", *Journal of Computational Finance* 1(1), 27-46.
//!   DOI: 10.21314/JCF.1997.005
//! - Glasserman, P. (2003), *Monte Carlo Methods in Financial Engineering*,
//!   Springer, §3.1. DOI: 10.1007/978-0-387-21617-1

use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::special::ndtri;

use super::sobol::SobolSeq;
use crate::traits::FloatExt;

/// Bisection order of the bridge: the index list `(target, left, right)`
/// where `left`/`right` are the already-filled neighbours (`None` for the
/// origin at time zero); the first entry is the endpoint.
fn bridge_schedule(steps: usize) -> Vec<(usize, Option<usize>, usize)> {
  let mut schedule = Vec::with_capacity(steps);
  schedule.push((steps - 1, None, steps - 1));
  let mut queue = std::collections::VecDeque::new();
  // Half-open bracket (left_time_index, right_step_index): left is a time
  // index where `None` stands for t = 0, right a filled step.
  queue.push_back((None, steps - 1));
  while let Some((left, right)) = queue.pop_front() {
    let left_time = left.map_or(0, |l| l + 1);
    let right_time = right + 1;
    if right_time - left_time < 2 {
      continue;
    }
    let mid_time = (left_time + right_time) / 2;
    let mid = mid_time - 1;
    schedule.push((mid, left, right));
    queue.push_back((left, mid));
    queue.push_back((Some(mid), right));
  }
  schedule
}

/// Sobol-driven Brownian paths on an equispaced grid $t_i = iT/m$,
/// $i = 1, \ldots, m$, built by Brownian-bridge bisection.
#[derive(Debug, Clone)]
pub struct BrownianBridgeQmc {
  steps: usize,
  horizon: f64,
  sequence: SobolSeq,
  schedule: Vec<(usize, Option<usize>, usize)>,
}

impl BrownianBridgeQmc {
  /// Unscrambled Sobol driver for `steps` time steps up to `horizon`.
  pub fn new(steps: usize, horizon: f64) -> Self {
    assert!(steps >= 1, "steps must be at least 1");
    Self::with_sequence(steps, horizon, SobolSeq::new(steps))
  }

  /// Owen-scrambled Sobol driver seeded from `seed`, for randomised-QMC
  /// replications.
  pub fn scrambled<S: SeedExt>(steps: usize, horizon: f64, seed: &S) -> Self {
    assert!(steps >= 1, "steps must be at least 1");
    Self::with_sequence(steps, horizon, SobolSeq::scrambled(steps, seed))
  }

  fn with_sequence(steps: usize, horizon: f64, sequence: SobolSeq) -> Self {
    assert!(steps >= 1, "steps must be at least 1");
    assert!(horizon > 0.0, "horizon must be positive");
    assert_eq!(
      sequence.n_dims(),
      steps,
      "the sequence must have one dimension per step"
    );
    Self {
      steps,
      horizon,
      sequence,
      schedule: bridge_schedule(steps),
    }
  }

  /// Number of time steps.
  pub fn steps(&self) -> usize {
    self.steps
  }

  /// Horizon $T$.
  pub fn horizon(&self) -> f64 {
    self.horizon
  }

  /// One path from a row of uniforms (`steps` coordinates) into `out`
  /// (`steps` Brownian levels, $W_{t_1}, \ldots, W_{t_m}$).
  pub fn path_from_uniforms(&self, uniforms: &[f64], out: &mut [f64]) {
    assert_eq!(uniforms.len(), self.steps);
    assert_eq!(out.len(), self.steps);
    let dt = self.horizon / self.steps as f64;
    for (k, &(target, left, right)) in self.schedule.iter().enumerate() {
      let z = ndtri(uniforms[k].clamp(1e-300, 1.0 - 1e-16));
      let t_j = (target + 1) as f64 * dt;
      let (t_l, w_l) = match left {
        None => (0.0, 0.0),
        Some(l) => ((l + 1) as f64 * dt, out[l]),
      };
      if k == 0 {
        out[target] = (t_j - t_l).sqrt() * z;
        continue;
      }
      let t_r = (right + 1) as f64 * dt;
      let w_r = out[right];
      let mean = ((t_r - t_j) * w_l + (t_j - t_l) * w_r) / (t_r - t_l);
      let var = (t_j - t_l) * (t_r - t_j) / (t_r - t_l);
      out[target] = mean + var.sqrt() * z;
    }
  }

  /// `n_paths` Brownian paths as an `(n_paths, steps)` array of levels
  /// $W_{t_1}, \ldots, W_{t_m}$.
  pub fn paths<T: FloatExt>(&self, n_paths: usize) -> Array2<T> {
    let uniforms: Array2<f64> = self.sequence.sample(n_paths);
    let mut out = Array2::<T>::zeros((n_paths, self.steps));
    let mut row = vec![0.0; self.steps];
    for i in 0..n_paths {
      let u = uniforms.row(i);
      self.path_from_uniforms(u.as_slice().expect("contiguous row"), &mut row);
      for j in 0..self.steps {
        out[[i, j]] = T::from_f64_fast(row[j]);
      }
    }
    out
  }

  /// `n_paths` paths of Brownian increments $\Delta W_i = W_{t_i} - W_{t_{i-1}}$.
  pub fn increments<T: FloatExt>(&self, n_paths: usize) -> Array2<T> {
    let mut levels = self.paths::<T>(n_paths);
    for i in 0..n_paths {
      for j in (1..self.steps).rev() {
        let prev = levels[[i, j - 1]];
        levels[[i, j]] -= prev;
      }
    }
    levels
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn schedule_fills_the_endpoint_first_then_bisects() {
    let s = bridge_schedule(8);
    assert_eq!(s.len(), 8);
    assert_eq!(s[0], (7, None, 7));
    assert_eq!(s[1], (3, None, 7));
    assert!(s[2..4].contains(&(1, None, 3)) && s[2..4].contains(&(5, Some(3), 7)));
    let mut targets: Vec<usize> = s.iter().map(|e| e.0).collect();
    targets.sort_unstable();
    assert_eq!(targets, (0..8).collect::<Vec<_>>());
    assert_eq!(bridge_schedule(1), vec![(0, None, 0)]);
    let odd = bridge_schedule(5);
    let mut t: Vec<usize> = odd.iter().map(|e| e.0).collect();
    t.sort_unstable();
    assert_eq!(t, vec![0, 1, 2, 3, 4]);
  }

  /// With 2^12 scrambled Sobol paths the sample covariance of the bridge
  /// levels reproduces $\mathrm{Cov}(W_s, W_t) = \min(s, t)$.
  #[test]
  fn bridge_levels_have_brownian_covariance() {
    let steps = 8;
    let horizon = 2.0;
    let qmc = BrownianBridgeQmc::scrambled(steps, horizon, &Deterministic::new(3));
    let n = 4096;
    let w: Array2<f64> = qmc.paths(n);
    let dt = horizon / steps as f64;
    for a in 0..steps {
      let mean_a: f64 = w.column(a).sum() / n as f64;
      assert!(mean_a.abs() < 0.03, "mean at step {a}: {mean_a}");
      for b in a..steps {
        let cov: f64 = (0..n).map(|i| w[[i, a]] * w[[i, b]]).sum::<f64>() / n as f64;
        let want = (a + 1) as f64 * dt;
        assert!((cov - want).abs() < 0.04, "cov({a},{b}) = {cov} vs {want}");
      }
    }
  }

  /// The endpoint is the first Sobol coordinate through Φ⁻¹, and the
  /// increments telescope back to the levels.
  #[test]
  fn endpoint_and_increments_are_consistent() {
    let qmc = BrownianBridgeQmc::new(4, 1.0);
    let u: Array2<f64> = SobolSeq::new(4).sample(5);
    let w: Array2<f64> = qmc.paths(5);
    let dw: Array2<f64> = qmc.increments(5);
    for i in 0..5 {
      assert!((w[[i, 3]] - ndtri(u[[i, 0]])).abs() < 1e-12);
      let mut level = 0.0;
      for j in 0..4 {
        level += dw[[i, j]];
        assert!((level - w[[i, j]]).abs() < 1e-12);
      }
    }
  }

  #[test]
  #[should_panic(expected = "steps must be at least 1")]
  fn rejects_zero_steps() {
    let _ = BrownianBridgeQmc::new(0, 1.0);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBrownianBridgeQmc {
  inner: BrownianBridgeQmc,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBrownianBridgeQmc {
  /// Sobol-driven Brownian paths on `steps` equispaced steps to `horizon`,
  /// built by Brownian-bridge bisection; a `seed` switches on the Owen-type
  /// scramble of the underlying sequence.
  #[new]
  #[pyo3(signature = (steps, horizon, seed=None))]
  fn new(steps: usize, horizon: f64, seed: Option<u64>) -> Self {
    let inner = match seed {
      Some(s) => BrownianBridgeQmc::scrambled(
        steps,
        horizon,
        &stochastic_rs_core::simd_rng::Deterministic::new(s),
      ),
      None => BrownianBridgeQmc::new(steps, horizon),
    };
    Self { inner }
  }

  /// `(n_paths, steps)` array of Brownian levels `W_{t_1} … W_{t_m}`.
  fn paths<'py>(
    &self,
    py: pyo3::Python<'py>,
    n_paths: usize,
  ) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.paths::<f64>(n_paths).into_pyarray(py)
  }

  /// `(n_paths, steps)` array of Brownian increments.
  fn increments<'py>(
    &self,
    py: pyo3::Python<'py>,
    n_paths: usize,
  ) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.increments::<f64>(n_paths).into_pyarray(py)
  }

  #[getter]
  fn steps(&self) -> usize {
    self.inner.steps()
  }

  #[getter]
  fn horizon(&self) -> f64 {
    self.inner.horizon()
  }
}
