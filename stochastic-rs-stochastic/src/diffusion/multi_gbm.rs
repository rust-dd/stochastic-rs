//! # MultiGbm
//!
//! $$
//! dS^{(i)}_t = \mu_i S^{(i)}_t\,dt + \sigma_i S^{(i)}_t\,dW^{(i)}_t,\qquad
//! d\langle W^{(i)}, W^{(j)}\rangle_t = \rho_{ij}\,dt,
//! $$
//!
//! `k` geometric Brownian motions driven by the correlated Brownian motion
//! of [`crate::noise::mcgns::Mcgns`], stepped by the exact log-Euler
//! scheme $S^{(i)}_{t+\Delta t} = S^{(i)}_t\exp\bigl((\mu_i - \tfrac12\sigma_i^2)\Delta t + \sigma_i\,\Delta W^{(i)}_t\bigr)$,
//! which is distributionally exact on the grid. The `k × n` output holds
//! one asset per row, starting from `x0` at time zero.
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial
//! Engineering*, Springer, §3.2.3 (multidimensional geometric Brownian
//! motion). DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::mcgns::Mcgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct MultiGbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drifts $\mu_i$, one per asset.
  pub mu: Array1<T>,
  /// Volatilities $\sigma_i$, one per asset.
  pub sigma: Array1<T>,
  /// Instantaneous correlation matrix ρ of the driving Brownian motions.
  pub rho: Array2<T>,
  /// Number of grid points per path, the first being `x0`.
  pub n: usize,
  /// Initial levels $S^{(i)}_0$, one per asset.
  pub x0: Array1<T>,
  /// Simulation horizon [0, t] (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  driver: Mcgns<T, S>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> MultiGbm<T, S> {
  /// `mu`, `sigma` and `x0` must share the length `k` of `rho`'s side; `n`
  /// counts grid points including the initial one.
  pub fn new(
    mu: Array1<T>,
    sigma: Array1<T>,
    rho: Array2<T>,
    n: usize,
    x0: Array1<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let k = rho.nrows();
    assert!(k >= 1, "need at least one asset");
    assert_eq!(mu.len(), k, "mu must have one entry per asset");
    assert_eq!(sigma.len(), k, "sigma must have one entry per asset");
    assert_eq!(x0.len(), k, "x0 must have one entry per asset");
    assert!(n >= 1, "n must be at least 1");
    assert!(
      sigma.iter().all(|s| *s >= T::zero()),
      "volatilities must be non-negative"
    );
    let driver = Mcgns::new(rho.clone(), n.saturating_sub(1), t, seed.clone());
    Self {
      backend: Cpu,
      mu,
      sigma,
      rho,
      n,
      x0,
      t,
      seed,
      driver,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> MultiGbm<T, S, B> {
  /// Number of assets `k`.
  pub fn assets(&self) -> usize {
    self.rho.nrows()
  }

  fn fill_paths<S2: SeedExt>(&self, seed: &S2, out: &mut Array2<T>) {
    let k = self.assets();
    assert_eq!(out.dim(), (k, self.n), "output must be k × n");
    for i in 0..k {
      out[(i, 0)] = self.x0[i];
    }
    if self.n == 1 {
      return;
    }
    let steps = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(steps);
    let mut dw = Array2::<T>::zeros((k, steps));
    self.driver.fill_increments(seed, &mut dw);
    for i in 0..k {
      let drift = (self.mu[i] - self.sigma[i] * self.sigma[i] / T::from_f64_fast(2.0)) * dt;
      for j in 1..self.n {
        out[(i, j)] = out[(i, j - 1)] * (drift + self.sigma[i] * dw[(i, j - 1)]).exp();
      }
    }
  }
}

backend_switch!([T: FloatExt, S: SeedExt] MultiGbm<T, S> { mu, sigma, rho, n, x0, t, seed, driver } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for MultiGbm<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = MultiGbmSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> MultiGbmSampler<T, S> {
    let seed = self.seed.derive();
    MultiGbmSampler {
      process: MultiGbm::new(
        self.mu.clone(),
        self.sigma.clone(),
        self.rho.clone(),
        self.n,
        self.x0.clone(),
        self.t,
        seed,
      ),
    }
  }
}

/// Reusable [`MultiGbm`] sampling state with a derived seed, so parallel
/// chunks draw independent paths.
#[doc(hidden)]
pub struct MultiGbmSampler<T: FloatExt, S: SeedExt> {
  process: MultiGbm<T, S>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for MultiGbmSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    // Same by-reference seed as `McgnsSampler`: the derived seed advances
    // on every call, so the paths of one chunk are independent.
    self.process.fill_paths(&self.process.seed, out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((self.process.assets(), self.process.n));
    self.sample_into(&mut out);
    out
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Terminal log-returns carry the exact mean $(\mu - \sigma^2/2)t$, the
  /// variance $\sigma^2 t$ and the cross-correlation ρ.
  #[test]
  fn terminal_log_returns_match_the_multivariate_lognormal_law() {
    let mu = array![0.05_f64, 0.02];
    let sigma = array![0.2_f64, 0.3];
    let rho = array![[1.0_f64, -0.4], [-0.4, 1.0]];
    let process = MultiGbm::new(
      mu.clone(),
      sigma.clone(),
      rho.clone(),
      51,
      array![100.0, 50.0],
      Some(2.0),
      Deterministic::new(5),
    );
    let paths = process.sample_par(20_000);
    let logs: Vec<[f64; 2]> = paths
      .iter()
      .map(|p| [(p[(0, 50)] / 100.0).ln(), (p[(1, 50)] / 50.0).ln()])
      .collect();
    let n = logs.len() as f64;
    let mean0 = logs.iter().map(|l| l[0]).sum::<f64>() / n;
    let mean1 = logs.iter().map(|l| l[1]).sum::<f64>() / n;
    let var0 = logs.iter().map(|l| (l[0] - mean0).powi(2)).sum::<f64>() / n;
    let var1 = logs.iter().map(|l| (l[1] - mean1).powi(2)).sum::<f64>() / n;
    let cov = logs
      .iter()
      .map(|l| (l[0] - mean0) * (l[1] - mean1))
      .sum::<f64>()
      / n;
    assert!((mean0 - (0.05 - 0.02) * 2.0).abs() < 0.01, "mean0 {mean0}");
    assert!(
      (mean1 - (0.02 - 0.045) * 2.0).abs() < 0.012,
      "mean1 {mean1}"
    );
    assert!((var0 - 0.08).abs() < 0.004, "var0 {var0}");
    assert!((var1 - 0.18).abs() < 0.008, "var1 {var1}");
    assert!(
      (cov / (var0 * var1).sqrt() + 0.4).abs() < 0.03,
      "corr {}",
      cov / (var0 * var1).sqrt()
    );
    assert!(
      paths
        .iter()
        .all(|p| p[(0, 0)] == 100.0 && p[(1, 0)] == 50.0)
    );
  }

  #[test]
  fn single_asset_is_a_plain_gbm_in_law() {
    let process = MultiGbm::new(
      array![0.1_f64],
      array![0.25],
      array![[1.0_f64]],
      11,
      array![1.0],
      Some(1.0),
      Deterministic::new(9),
    );
    let paths = process.sample_par(5_000);
    let mean = paths.iter().map(|p| p[(0, 10)]).sum::<f64>() / 5_000.0;
    assert!((mean - 0.1_f64.exp()).abs() < 0.02, "E[S_T] = {mean}");
  }

  /// The derived seed travels by reference, so two consecutive paths of one
  /// sampler differ instead of replaying the first.
  #[test]
  fn consecutive_paths_of_one_sampler_differ() {
    let process = MultiGbm::new(
      array![0.0_f64],
      array![0.2],
      array![[1.0_f64]],
      8,
      array![1.0],
      None,
      Deterministic::new(11),
    );
    let mut sampler = process.sampler();
    let a = sampler.sample();
    let b = sampler.sample();
    assert_ne!(a, b);
  }

  #[test]
  fn deterministic_seed_reproduces() {
    let a = MultiGbm::new(
      array![0.0_f64, 0.0],
      array![0.2, 0.2],
      array![[1.0_f64, 0.5], [0.5, 1.0]],
      8,
      array![1.0, 1.0],
      None,
      Deterministic::new(7),
    )
    .sample();
    let b = MultiGbm::new(
      array![0.0_f64, 0.0],
      array![0.2, 0.2],
      array![[1.0_f64, 0.5], [0.5, 1.0]],
      8,
      array![1.0, 1.0],
      None,
      Deterministic::new(7),
    )
    .sample();
    assert_eq!(a, b);
  }

  #[test]
  #[should_panic(expected = "sigma must have one entry per asset")]
  fn rejects_mismatched_parameter_lengths() {
    let _ = MultiGbm::new(
      array![0.0_f64, 0.0],
      array![0.2],
      array![[1.0_f64, 0.5], [0.5, 1.0]],
      8,
      array![1.0, 1.0],
      None,
      Unseeded,
    );
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyMultiGbm {
  inner: Option<MultiGbm<f64>>,
  seeded: Option<MultiGbm<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyMultiGbm {
  /// Correlated multi-asset GBM: `mu`, `sigma`, `x0` per asset, `rho` the
  /// `k × k` correlation matrix, `n` grid points over `[0, t]`.
  #[new]
  #[pyo3(signature = (mu, sigma, rho, n, x0, t=None, seed=None))]
  fn new(
    mu: Vec<f64>,
    sigma: Vec<f64>,
    rho: numpy::PyReadonlyArray2<'_, f64>,
    n: usize,
    x0: Vec<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    let rho = rho.as_array().to_owned();
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(MultiGbm::new(
          Array1::from_vec(mu),
          Array1::from_vec(sigma),
          rho,
          n,
          Array1::from_vec(x0),
          t,
          stochastic_rs_core::simd_rng::Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(MultiGbm::new(
          Array1::from_vec(mu),
          Array1::from_vec(sigma),
          rho,
          n,
          Array1::from_vec(x0),
          t,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  /// One `(k, n)` path matrix.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// `m` independent path matrices as a list of `(k, n)` arrays.
  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| {
      let samples = inner.sample_par(m);
      pyo3::types::PyList::new(
        py,
        samples
          .iter()
          .map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
      )
      .unwrap()
      .into_py_any(py)
      .unwrap()
    })
  }
}
