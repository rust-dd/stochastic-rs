//! # Rbergomi
//!
//! $$
//! dS_t = r S_t\,dt + S_t\sqrt{v_t}\,dW^1_t,\qquad
//! v_t = v_0^2\exp\!\Bigl(\nu W^H_t - \tfrac12\nu^2 t^{2H}\Bigr),\qquad
//! W^H_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW^2_s
//! $$
//!
//! Rough Bergomi of Bayer, Friz & Gatheral (2016) with a flat forward
//! variance curve $\xi_0(t) = v_0^2$. The variance driver $W^H$ is the
//! Riemann–Liouville (Volterra) fractional Brownian motion, simulated with
//! the hybrid scheme of Bennedsen, Lunde & Pakkanen (2017) at $\kappa = 1$:
//! on the grid $t_i = i\Delta$ the Volterra process is
//!
//! $$
//! X_{t_i} = \int_{t_{i-1}}^{t_i} (t_i-s)^{\alpha}\,dW^2_s
//!         + \sum_{k=2}^{i} \bigl(b_k^*\Delta\bigr)^{\alpha}\,
//!           \bigl(W^2_{t_{i-k+1}} - W^2_{t_{i-k}}\bigr),\qquad
//! \alpha = H - \tfrac12,\quad
//! b_k^* = \Bigl(\frac{k^{\alpha+1}-(k-1)^{\alpha+1}}{\alpha+1}\Bigr)^{1/\alpha},
//! $$
//!
//! and $W^H_{t_i} = \sqrt{2H}\,X_{t_i}$. The first term is drawn exactly:
//! it is jointly Gaussian with the increment $\Delta W^2_i$, with
//! $\operatorname{Cov} = \Delta^{\alpha+1}/(\alpha+1)$ and variance
//! $\Delta^{2\alpha+1}/(2\alpha+1)$, so it is the increment's regression
//! plus an independent residual. The remaining sum uses the kernel at the
//! optimal discretisation points $b_k^*$ (their Proposition 2.8), which
//! makes the scheme's covariance error second order in $\Delta$. The
//! scheme reproduces $\operatorname{Var}(W^H_t) = t^{2H}$, hence the
//! $\tfrac12\nu^2 t^{2H}$ compensator keeps $\mathbb E(v_t) = v_0^2$, and
//! reduces to the plain Brownian recipe at $H = \tfrac12$. The price uses
//! Euler steps on $S$ with $dW^1 = \rho\,dW^2 + \sqrt{1-\rho^2}\,dW^\perp$.
//!
//! The convolution costs $O(n^2)$ per path, the same order as the
//! Markov-lift alternatives at small $n$; for long grids see
//! [`crate::rough::MarkovLift`] (exponential-sum kernel, $O(nN')$) and
//! [`crate::rough::rl_heston::RlHeston`].
//!
//! References: Bayer, C., Friz, P. & Gatheral, J. (2016), *Pricing under
//! rough volatility*, Quantitative Finance 16(6), 887–904; Bennedsen, M.,
//! Lunde, A. & Pakkanen, M. S. (2017), *Hybrid scheme for Brownian
//! semistationary processes*, Finance and Stochastics 21, 931–965.

use std::marker::PhantomData;

use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct RoughBergomi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent `H ∈ (0, ½]` of the Volterra fractional Brownian
  /// driver, simulated with the hybrid scheme (see the module doc); `H = ½`
  /// is the Brownian case.
  pub hurst: T,
  /// Vol-of-vol ν scaling the log-variance driver's dispersion.
  pub nu: T,
  /// Initial variance level v₀ (variance, not volatility — squared inside
  /// the sampler to seed `v(t_0) = v_0²`).
  pub v0: Option<T>,
  /// Initial asset price S₀.
  pub s0: Option<T>,
  /// Constant proportional drift rate of the asset (a GBM-style drift, not
  /// a correlation — that role belongs to `rho`).
  pub r: T,
  /// Correlation ρ between the asset's Brownian shock and the
  /// log-variance driver's innovations.
  pub rho: T,
  /// Number of points sampled along the rBergomi path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `RoughBergomi::default().with_hurst(0.3).with_rho(-0.2)`.
impl<T: FloatExt, S: SeedExt> RoughBergomi<T, S> {
  pub fn new(
    hurst: T,
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    RoughBergomi {
      backend: PhantomData,
      hurst,
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> RoughBergomi<T, S, B> {
  /// Replace `hurst`, all else unchanged.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    self.hurst = hurst;
    self
  }

  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    self.nu = nu;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    self.v0 = v0;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `r`, all else unchanged.
  pub fn with_r(mut self, r: T) -> Self {
    self.r = r;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// H=0.1, ν=0.4, v₀=0.2, s₀=100, r=0.01, ρ=-0.6 — a textbook Rough Bergomi
/// parameterization. t=1, n=252 — one trading year of daily steps (this
/// crate's `Default` convention).
impl<T: FloatExt> Default for RoughBergomi<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.1),
      T::from_f64_fast(0.4),
      Some(T::from_f64_fast(0.2)),
      Some(T::from_f64_fast(100.0)),
      T::from_f64_fast(0.01),
      T::from_f64_fast(-0.6),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] RoughBergomi<T, S> { hurst, nu, v0, s0, r, rho, n, t, seed, cgns } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for RoughBergomi<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = RoughBergomiSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> RoughBergomiSampler<T, S> {
    let dt = self.cgns.dt();
    let (kernel, integral_on_increment, integral_residual_sd) =
      RoughBergomiSampler::<T, S>::hybrid_weights(self.hurst, dt, self.n);
    RoughBergomiSampler {
      n: self.n,
      hurst: self.hurst,
      nu: self.nu,
      v0_sq: self.v0.unwrap_or(T::one()).powi(2),
      s0: self.s0.unwrap_or(T::from_usize_(100)),
      r: self.r,
      dt,
      kernel,
      integral_on_increment,
      integral_residual_sd,
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`RoughBergomi`] sampling state: owns the correlated-Gaussian
/// generator, the hybrid-scheme kernel weights and the seed source so a
/// Monte-Carlo loop reuses both output buffers and the noise setup.
#[doc(hidden)]
pub struct RoughBergomiSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  hurst: T,
  nu: T,
  v0_sq: T,
  s0: T,
  r: T,
  dt: T,
  /// `g(b_k^* Δ)` for `k = 2..n`, stored at index `k`, index 0 and 1 unused.
  kernel: Vec<T>,
  /// Regression coefficient of the exact last-interval integral on the
  /// Brownian increment, `Δ^α / (α + 1)`.
  integral_on_increment: T,
  /// Conditional standard deviation of that integral,
  /// `Δ^{α + 1/2} √(1/(2α+1) − 1/(α+1)²)`.
  integral_residual_sd: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> RoughBergomiSampler<T, S> {
  /// Bennedsen–Lunde–Pakkanen hybrid scheme with `κ = 1` (2017, eq. (3.6)):
  /// the last interval's kernel integral is drawn exactly, jointly Gaussian
  /// with the Brownian increment; earlier intervals use the kernel at the
  /// optimal discretisation points `b_k^*` (Proposition 2.8).
  pub(crate) fn hybrid_weights(hurst: T, dt: T, n: usize) -> (Vec<T>, T, T) {
    let alpha = hurst - T::from_f64_fast(0.5);
    let one = T::one();
    let mut kernel = vec![T::zero(); n.max(2)];
    let dt_alpha = dt.powf(alpha);
    for (k, w) in kernel.iter_mut().enumerate().skip(2) {
      let k_t = T::from_usize_(k);
      *w = dt_alpha * (k_t.powf(alpha + one) - (k_t - one).powf(alpha + one)) / (alpha + one);
    }
    let on_increment = dt_alpha / (alpha + one);
    let residual_var = one / (T::from_usize_(2) * alpha + one) - one / (alpha + one).powi(2);
    let residual_sd = dt.powf(alpha + T::from_f64_fast(0.5)) * residual_var.max(T::zero()).sqrt();
    (kernel, on_increment, residual_sd)
  }

  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, z] = &self.cgns.sample_impl(&self.seed);
    let steps = self.n - 1;
    let mut eps = vec![T::zero(); steps];
    if steps > 0 {
      stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), T::one(), &self.seed)
        .fill_slice(&mut eps);
    }
    let sqrt_2h = (T::from_usize_(2) * self.hurst).sqrt();
    let two_h = T::from_usize_(2) * self.hurst;

    s[0] = self.s0;
    v2[0] = self.v0_sq;

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      // Volterra process X_{t_i} = Σ_k kernel-weighted increments; the k = 1
      // term is the exact integral over the last interval.
      let mut x = self.integral_on_increment * z[i - 1] + self.integral_residual_sd * eps[i - 1];
      for k in 2..=i {
        x += self.kernel[k] * z[i - k];
      }
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0_sq
        * (self.nu * sqrt_2h * x - T::from_f64_fast(0.5) * self.nu.powi(2) * t.powf(two_h)).exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for RoughBergomiSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("RoughBergomi output must be contiguous"),
      v2.as_slice_mut()
        .expect("RoughBergomi output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    [s, v2]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyRoughBergomi {
  inner_f32: Option<RoughBergomi<f32>>,
  inner_f64: Option<RoughBergomi<f64>>,
  seeded_f32: Option<RoughBergomi<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<RoughBergomi<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyRoughBergomi {
  #[new]
  #[pyo3(signature = (hurst, nu, r, rho, n, v0=None, s0=None, t=None, seed=None, dtype=None))]
  fn new(
    hurst: f64,
    nu: f64,
    r: f64,
    rho: f64,
    n: usize,
    v0: Option<f64>,
    s0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match (seed, dtype.unwrap_or("f64")) {
      (Some(sd), "f32") => {
        s.seeded_f32 = Some(RoughBergomi::new(
          hurst as f32,
          nu as f32,
          v0.map(|v| v as f32),
          s0.map(|v| v as f32),
          r as f32,
          rho as f32,
          n,
          t.map(|v| v as f32),
          Deterministic::new(sd),
        ));
      }
      (Some(sd), _) => {
        s.seeded_f64 = Some(RoughBergomi::new(
          hurst,
          nu,
          v0,
          s0,
          r,
          rho,
          n,
          t,
          Deterministic::new(sd),
        ));
      }
      (None, "f32") => {
        s.inner_f32 = Some(RoughBergomi::new(
          hurst as f32,
          nu as f32,
          v0.map(|v| v as f32),
          s0.map(|v| v as f32),
          r as f32,
          rho as f32,
          n,
          t.map(|v| v as f32),
          Unseeded,
        ));
      }
      (None, _) => {
        s.inner_f64 = Some(RoughBergomi::new(hurst, nu, v0, s0, r, rho, n, t, Unseeded));
      }
    }
    s
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::zeros((m, n));
      let mut r1 = Array2::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn model(hurst: f64, n: usize, seed: u64) -> RoughBergomi<f64, Deterministic> {
    RoughBergomi::new(
      hurst,
      0.4,
      Some(0.2),
      Some(100.0),
      0.0,
      -0.6,
      n,
      Some(1.0),
      Deterministic::new(seed),
    )
  }

  /// Bennedsen–Lunde–Pakkanen (2017) Proposition 2.8: the `k = 2` weight is
  /// `Δ^α (2^{α+1} − 1)/(α+1)`; at `H = ½` every weight is one and the exact
  /// last-interval integral collapses onto the increment.
  #[test]
  fn hybrid_weights_match_the_closed_forms() {
    let (kernel, on_increment, residual_sd) =
      RoughBergomiSampler::<f64, Deterministic>::hybrid_weights(0.1, 0.01, 6);
    let alpha = -0.4;
    let expected_k2 = 0.01_f64.powf(alpha) * (2.0_f64.powf(alpha + 1.0) - 1.0) / (alpha + 1.0);
    assert!((kernel[2] - expected_k2).abs() < 1e-12);
    assert!((on_increment - 0.01_f64.powf(alpha) / (alpha + 1.0)).abs() < 1e-12);
    let expected_var =
      0.01_f64.powf(2.0 * alpha + 1.0) * (1.0 / (2.0 * alpha + 1.0) - 1.0 / (alpha + 1.0).powi(2));
    assert!((residual_sd * residual_sd - expected_var).abs() < 1e-14);
    assert!(
      kernel[2] > kernel[3] && kernel[3] > kernel[4],
      "weights decay"
    );
    let (brownian, on_inc, sd) =
      RoughBergomiSampler::<f64, Deterministic>::hybrid_weights(0.5, 0.01, 6);
    assert!(brownian[2..].iter().all(|w| (w - 1.0).abs() < 1e-12));
    assert!((on_inc - 1.0).abs() < 1e-12 && sd.abs() < 1e-9);
  }

  /// `E[v_t] = v_0²` for every `t` (the `½ν²t^{2H}` compensator is exact
  /// when `Var[W^H_t] = t^{2H}`), and the sample variance of the implied
  /// `W^H_t` tracks `t^{2H}` — the scheme's variance error is second order.
  #[test]
  fn variance_driver_is_a_martingale_with_t_2h_variance() {
    let n = 65;
    let paths = 20_000;
    let m = model(0.1, n, 7);
    let v2 = m
      .sample_par(paths)
      .iter()
      .map(|[_, v2]| v2.clone())
      .collect::<Vec<_>>();
    let nu = 0.4;
    for i in [8usize, 32, 64] {
      let t = i as f64 / 64.0;
      let mean: f64 = v2.iter().map(|p| p[i]).sum::<f64>() / paths as f64 / 0.04;
      assert!((mean - 1.0).abs() < 0.04, "t {t}: E[v]/v0² = {mean}");
      let logs: Vec<f64> = v2
        .iter()
        .map(|p| ((p[i] / 0.04).ln() + 0.5 * nu * nu * t.powf(0.2)) / nu)
        .collect();
      let mean_w = logs.iter().sum::<f64>() / paths as f64;
      let var_w = logs.iter().map(|w| (w - mean_w).powi(2)).sum::<f64>() / paths as f64;
      assert!(
        (var_w / t.powf(0.2) - 1.0).abs() < 0.05,
        "t {t}: Var[W^H] = {var_w}, t^2H = {}",
        t.powf(0.2)
      );
    }
  }

  /// The lag-one autocorrelation of the log-variance increments is the
  /// fractional-Gaussian-noise value `½(2^{2H} − 2)` at `H = 0.1`, a
  /// negative number a driver with independent increments cannot produce.
  #[test]
  fn log_variance_increments_are_antipersistent() {
    let n = 257;
    let paths = 4_000;
    let m = model(0.1, n, 3);
    let (mut num, mut den) = (0.0, 0.0);
    for [_, v2] in m.sample_par(paths) {
      let w: Vec<f64> = v2.iter().map(|v| (v / 0.04).ln()).collect();
      let d: Vec<f64> = w.windows(2).map(|p| p[1] - p[0]).collect();
      for k in 128..d.len() - 1 {
        num += d[k] * d[k + 1];
        den += d[k] * d[k];
      }
    }
    let rho1 = num / den;
    let fgn = 0.5 * (2.0_f64.powf(0.2) - 2.0);
    assert!(
      rho1 < -0.3 && (rho1 - fgn).abs() < 0.08,
      "rho1 {rho1} vs fGN {fgn}"
    );
  }
}
