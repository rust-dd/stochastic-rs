//! # Bates
//!
//! $$
//! \begin{aligned}dS_t&=(r-r_f-\lambda k)S_tdt+\sqrt{v_t}S_t dW_t^S+(Y-1)S_{t^-}dN_t\\dv_t&=\kappa(\theta-v_t)dt+\sigma\sqrt{v_t}dW_t^v\end{aligned}
//! $$
//!
use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[inline]
fn validate_drift_args<T: FloatExt>(
  mu: Option<T>,
  b: Option<T>,
  r: Option<T>,
  r_f: Option<T>,
  type_name: &'static str,
) {
  let has_r_pair = r.is_some() && r_f.is_some();
  if !(has_r_pair || b.is_some() || mu.is_some()) {
    panic!("{type_name}: one of (r and r_f), b, or mu must be provided");
  }
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Bates1996::new(..).with_lambda(0.8).with_rho(-0.4)`.
pub struct Bates1996<T, D, S: SeedExt = Unseeded>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Direct drift rate μ of the asset — one of three mutually-exclusive
  /// drift specifications (`mu` xor `b` xor the `(r, r_f)` pair); exactly
  /// one must be `Some`.
  pub mu: Option<T>,
  /// Cost-of-carry rate b, an alternative drift specification to `mu`.
  pub b: Option<T>,
  /// Domestic risk-free rate; paired with `r_f` as a third drift
  /// specification via `r - r_f`.
  pub r: Option<T>,
  /// Foreign risk-free rate / dividend yield, paired with `r`.
  pub r_f: Option<T>,
  /// Jump (Poisson) intensity λ — arrival rate of the log-price jumps.
  pub lambda: T,
  /// Jump-size compensator κ (E[Y−1]-like term, matching the module
  /// header's own λκ_J), subtracted from the drift scaled by `lambda`.
  /// Unrelated to mean-reversion speed despite the letter k.
  pub k: T,
  /// Variance-drift intercept (κθ combined) in the reparametrized variance
  /// recursion `dv = (alpha − beta·v)dt + ...`, equivalent to `κ(θ−v)`
  /// with `alpha = κθ`.
  pub alpha: T,
  /// Variance-drift slope (mean-reversion speed κ) in the same
  /// reparametrized recursion.
  pub beta: T,
  /// Vol-of-vol σ scaling the variance factor's own diffusion.
  pub sigma: T,
  /// Instantaneous correlation ρ between the asset's and variance's
  /// driving Brownian motions.
  pub rho: T,
  /// Number of points sampled along the Bates path.
  pub n: usize,
  /// Initial asset price S₀.
  pub s0: Option<T>,
  /// Initial variance level v₀.
  pub v0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Reflect (true) instead of floor-at-zero (false/None) negative
  /// variance proposals.
  pub use_sym: Option<bool>,
  /// Correlated-Gaussian generator driving the price/variance diffusion.
  /// Constructed once (and rebuilt by `with_rho`/`with_steps`/`with_horizon`)
  /// with a `Cgns<T>` (`S = Unseeded`) that itself is never consulted — the
  /// sampler drives it via `cgns.sample_impl(&self.seed)` instead, so this
  /// field's own dead `Unseeded` is irrelevant to reproducibility. Private,
  /// so this indirection is an implementation detail.
  cgns: Cgns<T>,
  /// Compound-Poisson jump driver added to the asset's log-return.
  /// **Partial exception to [`ProcessExt`]'s reproducibility guarantee:**
  /// hard-wired to `Unseeded` (default `S`) by pre-existing design — the
  /// same shape as [`Merton`](crate::jump::merton::Merton)'s field of the
  /// same name — so the jump arrivals/sizes are never seed-reproducible
  /// even though the diffusion component above (driven by `cgns` and
  /// `self.seed`) is. See MIGRATION.md and [`ProcessExt`]'s trait-level
  /// reproducibility section.
  pub cpoisson: CompoundPoisson<T, D>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
}

impl<T, D, S: SeedExt> Bates1996<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    lambda: T,
    k: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    cpoisson: CompoundPoisson<T, D>,
    seed: S,
  ) -> Self {
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }
    validate_drift_args(mu, b, r, r_f, "Bates1996");

    Self {
      mu,
      b,
      r,
      r_f,
      lambda,
      k,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      use_sym,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      cpoisson,
      seed,
    }
  }

  /// Replace `mu`; re-validates that a drift specification still exists.
  pub fn with_mu(mut self, mu: Option<T>) -> Self {
    self.mu = mu;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `b`; re-validates that a drift specification still exists.
  pub fn with_b(mut self, b: Option<T>) -> Self {
    self.b = b;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `r`; re-validates that a drift specification still exists.
  pub fn with_r(mut self, r: Option<T>) -> Self {
    self.r = r;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `r_f`; re-validates that a drift specification still exists.
  pub fn with_r_f(mut self, r_f: Option<T>) -> Self {
    self.r_f = r_f;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self
  }

  /// Replace `k`, all else unchanged.
  pub fn with_k(mut self, k: T) -> Self {
    self.k = k;
    self
  }

  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    self.alpha = alpha;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    self.beta = beta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
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

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    if let Some(v) = v0 {
      assert!(v >= T::zero(), "v0 must be non-negative");
    }
    self.v0 = v0;
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
    self
  }

  /// Replace the compound-Poisson jump driver, all else unchanged.
  pub fn with_cpoisson(mut self, cpoisson: CompoundPoisson<T, D>) -> Self {
    self.cpoisson = cpoisson;
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

impl<T, D, S: SeedExt> Bates1996<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  #[inline]
  fn effective_drift(&self) -> T {
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => unreachable!("validate_drift_args ensures at least one of (r+r_f), b, mu is set"),
    }
  }
}

impl<T, D, S: SeedExt> ProcessExt<T> for Bates1996<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BatesSampler<'s, T, D, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler — the same
  /// shape `Cgns`'s own `sampler()` uses — so the correlated-Gaussian
  /// diffusion driver (`cgns`, otherwise permanently `Unseeded`; see the
  /// type doc) is driven via `sample_impl(&self.seed)` instead of a bare
  /// `.sample()` that only ever reads `cgns`'s own dead `Unseeded` field.
  /// Adjacent chunks land on hash-scrambled, mutually independent bases for
  /// the same reason every other `derive()`-based sampler in this crate
  /// does (see `ProcessExt`'s trait-level reproducibility section).
  fn sampler(&self) -> BatesSampler<'_, T, D, S> {
    BatesSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::zero()),
      v0: self.v0.unwrap_or(T::zero()).max(T::zero()),
      lambda: self.lambda,
      k: self.k,
      alpha: self.alpha,
      beta: self.beta,
      sigma: self.sigma,
      drift: self.effective_drift(),
      use_sym: self.use_sym.unwrap_or(false),
      dt: self.cgns.dt(),
      cgns: self.cgns,
      cpoisson: &self.cpoisson,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Bates1996`] sampling state: owns the correlated-Gaussian generator
/// and an owned, already-derived seed to drive it, and borrows the (non-`Clone`)
/// compound-Poisson driver so a Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct BatesSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  s0: T,
  v0: T,
  lambda: T,
  k: T,
  alpha: T,
  beta: T,
  sigma: T,
  drift: T,
  use_sym: bool,
  dt: T,
  cgns: Cgns<T>,
  cpoisson: &'a CompoundPoisson<T, D>,
  seed: S,
}

impl<T, D, S: SeedExt> BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);
    let jump_increments = self.cpoisson.sample_grid_relative_increments(self.n, dt);

    s[0] = self.s0;
    v[0] = self.v0;

    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1]
        + (self.drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v_prev.sqrt() * cgn1[i - 1]
        + s[i - 1] * jump_increments[i];

      let dv = (self.alpha - self.beta * v_prev) * dt + self.sigma * v_prev.sqrt() * cgn2[i - 1];

      v[i] = match self.use_sym {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }
  }
}

impl<T, D, S: SeedExt> PathSampler<T> for BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bates output must be contiguous"),
      v.as_slice_mut().expect("Bates output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBates {
  inner_f32: Option<Bates1996<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Bates1996<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32:
    Option<Bates1996<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64:
    Option<Bates1996<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBates {
  #[new]
  #[pyo3(signature = (lambda_, k, alpha, beta, sigma, rho, distribution, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, seed=None, dtype=None))]
  fn new(
    lambda_: f64,
    k: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    rho: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    use crate::process::poisson::Poisson;
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match dtype.unwrap_or("f64") {
      "f32" => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, Some(n), t.map(|v| v as f32), Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(Bates1996::new(
              mu.map(|v| v as f32),
              b.map(|v| v as f32),
              r.map(|v| v as f32),
              r_f.map(|v| v as f32),
              lambda_ as f32,
              k as f32,
              alpha as f32,
              beta as f32,
              sigma as f32,
              rho as f32,
              n,
              s0.map(|v| v as f32),
              v0.map(|v| v as f32),
              t.map(|v| v as f32),
              use_sym,
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(Bates1996::new(
              mu.map(|v| v as f32),
              b.map(|v| v as f32),
              r.map(|v| v as f32),
              r_f.map(|v| v as f32),
              lambda_ as f32,
              k as f32,
              alpha as f32,
              beta as f32,
              sigma as f32,
              rho as f32,
              n,
              s0.map(|v| v as f32),
              v0.map(|v| v as f32),
              t.map(|v| v as f32),
              use_sym,
              cpoisson,
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, Some(n), t, Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(Bates1996::new(
              mu,
              b,
              r,
              r_f,
              lambda_,
              k,
              alpha,
              beta,
              sigma,
              rho,
              n,
              s0,
              v0,
              t,
              use_sym,
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(Bates1996::new(
              mu, b, r, r_f, lambda_, k, alpha, beta, sigma, rho, n, s0, v0, t, use_sym, cpoisson,
              Unseeded,
            ));
          }
        }
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

// Split out to keep this file under the project's 600-line cap (this type
// now carries a full set of `with_*` builder setters on top of the model
// itself). Same pattern as `volatility/bates_svj.rs`.
#[cfg(test)]
#[path = "bates_tests.rs"]
mod tests;
