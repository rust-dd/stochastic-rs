//! # Jump fOU
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma dB_t^H+dJ_t
//! $$
//!
//! Composition of a fractional Ornstein-Uhlenbeck diffusion (Cheridito,
//! Kawaguchi, Maejima (2003), *Fractional Ornstein-Uhlenbeck Processes*,
//! Electronic Journal of Probability 8, paper 3, 1–14,
//! DOI: 10.1214/EJP.v8-125) with an additive, independent
//! compound-Poisson jump term `dJ_t` in the style of Merton (1976) —
//! *Option Pricing When Underlying Stock Returns Are Discontinuous*,
//! Journal of Financial Economics 3(1-2), 125–144,
//! DOI: 10.1016/0304-405X(76)90022-2. This exact combination (fOU base
//! plus an independent jump driver) is this crate's own composition
//! rather than a single named model from one paper.
//!
use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::process::cpoisson::CompoundPoisson;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fully seed-reproducible: no exception to [`ProcessExt`]'s reproducibility
/// guarantee. Both halves consult `self.seed`, independently:
///
/// - The private `fgn: Fgn<T, Unseeded, B>` field is never consulted for its
///   own seed — `sampler()` builds a plain [`SimdNormal`] from
///   `self.seed.derive()` and borrows `fgn` only for its `Arc`-shared FFT
///   plan and eigenvalues, the same pattern
///   [`JumpFOUCustom`](crate::jump::jump_fou_custom::JumpFOUCustom) and
///   [`Fbm`](crate::process::fbm::Fbm) use for their own embedded,
///   permanently-`Unseeded` `fgn`. (This was fixable non-breakingly because
///   the field is private — rewiring how it's used carries no signature
///   change.)
/// - `cpoisson` is built internally by [`new`](Self::new) from `seed`,
///   exactly like [`Merton`](crate::jump::merton::Merton)'s field of the
///   same name — see that field's own doc below.
///
/// (This type was previously documented as a *full* exception — "no
/// randomness derives from `self.seed` at all" — on the grounds that both
/// halves were hard-wired away from it. That was correct about the values at
/// the time, but not about what was fixable: `fgn` needed only a private,
/// non-breaking rewire (done first); `cpoisson` needed the same breaking
/// widening `Merton`/`Kou`/`LevyDiffusion`/`Bates1996` needed, applied here
/// last.)
pub struct JumpFou<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Hurst exponent H of the driving fractional Gaussian noise (roughness
  /// / long-memory of the diffusion part; H = 0.5 recovers a standard
  /// OU-with-jumps).
  pub hurst: T,
  /// Mean-reversion speed (κ in the module header's `dX_t=κ(θ−X_t)dt+...`).
  /// Multiplies `(mu - X_t)`, despite the field's own name.
  pub theta: T,
  /// Long-run mean level (θ in the module header). The level `X` reverts
  /// to between jumps.
  pub mu: T,
  /// Diffusion scale for the fractional-Gaussian-noise term (σ in the
  /// module header).
  pub sigma: T,
  /// Number of points sampled along the fOU-plus-jumps path.
  pub n: usize,
  /// Initial value X₀ of the fOU-plus-jumps path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Jump (Poisson) intensity λ — arrival rate of the jumps added to the
  /// fOU path. Single source of truth: `sampler()` reads this field
  /// directly (not `cpoisson.poisson.lambda`) for the jump-arrival rate.
  /// `JumpFou` has no `with_*` builder setters, so unlike
  /// [`Bates1996`](crate::jump::bates::Bates1996) or
  /// [`Merton`](crate::jump::merton::Merton) there is no setter that could
  /// let this drift out of sync with the mirror — [`new`](Self::new)
  /// establishes the invariant once, at construction.
  pub lambda: T,
  /// Compound-Poisson jump driver adding `dJ_t` on top of the fOU path.
  /// Fully seed-reproducible: [`new`](Self::new) builds it internally from
  /// `seed` (`seed.clone().derive()` — a hash-mixed child, decorrelated
  /// from but a deterministic function of the same `seed` the diffusion
  /// component consults directly), and `sampler()` derives a fresh,
  /// chunk-local basis off `self.cpoisson.seed` for every chunk, mirroring
  /// the diffusion component's own per-chunk `self.seed`-derived basis.
  ///
  /// `sampler()` reads only `cpoisson.distribution` (the jump-size law)
  /// and `self.lambda` — **not** `cpoisson.poisson.lambda` — from this
  /// field on the sampling path; `cpoisson.poisson.{n,t_max,seed}` are
  /// inert there (`grid_increments` never consults them). That inertness
  /// is scoped to *this type's own* sampling, though: `cpoisson` is a
  /// `CompoundPoisson` in its own right, and calling `.sample()` on it
  /// directly (bypassing `JumpFou` entirely) drives it through
  /// `Poisson::sample_impl`, which *does* branch on `.n`/`.t_max` and
  /// *does* consult `.seed` — genuinely live there. Left `pub` for both
  /// reasons, matching [`Merton::cpoisson`](crate::jump::merton::Merton::cpoisson):
  /// a caller can inspect or directly `.sample()` the embedded
  /// compound-Poisson process as its own standalone `ProcessExt`, and can
  /// replace it via direct field assignment — which does not adopt the
  /// replacement's `lambda` into `self.lambda` (there is no `with_cpoisson`
  /// setter here to do that adoption for you, unlike `Merton`/`Bates1996`),
  /// so assign `self.lambda` to match separately if you do this.
  pub cpoisson: CompoundPoisson<T, D, S>,
  fgn: Fgn<T, Unseeded, B>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`). Consulted
  /// directly by the diffusion component; `cpoisson`'s own seed (set at
  /// construction from this same value — see `cpoisson`'s doc above)
  /// drives the jump component.
  pub seed: S,
}

impl<T, D, S: SeedExt> JumpFou<T, D, S, Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Builds the compound-Poisson jump driver internally from `jump_dist`
  /// and `lambda`, seeded from `seed` (see `cpoisson`'s field doc) — the
  /// caller supplies the jump-size distribution and intensity directly
  /// instead of pre-building a `Poisson`/`CompoundPoisson` pair and
  /// threading a third, independent seed through it by hand.
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    lambda: T,
    jump_dist: D,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");

    let cpoisson = CompoundPoisson::new(
      jump_dist,
      Poisson::new(lambda, Some(n), t, Unseeded),
      seed.clone().derive(),
    );

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      lambda,
      cpoisson,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B: FgnBackend> ProcessExt<T> for JumpFou<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = JumpFouSampler<'s, T, D, S, B>
  where
    Self: 's;

  /// Owns a Gaussian source derived from `self.seed` (not `fgn.sampler()`,
  /// which would build it from `fgn`'s own permanently-`Unseeded` field —
  /// see the type doc) and borrows `fgn` (for its `Arc`-shared FFT
  /// plan/eigenvalues only). Also owns a separate, independently-derived
  /// jump seed and borrows only the jump-size distribution — never a
  /// borrowed `&self.cpoisson` shared across chunks, which would let
  /// concurrent chunks race on the same shared atomic during the parallel
  /// region (see `ProcessExt`'s trait-level reproducibility requirement).
  fn sampler(&self) -> JumpFouSampler<'_, T, D, S, B> {
    JumpFouSampler {
      n: self.n,
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt: self.fgn.dt(),
      fgn: &self.fgn,
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed.derive()),
      jump_distribution: &self.cpoisson.distribution,
      lambda: self.lambda,
      jump_seed: self.cpoisson.seed.derive(),
    }
  }
}

/// Reusable [`JumpFou`] sampling state: borrows `fgn` for its `Arc`-shared
/// FFT plan/eigenvalues and the jump-size distribution, and owns the
/// Gaussian source and a separately-derived jump seed, so a Monte-Carlo loop
/// pays the fGn `SimdNormal` setup once and reuses the FFT plan.
#[doc(hidden)]
pub struct JumpFouSampler<'a, T, D, S: SeedExt, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend,
{
  n: usize,
  theta: T,
  mu: T,
  sigma: T,
  x0: T,
  dt: T,
  fgn: &'a Fgn<T, Unseeded, B>,
  normal: SimdNormal<T>,
  jump_distribution: &'a D,
  lambda: T,
  jump_seed: S,
}

impl<T, D, S: SeedExt, B> JumpFouSampler<'_, T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let mut fgn = Array1::<T>::zeros(self.fgn.out_len);
    self
      .fgn
      .fill_cpu(&mut self.normal, fgn.as_slice_mut().unwrap());
    let jump_increments = crate::process::cpoisson::grid_increments(
      self.jump_distribution,
      self.lambda,
      &self.jump_seed,
      out.len(),
      self.dt,
    );

    out[0] = self.x0;

    for i in 1..out.len() {
      out[i] = out[i - 1]
        + self.theta * (self.mu - out[i - 1]) * self.dt
        + self.sigma * fgn[i - 1]
        + jump_increments[i];
    }
  }
}

impl<T, D, S: SeedExt, B> PathSampler<T> for JumpFouSampler<'_, T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("JumpFou output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

backend_switch!([T, D, S: SeedExt] JumpFou<T, D, S> { hurst, theta, mu, sigma, n, x0, t, lambda, cpoisson, seed } via fgn
  where T: FloatExt, D: Distribution<T> + Send + Sync);

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyJumpFou {
  inner_f32: Option<JumpFou<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<JumpFou<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32:
    Option<JumpFou<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64:
    Option<JumpFou<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyJumpFou {
  #[new]
  #[pyo3(signature = (hurst, theta, mu, sigma, distribution, lambda_, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: usize,
    x0: Option<f64>,
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
    match dtype.unwrap_or("f64") {
      "f32" => {
        let jump_dist = crate::traits::CallableDist::new(distribution);
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(JumpFou::new(
              hurst as f32,
              theta as f32,
              mu as f32,
              sigma as f32,
              lambda_ as f32,
              jump_dist,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(JumpFou::new(
              hurst as f32,
              theta as f32,
              mu as f32,
              sigma as f32,
              lambda_ as f32,
              jump_dist,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let jump_dist = crate::traits::CallableDist::new(distribution);
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(JumpFou::new(
              hurst,
              theta,
              mu,
              sigma,
              lambda_,
              jump_dist,
              n,
              x0,
              t,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(JumpFou::new(
              hurst, theta, mu, sigma, lambda_, jump_dist, n, x0, t, Unseeded,
            ));
          }
        }
      }
    }
    s
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    })
  }
}
