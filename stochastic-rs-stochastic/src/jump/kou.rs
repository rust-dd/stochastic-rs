//! # Kou
//!
//! $$
//! \log Y\sim p\,\mathrm{Exp}(\eta_1)-(1-p)\,\mathrm{Exp}(\eta_2),\quad dS/S=\cdots+d\left(\sum(J-1)\right)
//! $$
//!
use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::process::cpoisson::CompoundPoisson;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Kou process
///
/// <https://www.columbia.edu/~sk75/MagSci02.pdf>
///
/// **No `Default` impl.** `KouSampler::fill_path` is driven entirely by the
/// generic jump-size distribution `D` — it is line-for-line the same
/// recursion [`crate::jump::merton::MertonSampler::fill_path`] runs, so `D`
/// is the *only* thing that makes a `Kou` a Kou rather than a Merton in
/// this crate. This model's own definition (module doc above) is a
/// double-exponential log-jump `log Y ~ p·Exp(η₁) − (1−p)·Exp(η₂)`, and the
/// crate does not yet ship an asymmetric-double-exponential distribution
/// type (`stochastic_rs_distributions::scalar` has `ScalarNormal` /
/// `ScalarExp`, not a signed double-exponential) — a genuine gap, not an
/// oversight of this note. A Gaussian `D` does not approximate that law in
/// the tails, which is the entire reason Kou (2002) exists as a distinct
/// model from Merton (1976), so shipping a `Default` here would silently
/// hand out Merton-with-Gaussian-jumps under the `Kou` name. Construct with
/// your own `D: Distribution<T> + Send + Sync` implementing the true
/// double-exponential law instead (see [`Kou::new`]).
#[derive(Clone)]
pub struct Kou<T, D, S: SeedExt = Unseeded>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Drift rate μ of the log-price (this field is named `alpha`, not the
  /// module header's jump-size rates η₁/η₂).
  pub alpha: T,
  /// Diffusion scale σ of the continuous (Brownian) component.
  pub sigma: T,
  /// Jump (Poisson) intensity λ — arrival rate of the double-exponential
  /// jumps. Single source of truth, the same convention
  /// [`Merton`](crate::jump::merton::Merton) uses: `sampler()` reads this
  /// field directly, not `cpoisson.poisson.lambda`. This type has no
  /// `with_*` setters, so unlike `Merton` there is no setter path that
  /// could desync the two — [`new`](Self::new) sets both from the same
  /// argument and nothing mutates either afterward short of direct field
  /// assignment.
  pub lambda: T,
  /// Jump-size compensator (E\[Y−1\]-like term), subtracted from the
  /// drift scaled by `lambda` — not a mean-reversion level; Kou's jump
  /// process has no mean reversion.
  pub theta: T,
  /// Number of points sampled along the Kou path.
  pub n: usize,
  /// Initial value X₀ of the Kou path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Compound-Poisson jump driver generating the double-exponential
  /// log-jump sizes. Fully seed-reproducible: [`new`](Self::new) builds it
  /// internally from `seed` (`seed.clone().derive()` — a hash-mixed child,
  /// decorrelated from but a deterministic function of the same `seed` the
  /// diffusion component consults directly — the same shape
  /// [`Merton`](crate::jump::merton::Merton)'s field of the same name
  /// uses), and `sampler()` derives a fresh, chunk-local basis off
  /// `self.cpoisson.seed` for every chunk, mirroring the diffusion
  /// component's own per-chunk `self.seed`-derived basis.
  ///
  /// `sampler()` reads only `cpoisson.distribution` (the jump-size law)
  /// and `self.lambda` — **not** `cpoisson.poisson.lambda` — from this
  /// field on the sampling path; `cpoisson.poisson.{n,t_max,seed}` are
  /// inert there (`grid_increments` never consults them). That inertness
  /// is scoped to *this type's own* sampling, though: `cpoisson` is a
  /// `CompoundPoisson` in its own right, and calling `.sample()` on it
  /// directly (bypassing `Kou` entirely) drives it through
  /// `Poisson::sample_impl`, which *does* branch on `.n`/`.t_max` (fixed
  /// count vs. horizon mode) and *does* consult `.seed` — genuinely live
  /// there. Left `pub` for both reasons: a caller can inspect or directly
  /// `.sample()` the embedded compound-Poisson process as its own
  /// standalone `ProcessExt`, and can replace it wholesale via direct
  /// field assignment (which does not update `self.lambda` — assign that
  /// separately to match, since this type has no `with_cpoisson`-style
  /// setter to do it automatically).
  pub cpoisson: CompoundPoisson<T, D, S>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`). Consulted
  /// directly by the diffusion component; `cpoisson`'s own seed (set at
  /// construction from this same value — see `cpoisson`'s doc above) drives
  /// the jump component.
  pub seed: S,
}

impl<T, D, S: SeedExt> Kou<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Create a new Kou process. Builds the compound-Poisson jump driver
  /// internally from `jump_dist` and `lambda`, seeded from `seed` (see
  /// `cpoisson`'s field doc) — the caller supplies the jump-size
  /// distribution and intensity directly instead of pre-building a
  /// `Poisson`/`CompoundPoisson` pair and threading a third, independent
  /// seed through it by hand.
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    jump_dist: D,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let cpoisson = CompoundPoisson::new(
      jump_dist,
      Poisson::new(lambda, Some(n), t, Unseeded),
      seed.clone().derive(),
    );
    Self {
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      cpoisson,
      seed,
    }
  }
}

impl<T, D, S: SeedExt> ProcessExt<T> for Kou<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = KouSampler<'s, T, D, S>
  where
    Self: 's;

  fn sampler(&self) -> KouSampler<'_, T, D, S> {
    // The diffusion source is owned and derived from `self.seed`. The jump
    // driver's distribution/lambda are borrowed straight off `self.cpoisson`
    // (read-only parameters, safe to share across chunks), but its seed is
    // captured as an owned, chunk-local `self.cpoisson.seed.derive()` —
    // never a borrowed `&self.cpoisson`, which would let every chunk's
    // sampler race on the same shared atomic during the parallel region
    // (see `ProcessExt`'s "Reproducibility requirement on implementors").
    // Each path within one chunk still re-derives its own jump sub-stream
    // from that owned basis exactly as the legacy `sample()` did from the
    // old (always-`Unseeded`) field, so only the seed *source* changed.
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let drift_dt = (self.alpha
      - self.sigma.powf(T::from_usize_(2)) / T::from_usize_(2)
      - self.lambda * self.theta)
      * dt;
    KouSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      drift_dt,
      jump_distribution: &self.cpoisson.distribution,
      lambda: self.lambda,
      jump_seed: self.cpoisson.seed.derive(),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Kou`] sampling state: owns the Gaussian diffusion source and
/// borrows the compound-Poisson jump driver, so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct KouSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  sigma: T,
  x0: T,
  dt: T,
  drift_dt: T,
  jump_distribution: &'a D,
  lambda: T,
  jump_seed: S,
  normal: SimdNormal<T>,
}

impl<T, D, S: SeedExt> KouSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let jump_increments = crate::process::cpoisson::grid_increments(
      self.jump_distribution,
      self.lambda,
      &self.jump_seed,
      out.len(),
      self.dt,
    );
    let mut gn = Array1::<T>::zeros(out.len() - 1);
    if let Some(gn_slice) = gn.as_slice_mut() {
      self.normal.fill_slice(gn_slice);
    }

    out[0] = self.x0;

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.drift_dt + self.sigma * gn[i - 1] + jump_increments[i];
    }
  }
}

impl<T, D, S: SeedExt> PathSampler<T> for KouSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Kou output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyKou {
  inner_f32: Option<Kou<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Kou<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32: Option<Kou<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Kou<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyKou {
  #[new]
  #[pyo3(signature = (alpha, sigma, lambda_, theta, distribution, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    alpha: f64,
    sigma: f64,
    lambda_: f64,
    theta: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
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
            s.seeded_f32 = Some(Kou::new(
              alpha as f32,
              sigma as f32,
              lambda_ as f32,
              theta as f32,
              jump_dist,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(Kou::new(
              alpha as f32,
              sigma as f32,
              lambda_ as f32,
              theta as f32,
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
            s.seeded_f64 = Some(Kou::new(
              alpha,
              sigma,
              lambda_,
              theta,
              jump_dist,
              n,
              x0,
              t,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(Kou::new(
              alpha, sigma, lambda_, theta, jump_dist, n, x0, t, Unseeded,
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
