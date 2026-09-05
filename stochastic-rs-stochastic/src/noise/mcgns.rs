//! # Mcgns
//!
//! $$
//! \Delta W_t = L\,\varepsilon_t\sqrt{\Delta t},\quad \varepsilon_t\sim\mathcal N(0,I_k),\ LL^\top=\rho
//! $$
//!
//! Multivariate correlated Gaussian noise: `k` Brownian increment streams
//! with instantaneous correlation matrix `ρ`, built from independent
//! standard normals through the lower Cholesky factor of `ρ` — the
//! `k`-dimensional driver behind [`crate::diffusion::multi_gbm::MultiGbm`]
//! and the generalisation of the two-stream [`crate::noise::cgns::Cgns`].
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial
//! Engineering*, Springer, §2.3.3. DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::linalg::cholesky_lower;
use crate::linalg::validate_correlation;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Mcgns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Instantaneous correlation matrix ρ of the `k` streams (`k × k`).
  pub rho: Array2<T>,
  /// Number of increments sampled along each stream.
  pub n: usize,
  /// Simulation horizon [0, t] (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  chol: Array2<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Mcgns<T, S> {
  /// `rho` must be a valid, positive-definite correlation matrix.
  pub fn new(rho: Array2<T>, n: usize, t: Option<T>, seed: S) -> Self {
    validate_correlation(&rho);
    let chol = cholesky_lower(&rho);
    Self {
      backend: Cpu,
      rho,
      n,
      t,
      seed,
      chol,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Mcgns<T, S, B> {
  /// Number of streams `k`.
  pub fn dims(&self) -> usize {
    self.rho.nrows()
  }

  /// Time step $\Delta t = t/n$.
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }

  /// Lower Cholesky factor $L$ of `rho`.
  pub fn cholesky(&self) -> &Array2<T> {
    &self.chol
  }

  /// Fills `out` (`k × n`) with correlated increments: row `i` is the
  /// `i`-th stream, each increment has variance $\Delta t$.
  pub(crate) fn fill_increments<S2: SeedExt>(&self, seed: &S2, out: &mut Array2<T>) {
    let k = self.dims();
    assert_eq!(out.dim(), (k, self.n), "output must be k × n");
    if self.n == 0 {
      return;
    }
    let sqrt_dt = self.dt().sqrt();
    let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
    let mut white = Array2::<T>::zeros((k, self.n));
    for mut row in white.rows_mut() {
      normal.fill_slice(row.as_slice_mut().expect("Mcgns rows must be contiguous"));
    }
    for j in 0..self.n {
      for i in 0..k {
        let mut acc = T::zero();
        for l in 0..=i {
          acc += self.chol[(i, l)] * white[(l, j)];
        }
        out[(i, j)] = acc;
      }
    }
  }
}

/// The streams in the engine's four slots. `Mcgns` reports a `k × n` matrix
/// and the correlated noise family steps four components under one Cholesky
/// factor, so this view carries a launch when `k <= 4`, the unused slots
/// padded with a unit diagonal and dropped. It borrows rather than owns, so
/// the seed it advances is the process's own.
#[doc(hidden)]
pub struct McgnsLaunch<'a, T: FloatExt, S: SeedExt, B>(&'a Mcgns<T, S, B>);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for McgnsLaunch<'_, T, S, B>
{
  type Output = [Array1<T>; 4];
  type Sampler<'s>
    = McgnsLaunchSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> McgnsLaunchSampler<T, S> {
    McgnsLaunchSampler {
      inner: <Mcgns<T, S, B> as ProcessExt<T>>::sampler(self.0),
    }
  }
}

/// [`McgnsLaunch`]'s sampler: the matrix sampler, its rows lifted into the
/// four slots.
#[doc(hidden)]
pub struct McgnsLaunchSampler<T: FloatExt, S: SeedExt> {
  inner: McgnsSampler<T, S>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for McgnsLaunchSampler<T, S> {
  type Output = [Array1<T>; 4];

  fn sample_into(&mut self, out: &mut [Array1<T>; 4]) {
    *out = self.sample();
  }

  fn sample(&mut self) -> [Array1<T>; 4] {
    let matrix = self.inner.sample();
    let n = matrix.ncols();
    std::array::from_fn(|i| {
      if i < matrix.nrows() {
        matrix.row(i).to_owned()
      } else {
        Array1::zeros(n)
      }
    })
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 4>
  for McgnsLaunch<'_, T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CorrelatedNoises4 {
      l: crate::euler::pack_cholesky(&self.0.chol),
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  /// Every grid point is a draw, so the frame steps before it writes the
  /// first one.
  fn step_first(&self) -> bool {
    true
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.0.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 4] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Mcgns<T, S> { rho, n, t, seed, chol } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Mcgns<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = McgnsSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> McgnsSampler<T, S> {
    McgnsSampler {
      noise: Mcgns {
        backend: Cpu,
        rho: self.rho.clone(),
        n: self.n,
        t: self.t,
        seed: self.seed.derive(),
        chol: self.chol.clone(),
      },
    }
  }

  /// Through the Euler engine for up to four streams, which is what the
  /// correlated noise family carries: on a device the whole matrix is drawn
  /// in one kernel under its Cholesky factor. More streams stay on this
  /// process's own sampler whatever the backend.
  fn sample(&self) -> Array2<T> {
    if self.dims() <= crate::euler::CORRELATED_STREAMS {
      slots_to_matrix(self.backend.system_sample(&McgnsLaunch(self)), self.dims())
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    if self.dims() <= crate::euler::CORRELATED_STREAMS {
      let k = self.dims();
      self
        .backend
        .system_paths_map(&McgnsLaunch(self), m, |slots| {
          f(&slots_to_matrix(slots.clone(), k))
        })
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    if self.dims() <= crate::euler::CORRELATED_STREAMS {
      let k = self.dims();
      self
        .backend
        .system_paths(&McgnsLaunch(self), m)
        .into_iter()
        .map(|slots| slots_to_matrix(slots, k))
        .collect()
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array2<T>, crate::device::DeviceError> {
    if self.dims() <= crate::euler::CORRELATED_STREAMS {
      let slots = self.backend.try_system_sample(&McgnsLaunch(self))?;
      Ok(slots_to_matrix(slots, self.dims()))
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, crate::device::DeviceError> {
    if self.dims() <= crate::euler::CORRELATED_STREAMS {
      let k = self.dims();
      Ok(
        self
          .backend
          .try_system_paths(&McgnsLaunch(self), m)?
          .into_iter()
          .map(|slots| slots_to_matrix(slots, k))
          .collect(),
      )
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// The first `k` of the engine's four slots as the `k × n` matrix the process
/// reports; the padded slots are dropped.
fn slots_to_matrix<T: FloatExt>(slots: [Array1<T>; 4], k: usize) -> Array2<T> {
  let n = slots[0].len();
  let mut out = Array2::<T>::zeros((k, n));
  for (i, row) in slots.iter().take(k).enumerate() {
    out.row_mut(i).assign(row);
  }
  out
}

/// Reusable [`Mcgns`] sampling state with a derived seed, so parallel
/// chunks draw independent streams.
#[doc(hidden)]
pub struct McgnsSampler<T: FloatExt, S: SeedExt> {
  noise: Mcgns<T, S>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for McgnsSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    // Pass the derived seed by reference: each `SimdNormal::new` advances
    // its state, so consecutive paths of one chunk draw fresh increments.
    self.noise.fill_increments(&self.noise.seed, out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((self.noise.dims(), self.noise.n));
    self.sample_into(&mut out);
    out
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// The sample cross-covariance of the increments reproduces ρΔt.
  #[test]
  fn increments_carry_the_requested_correlation() {
    let rho = array![[1.0_f64, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]];
    let n = 200_000;
    let noise = Mcgns::new(rho.clone(), n, Some(2.0), Deterministic::new(11));
    let dw = noise.sample();
    assert_eq!(dw.dim(), (3, n));
    let dt = 2.0 / n as f64;
    for i in 0..3 {
      for j in 0..3 {
        let cov: f64 = (0..n).map(|s| dw[(i, s)] * dw[(j, s)]).sum::<f64>() / n as f64;
        assert!(
          (cov / dt - rho[(i, j)]).abs() < 0.02,
          "cov({i},{j})/dt = {} vs {}",
          cov / dt,
          rho[(i, j)]
        );
      }
    }
  }

  /// With ρ = I the streams are the independent white-noise rows.
  #[test]
  fn identity_correlation_leaves_the_streams_independent() {
    let noise = Mcgns::new(Array2::<f64>::eye(2), 50_000, None, Deterministic::new(3));
    let dw = noise.sample();
    let cross: f64 = (0..50_000).map(|s| dw[(0, s)] * dw[(1, s)]).sum::<f64>() / 50_000.0;
    assert!(cross.abs() * 50_000.0 < 0.02 * 50_000.0 / 50_000.0_f64.sqrt() * 3.0 + 1e-3);
  }

  /// Consecutive paths of one sampler differ: the derived seed is passed by
  /// reference so each call advances it, where a cloned seed would replay
  /// the same increments for every path of a chunk.
  #[test]
  fn consecutive_paths_of_one_sampler_differ() {
    let noise = Mcgns::new(
      array![[1.0_f64, 0.2], [0.2, 1.0]],
      16,
      None,
      Deterministic::new(3),
    );
    let mut sampler = noise.sampler();
    let a = sampler.sample();
    let b = sampler.sample();
    assert_ne!(a, b);
  }

  #[test]
  fn deterministic_seed_reproduces_and_derived_chunks_differ() {
    let a = Mcgns::new(
      array![[1.0_f64, 0.5], [0.5, 1.0]],
      64,
      None,
      Deterministic::new(7),
    )
    .sample();
    let b = Mcgns::new(
      array![[1.0_f64, 0.5], [0.5, 1.0]],
      64,
      None,
      Deterministic::new(7),
    )
    .sample();
    assert_eq!(a, b);
    let chunks = Mcgns::new(
      array![[1.0_f64, 0.5], [0.5, 1.0]],
      64,
      None,
      Deterministic::new(7),
    )
    .sample_par(2);
    assert_ne!(chunks[0], chunks[1]);
  }

  #[test]
  #[should_panic(expected = "unit diagonal")]
  fn rejects_a_covariance_matrix() {
    let _ = Mcgns::new(array![[2.0_f64, 0.5], [0.5, 1.0]], 8, None, Unseeded);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyMcgns {
  inner: Option<Mcgns<f64>>,
  seeded: Option<Mcgns<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyMcgns {
  /// `k` correlated Brownian increment streams with correlation matrix
  /// `rho` (`k × k`), `n` increments over `[0, t]`.
  #[new]
  #[pyo3(signature = (rho, n, t=None, seed=None))]
  fn new(
    rho: numpy::PyReadonlyArray2<'_, f64>,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    let rho = rho.as_array().to_owned();
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Mcgns::new(
          rho,
          n,
          t,
          stochastic_rs_core::simd_rng::Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(Mcgns::new(rho, n, t, Unseeded)),
        seeded: None,
      },
    }
  }

  /// One `(k, n)` matrix of increments.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// `m` independent increment matrices as a list of `(k, n)` arrays.
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
