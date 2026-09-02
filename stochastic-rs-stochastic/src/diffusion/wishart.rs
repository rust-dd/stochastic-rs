//! # Wishart
//!
//! $$
//! dX_t = (\alpha\, a^\top a + b X_t + X_t b^\top)\,dt + \sqrt{X_t}\,dW_t\,a + a^\top dW_t^\top \sqrt{X_t}
//! $$
//!
//! Wishart process on the cone of positive semidefinite `d × d` matrices
//! (Bru 1991): the matrix analogue of CIR and the stochastic covariance of
//! the Gouriéroux–Sufana and Da Fonseca–Grasselli–Tebaldi models. `W` is a
//! `d × d` matrix of independent Brownian motions, `a, b ∈ M_d(ℝ)`, and the
//! degree `α ≥ d − 1` keeps the SDE well posed on the cone.
//!
//! Sampled **exactly** on the grid by the Ahdida–Alfonsi splitting. The law
//! identity (14) maps one step to `WIS_d(y, α, 0, I_dⁿ; Δt)` with
//! `y = θ⁻¹ m x mᵀ θ⁻ᵀ`, `m = exp(Δt b)`, `θ` the extended Cholesky factor of
//! `q_Δt / Δt` and `n = Rk(aᵀa)` (Proposition 6, Algorithm 3). That law is
//! the composition of `n` commuting one-coordinate generators (Theorem 7,
//! Algorithm 2), each solved in closed form by one noncentral χ² variable
//! of degree `α − r` and `r` Gaussians after an extended Cholesky of the
//! remaining `(d − 1) × (d − 1)` block of rank `r` (Theorem 9, Algorithm 1).
//! There is no discretisation error for any admissible `α`, and the paths
//! stay in the cone by construction.
//!
//! Reference: Ahdida, A. & Alfonsi, A. (2013), *Exact and high-order
//! discretization schemes for Wishart processes and their affine extensions*,
//! Ann. Appl. Probab. 23(3), 1025–1073. DOI: 10.1214/12-AAP863

use ndarray::Array2;
use ndarray::Array3;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::non_central_chi_squared::SimdNonCentralChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::linalg::determinant;
use crate::linalg::expm;
use crate::linalg::extended_cholesky;
use crate::linalg::invert_matrix;
use crate::linalg::solve_lower;
use crate::linalg::swap_symmetric;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Wishart<T: FloatExt, S: SeedExt = Unseeded> {
  /// Degree α ≥ d − 1.
  pub alpha: T,
  /// Drift matrix `b` (`d × d`).
  pub b: Array2<T>,
  /// Volatility matrix `a` (`d × d`); only `aᵀa` enters the law.
  pub a: Array2<T>,
  /// Initial positive semidefinite matrix.
  pub x0: Array2<T>,
  /// Number of grid points including the initial one.
  pub n: usize,
  /// Simulation horizon [0, t] (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  step: StepMaps<T>,
}

/// The maps of Proposition 6 for one grid step, computed once.
#[derive(Clone)]
struct StepMaps<T> {
  /// `m = exp(Δt b)`.
  m: Array2<T>,
  /// `θ` with `q_Δt = Δt · θ I_dⁿ θᵀ`.
  theta: Array2<T>,
  /// `θ⁻¹`.
  theta_inv: Array2<T>,
  /// `n = Rk(q_Δt) = Rk(aᵀa)`.
  rank: usize,
}

impl<T: FloatExt> StepMaps<T> {
  fn new(a: &Array2<T>, b: &Array2<T>, dt: T) -> Self {
    let d = a.nrows();
    let (m, q) = drift_maps(a, b, dt);
    let ec = extended_cholesky(&q.mapv(|v| v / dt));
    // θ = Pᵀ C̃ with C̃ = [[c_n, 0], [k_n, I_{d−n}]], i.e. θ[perm[i], j] = C̃[i, j].
    let mut theta = Array2::<T>::zeros((d, d));
    for i in 0..d {
      for j in 0..ec.rank {
        theta[(ec.perm[i], j)] = ec.factor[(i, j)];
      }
      if i >= ec.rank {
        theta[(ec.perm[i], i)] = T::one();
      }
    }
    let theta_inv = invert_matrix(&theta);
    Self {
      m,
      theta,
      theta_inv,
      rank: ec.rank,
    }
  }
}

/// `(m_t, q_t)` of Proposition 4 from one block exponential (Van Loan 1978):
/// `exp(t [[b, aᵀa], [0, −bᵀ]]) = [[m_t, q_t m_t⁻ᵀ], [0, m_t⁻ᵀ]]`, so
/// `q_t = ∫₀ᵗ exp(sb) aᵀa exp(sbᵀ) ds` is the top-right block times `m_tᵀ`.
fn drift_maps<T: FloatExt>(a: &Array2<T>, b: &Array2<T>, t: T) -> (Array2<T>, Array2<T>) {
  let d = a.nrows();
  let ata = a.t().dot(a);
  let mut block = Array2::<T>::zeros((2 * d, 2 * d));
  for i in 0..d {
    for j in 0..d {
      block[(i, j)] = b[(i, j)] * t;
      block[(i, d + j)] = ata[(i, j)] * t;
      block[(d + i, d + j)] = -b[(j, i)] * t;
    }
  }
  let f = expm(&block);
  let m = f.slice(s![..d, ..d]).to_owned();
  let mut q = f.slice(s![..d, d..]).dot(&m.t());
  symmetrise(&mut q);
  (m, q)
}

/// Averages a nearly symmetric matrix with its transpose.
fn symmetrise<T: FloatExt>(x: &mut Array2<T>) {
  let d = x.nrows();
  let half = T::from_f64_fast(0.5);
  for i in 0..d {
    for j in 0..i {
      let v = (x[(i, j)] + x[(j, i)]) * half;
      x[(i, j)] = v;
      x[(j, i)] = v;
    }
  }
}

fn assert_symmetric<T: FloatExt>(x: &Array2<T>, name: &str) {
  let d = x.nrows();
  let tol = T::from_f64_fast(1e-10);
  for i in 0..d {
    for j in 0..i {
      assert!(
        (x[(i, j)] - x[(j, i)]).abs() <= tol,
        "{name} must be symmetric (entry {i},{j})"
      );
    }
  }
}

/// Random-variate sources of the coordinate steps: standard normals for the
/// Gaussian coordinates and, per rank `r`, the noncentral χ² of degree
/// `α − r`; the degree-zero boundary falls back to its Poisson mixture of
/// Gammas with an atom at zero.
struct StepDraws<T: FloatExt, S: SeedExt> {
  alpha: T,
  seed: S,
  normal: SimdNormal<T>,
  ncx2: Vec<Option<SimdNonCentralChiSquared<T>>>,
}

impl<T: FloatExt, S: SeedExt> StepDraws<T, S> {
  fn new(alpha: T, d: usize, seed: &S) -> Self {
    Self {
      alpha,
      seed: seed.derive(),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), seed),
      ncx2: (0..d).map(|_| None).collect(),
    }
  }

  /// `U_Δt` of the squared Bessel process `dU = (α − r) dt + 2√U dZ` from
  /// `u`: `Δt · χ'²(α − r, u / Δt)`.
  fn squared_bessel(&mut self, r: usize, u: T, dt: T) -> T {
    let df = self.alpha - T::from_usize_(r);
    let two = T::from_f64_fast(2.0);
    let ncp = u / dt;
    if df <= T::zero() {
      let half = (ncp / two).to_f64().unwrap_or(0.0);
      let jumps = if half > 0.0 {
        SimdPoisson::<u64>::new(half, &self.seed).sample_fast()
      } else {
        0
      };
      if jumps == 0 {
        return T::zero();
      }
      return dt
        * SimdGamma::<T>::new(T::from_f64_fast(jumps as f64), two, &self.seed).sample_fast();
    }
    if self.ncx2[r].is_none() {
      self.ncx2[r] = Some(SimdNonCentralChiSquared::<T>::new(df, &self.seed));
    }
    dt * self.ncx2[r]
      .as_ref()
      .expect("sampler built above")
      .sample_ncp(ncp)
  }
}

/// Algorithm 1 in place on coordinate 0 of the symmetric PSD matrix `z`:
/// samples `WIS_d(z, α, 0, e¹_d; Δt)`, which changes only the first row and
/// column.
fn coordinate_step<T: FloatExt, S: SeedExt>(z: &mut Array2<T>, dt: T, draws: &mut StepDraws<T, S>) {
  let d = z.nrows();
  let sub = z.slice(s![1.., 1..]).to_owned();
  let ec = extended_cholesky(&sub);
  let r = ec.rank;
  // First row in the pivoted order of the block, then u_{1,l+1} = c_r⁻¹ x̃_{1,l+1}.
  let x_row: Vec<T> = (0..d - 1).map(|i| z[(0, 1 + ec.perm[i])]).collect();
  let u_off = solve_lower(ec.c_r(), &x_row[..r]);
  let mut u11 = z[(0, 0)];
  for u in &u_off {
    u11 -= *u * *u;
  }
  if u11 < T::zero() {
    u11 = T::zero();
  }
  let sqrt_dt = dt.sqrt();
  let u11_next = draws.squared_bessel(r, u11, dt);
  let u_off_next: Vec<T> = u_off
    .iter()
    .map(|u| *u + sqrt_dt * draws.normal.sample_fast())
    .collect();
  let mut z00 = u11_next;
  for u in &u_off_next {
    z00 += *u * *u;
  }
  z[(0, 0)] = z00;
  for i in 0..d - 1 {
    let mut val = T::zero();
    for l in 0..r {
      val += ec.factor[(i, l)] * u_off_next[l];
    }
    z[(0, 1 + ec.perm[i])] = val;
    z[(1 + ec.perm[i], 0)] = val;
  }
}

impl<T: FloatExt, S: SeedExt> Wishart<T, S> {
  /// `a`, `b`, `x0` are `d × d`; `x0` must be symmetric positive semidefinite
  /// and `alpha ≥ d − 1`; `n` counts grid points including the initial one.
  pub fn new(
    alpha: T,
    b: Array2<T>,
    a: Array2<T>,
    x0: Array2<T>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let d = x0.nrows();
    assert!(d >= 1, "need at least one dimension");
    assert_eq!(x0.dim(), (d, d), "x0 must be square");
    assert_eq!(a.dim(), (d, d), "a must be d × d");
    assert_eq!(b.dim(), (d, d), "b must be d × d");
    assert!(
      alpha >= T::from_usize_(d - 1),
      "alpha must be at least d - 1"
    );
    assert!(n >= 1, "n must be at least 1");
    assert_symmetric(&x0, "x0");
    let _ = extended_cholesky(&x0);
    let dt = t.unwrap_or(T::one()) / T::from_usize_(n.max(2) - 1);
    let step = StepMaps::new(&a, &b, dt);
    Self {
      alpha,
      b,
      a,
      x0,
      n,
      t,
      seed,
      step,
    }
  }

  /// Matrix dimension `d`.
  pub fn dim(&self) -> usize {
    self.x0.nrows()
  }

  /// Time step `Δt = t / (n − 1)`.
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.max(2) - 1)
  }

  /// Rank of `aᵀa`, the number of coordinates the noise acts on.
  pub fn noise_rank(&self) -> usize {
    self.step.rank
  }

  /// Closed-form mean `E[X_t] = m_t x₀ m_tᵀ + α q_t` (first moment of the
  /// Laplace transform (10)).
  pub fn mean(&self, t: T) -> Array2<T> {
    let (m, q) = drift_maps(&self.a, &self.b, t);
    let mut out = m.dot(&self.x0).dot(&m.t());
    out.zip_mut_with(&q, |o, &qv| *o += self.alpha * qv);
    out
  }

  /// Laplace transform `E[exp(Tr(v X_t))]` of Proposition 4, eq. (10):
  /// `exp(Tr[v (I − 2 q_t v)⁻¹ m_t x₀ m_tᵀ]) / det(I − 2 q_t v)^{α/2}`, valid
  /// for symmetric `v` in the convergence domain, in particular for every
  /// negative semidefinite `v`.
  pub fn laplace_transform(&self, v: &Array2<T>, t: T) -> T {
    let d = self.dim();
    let two = T::from_f64_fast(2.0);
    let (m, q) = drift_maps(&self.a, &self.b, t);
    let k = &Array2::<T>::eye(d) - &q.dot(v).mapv(|x| x * two);
    let det = determinant(&k);
    assert!(
      det > T::zero(),
      "v lies outside the convergence domain of the Laplace transform"
    );
    let inner = v.dot(&invert_matrix(&k)).dot(&m).dot(&self.x0).dot(&m.t());
    let trace = (0..d).fold(T::zero(), |acc, i| acc + inner[(i, i)]);
    trace.exp() / det.powf(self.alpha / two)
  }

  /// One exact step (Algorithm 3): `X ↦ θ Y θᵀ` with
  /// `Y ~ WIS_d(θ⁻¹ m X mᵀ θ⁻ᵀ, α, 0, I_dⁿ; Δt)` from Algorithm 2.
  fn step<S2: SeedExt>(&self, x: &Array2<T>, dt: T, draws: &mut StepDraws<T, S2>) -> Array2<T> {
    let mx = self.step.m.dot(x).dot(&self.step.m.t());
    let mut y = self.step.theta_inv.dot(&mx).dot(&self.step.theta_inv.t());
    symmetrise(&mut y);
    for k in 0..self.step.rank {
      swap_symmetric(&mut y, 0, k);
      coordinate_step(&mut y, dt, draws);
      swap_symmetric(&mut y, 0, k);
    }
    let mut out = self.step.theta.dot(&y).dot(&self.step.theta.t());
    symmetrise(&mut out);
    out
  }

  fn fill_path<S2: SeedExt>(&self, seed: &S2, out: &mut Array3<T>) {
    let d = self.dim();
    assert_eq!(out.dim(), (self.n, d, d), "output must be n × d × d");
    out.slice_mut(s![0, .., ..]).assign(&self.x0);
    if self.n == 1 {
      return;
    }
    let dt = self.dt();
    let mut draws = StepDraws::new(self.alpha, d, seed);
    let mut x = self.x0.clone();
    for j in 1..self.n {
      x = self.step(&x, dt, &mut draws);
      out.slice_mut(s![j, .., ..]).assign(&x);
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Wishart<T, S> {
  type Output = Array3<T>;
  type Sampler<'s>
    = WishartSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> WishartSampler<T, S> {
    WishartSampler {
      process: Wishart {
        alpha: self.alpha,
        b: self.b.clone(),
        a: self.a.clone(),
        x0: self.x0.clone(),
        n: self.n,
        t: self.t,
        seed: self.seed.derive(),
        step: self.step.clone(),
      },
    }
  }
}

/// Reusable [`Wishart`] sampling state with a derived seed, so parallel
/// chunks draw independent paths.
#[doc(hidden)]
pub struct WishartSampler<T: FloatExt, S: SeedExt> {
  process: Wishart<T, S>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for WishartSampler<T, S> {
  type Output = Array3<T>;

  fn sample_into(&mut self, out: &mut Array3<T>) {
    // The derived seed travels by reference: every constructor call advances
    // it, so consecutive paths of one chunk are independent.
    self.process.fill_path(&self.process.seed, out);
  }

  fn sample(&mut self) -> Array3<T> {
    let d = self.process.dim();
    let mut out = Array3::<T>::zeros((self.process.n, d, d));
    self.sample_into(&mut out);
    out
  }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyWishart {
  inner: Option<Wishart<f64>>,
  seeded: Option<Wishart<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyWishart {
  /// Wishart process with degree `alpha`, drift `b`, volatility `a` and
  /// initial matrix `x0` (all `d × d` arrays), `n` grid points over `[0, t]`.
  #[new]
  #[pyo3(signature = (alpha, b, a, x0, n, t=None, seed=None))]
  fn new(
    alpha: f64,
    b: numpy::PyReadonlyArray2<'_, f64>,
    a: numpy::PyReadonlyArray2<'_, f64>,
    x0: numpy::PyReadonlyArray2<'_, f64>,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    let (b, a, x0) = (
      b.as_array().to_owned(),
      a.as_array().to_owned(),
      x0.as_array().to_owned(),
    );
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Wishart::new(
          alpha,
          b,
          a,
          x0,
          n,
          t,
          stochastic_rs_core::simd_rng::Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(Wishart::new(alpha, b, a, x0, n, t, Unseeded)),
        seeded: None,
      },
    }
  }

  /// One `(n, d, d)` array of matrices along the grid.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// `m` independent paths as a list of `(n, d, d)` arrays.
  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| {
      let paths: Vec<pyo3::Py<pyo3::PyAny>> = inner
        .sample_par(m)
        .into_iter()
        .map(|p| p.into_pyarray(py).into_py_any(py).unwrap())
        .collect();
      paths.into_py_any(py).unwrap()
    })
  }

  /// Closed-form mean `E[X_t]` as a `(d, d)` array.
  fn mean<'py>(&self, py: pyo3::Python<'py>, t: f64) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| inner
      .mean(t)
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// Laplace transform `E[exp(Tr(v X_t))]` for a symmetric `v` in the
  /// convergence domain (every negative semidefinite `v` qualifies).
  fn laplace_transform(&self, v: numpy::PyReadonlyArray2<'_, f64>, t: f64) -> f64 {
    let v = v.as_array().to_owned();
    crate::py_dispatch_f64!(self, |inner| inner.laplace_transform(&v, t))
  }
}
