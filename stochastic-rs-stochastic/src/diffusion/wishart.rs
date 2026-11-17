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
//!
//! Reference: Siegel, A. F. (1979), "The noncentral chi-squared distribution with zero degrees of freedom and testing for uniformity", Biometrika 66(2) 381–386, DOI 10.1093/biomet/66.2.381.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::non_central_chi_squared::SimdNonCentralChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::linalg::determinant;
use crate::linalg::expm;
use crate::linalg::extended_cholesky;
use crate::linalg::invert_matrix;
use crate::linalg::solve_lower;
use crate::linalg::swap_symmetric;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Wishart<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  step: StepMaps<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// The maps of Proposition 6 for one grid step, computed once.
#[derive(Clone, Debug)]
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
        "{name}[({i}, {j})] must satisfy `({name}[(i, j)] - {name}[(j, i)]).abs() <= tol`, got {name}[({i}, {j})] = {:?}, {name}[({j}, {i})] = {:?}",
        x[(i, j)],
        x[(j, i)]
      );
    }
  }
}

/// Random-variate sources of the coordinate steps: standard normals for the Gaussian coordinates
/// and, per rank `r`, the noncentral χ² of degree `α − r`, the zero degree included.
struct StepDraws<T: FloatExt, S: SeedExt> {
  alpha: T,
  seed: S,
  normal: Seeded<SimdNormal<T>>,
  ncx2: Vec<Option<Seeded<SimdNonCentralChiSquared<T>>>>,
}

impl<T: FloatExt, S: SeedExt> StepDraws<T, S> {
  fn new(alpha: T, d: usize, seed: &S) -> Self {
    Self {
      alpha,
      seed: seed.derive(),
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed),
      ncx2: (0..d).map(|_| None).collect(),
    }
  }

  /// `U_Δt` of the squared Bessel process `dU = (α − r) dt + 2√U dZ` from
  /// `u`: `Δt · χ'²(α − r, u / Δt)`.
  fn squared_bessel(&mut self, r: usize, u: T, dt: T) -> T {
    let df = self.alpha - T::from_usize_(r);
    let ncp = u / dt;
    if self.ncx2[r].is_none() {
      self.ncx2[r] = Some(SimdNonCentralChiSquared::<T>::new(df).seeded(&self.seed));
    }
    dt * self.ncx2[r]
      .as_mut()
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
    .map(|u| *u + sqrt_dt * draws.normal.sample())
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
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert_entries("b", "b[(i, j)].is_finite()", &b, |v| v.is_finite());
    assert_entries("a", "a[(i, j)].is_finite()", &a, |v| v.is_finite());
    assert_entries("x0", "x0[(i, j)].is_finite()", &x0, |v| v.is_finite());
    let d = x0.nrows();
    assert!(
      x0.nrows() >= 1,
      "x0.nrows() must satisfy `x0.nrows() >= 1`, got x0.nrows() = {d:?}"
    );
    assert!(
      x0.dim() == (d, d),
      "x0.dim() must satisfy `x0.dim() == (d, d)`, got x0.dim() = {:?}, d = {d:?}",
      x0.dim()
    );
    assert!(
      a.dim() == (d, d),
      "a.dim() must satisfy `a.dim() == (d, d)`, got a.dim() = {:?}, d = {d:?}",
      a.dim()
    );
    assert!(
      b.dim() == (d, d),
      "b.dim() must satisfy `b.dim() == (d, d)`, got b.dim() = {:?}, d = {d:?}",
      b.dim()
    );
    assert!(
      alpha >= T::from_usize_(d - 1),
      "alpha must satisfy `alpha >= T::from_usize_(d - 1)`, got alpha = {alpha:?}, d = {d:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert_symmetric(&x0, "x0");
    let _ = extended_cholesky(&x0);
    let dt = t.unwrap_or(T::one()) / T::from_usize_(n.max(2) - 1);
    let step = StepMaps::new(&a, &b, dt);
    Self {
      backend: Cpu,
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
}

impl<T: FloatExt, S: SeedExt, B> Wishart<T, S, B> {
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

/// The Euler engine's view of the matrix path: the three entries of the
/// symmetric matrix as one launch's slots, the exact step's maps folded once.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct WishartLaunch<'a, T: FloatExt, S: SeedExt, B>(&'a Wishart<T, S, B>);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for WishartLaunch<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for WishartLaunch<'_, T, S, B>
{
  type Output = [Array1<T>; 3];
  type Sampler<'s>
    = WishartLaunchSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> WishartLaunchSampler<T, S> {
    WishartLaunchSampler {
      inner: <Wishart<T, S, B> as ProcessExt<T>>::sampler(self.0),
    }
  }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct WishartLaunchSampler<T: FloatExt, S: SeedExt> {
  inner: WishartSampler<T, S>,
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for WishartLaunchSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for WishartLaunchSampler<T, S> {
  type Output = [Array1<T>; 3];

  fn sample_into(&mut self, out: &mut [Array1<T>; 3]) {
    *out = self.sample();
  }

  fn sample(&mut self) -> [Array1<T>; 3] {
    let path = self.inner.sample();
    [
      path.slice(s![.., 0, 0]).to_owned(),
      path.slice(s![.., 0, 1]).to_owned(),
      path.slice(s![.., 1, 1]).to_owned(),
    ]
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for WishartLaunch<'_, T, S, B>
{
  /// The step's maps row-major. A configuration the family cannot carry
  /// never reaches a launch — [`ProcessExt::sample`] keeps it on the host —
  /// so asking here is a caller bypassing that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let p = self.0;
    assert!(
      p.device_ready(),
      "Wishart: a launch carries two dimensions at a degree of at least three; sample through \
       `ProcessExt`, which keeps the rest on the host"
    );
    let flat = |m: &Array2<T>| [m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]];
    crate::euler::EulerSpec::WishartTwo {
      m: flat(&p.step.m),
      theta: flat(&p.step.theta),
      theta_inv: flat(&p.step.theta_inv),
      two: if p.step.rank >= 2 {
        T::one()
      } else {
        T::zero()
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    let x0 = &self.0.x0;
    [x0[(0, 0)], x0[(0, 1)], x0[(1, 1)], T::zero()]
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

  /// The central χ²(α − 2) each coordinate's squared Bessel draw adds to its
  /// shifted normal's square: a gamma of shape `(α − 2) / 2` and scale two,
  /// one per coordinate.
  fn gamma_draws(&self) -> Option<crate::euler::GammaDraws<T>> {
    let two = T::from_usize_(2);
    let draw = ((self.0.alpha - two) / two, two, T::zero());
    Some(crate::euler::GammaDraws {
      first: draw,
      second: Some(draw),
    })
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 3] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

/// The launch's three slots back into the `n × 2 × 2` matrix path.
fn slots_to_matrices<T: FloatExt>(slots: [Array1<T>; 3]) -> Array3<T> {
  let n = slots[0].len();
  let mut out = Array3::<T>::zeros((n, 2, 2));
  for j in 0..n {
    out[(j, 0, 0)] = slots[0][j];
    out[(j, 0, 1)] = slots[1][j];
    out[(j, 1, 0)] = slots[1][j];
    out[(j, 1, 1)] = slots[2][j];
  }
  out
}

backend_switch!([T: FloatExt, S: SeedExt] Wishart<T, S> { alpha, b, a, x0, n, t, seed, step } via euler);

mod sampler;

#[doc(hidden)]
pub use sampler::WishartSampler;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyWishart;

#[cfg(test)]
mod tests;
