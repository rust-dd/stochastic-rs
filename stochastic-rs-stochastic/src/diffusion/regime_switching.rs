//! GBM whose volatility follows a continuous-time Markov chain on `M` regimes with generator `Q`.
//!
//! $$
//! dS_t = (r-q)S_t\,dt + \sigma_{Z_t}\,S_t\,dW_t
//! $$
//!
//! Source: Kirkby, PROJ_Option_Pricing_Matlab
//! Reference: Hamilton (1989), Econometrica 57(2), 357–384
//!

use ndarray::Array1;
use ndarray::Array2;
use rand::RngExt;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// GBM whose volatility is picked by a regime; the regimes switch by the CTMC generator `Q`.
#[derive(Debug, Clone)]
pub struct RegimeSwitchingDiffusion<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift rate (e.g. r - q).
  pub mu: T,
  /// CTMC generator matrix (M×M, rows sum to 0).
  pub q_matrix: Array2<T>,
  /// Per-regime volatilities.
  pub vols: Array1<T>,
  /// Initial regime state (0-indexed).
  pub initial_state: usize,
  /// Number of points sampled along the regime-switching path.
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> RegimeSwitchingDiffusion<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `q_matrix.nrows() == vols.len()`,
  /// `q_matrix.ncols() == vols.len()`, `initial_state < vols.len()`, and every parameter is finite.
  pub fn new(
    mu: T,
    q_matrix: Array2<T>,
    vols: Array1<T>,
    initial_state: usize,
    n: usize,
    s0: T,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert_entries("q_matrix", "q_matrix[(i, j)].is_finite()", &q_matrix, |v| {
      v.is_finite()
    });
    assert_entries("vols", "vols[i].is_finite()", &vols, |v| v.is_finite());
    assert!(
      q_matrix.nrows() == vols.len(),
      "q_matrix.nrows() must satisfy `q_matrix.nrows() == vols.len()`, got q_matrix.nrows() = {:?}, vols.len() = {:?}",
      q_matrix.nrows(),
      vols.len()
    );
    assert!(
      q_matrix.ncols() == vols.len(),
      "q_matrix.ncols() must satisfy `q_matrix.ncols() == vols.len()`, got q_matrix.ncols() = {:?}, vols.len() = {:?}",
      q_matrix.ncols(),
      vols.len()
    );
    assert!(
      initial_state < vols.len(),
      "initial_state must satisfy `initial_state < vols.len()`, got initial_state = {initial_state:?}, vols.len() = {:?}",
      vols.len()
    );

    Self {
      backend: Cpu,
      mu,
      q_matrix,
      vols,
      initial_state,
      n,
      s0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> RegimeSwitchingDiffusion<T, S, B> {
  /// One-step CTMC transition-probability matrix `exp(Q · dt)` for a step of
  /// size `dt`, computed via scaling-and-squaring of the generator `Q`.
  pub fn transition_prob_matrix(&self, dt: T) -> Array2<T> {
    let m = self.vols.len();
    let mut a = Array2::<T>::zeros((m, m));
    for i in 0..m {
      for j in 0..m {
        a[[i, j]] = self.q_matrix[[i, j]] * dt;
      }
    }
    matrix_exp_real(&a)
  }

  /// Draw the next regime from the `current` regime's row of a one-step
  /// transition matrix `p_matrix` via inverse-CDF sampling on `rng`.
  pub fn sample_next_regime<R: rand::Rng + ?Sized>(
    &self,
    current: usize,
    p_matrix: &Array2<T>,
    rng: &mut R,
  ) -> usize {
    let u: f64 = rng.random();
    let m = self.vols.len();
    let mut cum = T::zero();
    for j in 0..m {
      cum += p_matrix[[current, j]];
      if T::from_f64_fast(u) <= cum {
        return j;
      }
    }
    m - 1
  }
}

fn matrix_exp_real<T: FloatExt>(a: &Array2<T>) -> Array2<T> {
  let m = a.nrows();

  let mut norm = T::zero();
  for i in 0..m {
    let mut row_sum = T::zero();
    for j in 0..m {
      row_sum += a[[i, j]].abs();
    }
    if row_sum > norm {
      norm = row_sum;
    }
  }

  let s = if norm > T::zero() {
    (norm.to_f64().unwrap().log2().ceil() as usize).saturating_add(1)
  } else {
    0
  };
  let scale = T::from_f64_fast((2.0_f64).powi(s as i32));

  let mut scaled = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..m {
      scaled[[i, j]] = a[[i, j]] / scale;
    }
  }

  let mut result = Array2::<T>::zeros((m, m));
  for i in 0..m {
    result[[i, i]] = T::one();
  }

  let mut term = Array2::<T>::zeros((m, m));
  for i in 0..m {
    term[[i, i]] = T::one();
  }

  for k in 1..=20 {
    term = mat_mul(&term, &scaled);
    let divisor = T::from_usize_(k);
    for i in 0..m {
      for j in 0..m {
        term[[i, j]] = term[[i, j]] / divisor;
        result[[i, j]] += term[[i, j]];
      }
    }
  }

  for _ in 0..s {
    result = mat_mul(&result, &result);
  }

  result
}

fn mat_mul<T: FloatExt>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
  let m = a.nrows();
  let mut c = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..m {
      let mut sum = T::zero();
      for k in 0..m {
        sum += a[[i, k]] * b[[k, j]];
      }
      c[[i, j]] = sum;
    }
  }
  c
}

/// Regimes a launch carries: the family holds four volatilities and four rows of thresholds. A
/// longer chain stays on its host sampler.
const DEVICE_REGIMES: usize = 4;

/// The Euler engine's view: `exp(Q dt)`, the host sampler's own matrix, travels as per-row
/// cumulative thresholds for the kernel's inverse-CDF draw; the regime rides in the second slot.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for RegimeSwitchingDiffusion<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let m = self.vols.len();
    let p_matrix = self.transition_prob_matrix(self.dt());
    let sigma = std::array::from_fn(|r| if r < m { self.vols[r] } else { T::zero() });
    let thresholds = std::array::from_fn(|r| {
      let mut cum = T::zero();
      std::array::from_fn(|j| {
        if r < m && j < m {
          cum += p_matrix[[r, j]];
          cum
        } else {
          T::one()
        }
      })
    });
    crate::euler::EulerSpec::RegimeSwitching {
      mu: self.mu,
      sigma,
      thresholds,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.s0,
      T::from_usize_(self.initial_state),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

impl<T: FloatExt, S: SeedExt, B> RegimeSwitchingDiffusion<T, S, B> {
  /// The grid spacing: `n - 1` increments over the horizon, zero when the
  /// grid has no increment to take.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }
}

backend_switch!([T: FloatExt, S: SeedExt] RegimeSwitchingDiffusion<T, S> { mu, q_matrix, vols, initial_state, n, s0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for RegimeSwitchingDiffusion<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for RegimeSwitchingDiffusion<T, S, B>
{
  /// Returns [stock_prices, regime_states] where regime_states are cast to T.
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = RegimeSwitchingDiffusionSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> RegimeSwitchingDiffusionSampler<T, S> {
    RegimeSwitchingDiffusionSampler {
      mu: self.mu,
      q_matrix: self.q_matrix.clone(),
      vols: self.vols.clone(),
      initial_state: self.initial_state,
      n: self.n,
      s0: self.s0,
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine for up to four regimes, the spot and the chain stepped in one
  /// kernel; a longer chain stays on this process's own sampler.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.vols.len() <= DEVICE_REGIMES {
      self.backend.system_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    if self.vols.len() <= DEVICE_REGIMES {
      self.backend.system_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    if self.vols.len() <= DEVICE_REGIMES {
      self.backend.system_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    if self.vols.len() <= DEVICE_REGIMES {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    if self.vols.len() <= DEVICE_REGIMES {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`RegimeSwitchingDiffusion`] sampling state: the generator, the volatilities and the
/// seed source. Each call rebuilds the Gaussian stream, then the regime RNG, from the derived seed.
#[doc(hidden)]
#[derive(Debug)]
pub struct RegimeSwitchingDiffusionSampler<T: FloatExt, S: SeedExt> {
  mu: T,
  q_matrix: Array2<T>,
  vols: Array1<T>,
  initial_state: usize,
  n: usize,
  s0: T,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> RegimeSwitchingDiffusionSampler<T, S> {
  fn transition_prob_matrix(&self, dt: T) -> Array2<T> {
    let m = self.vols.len();
    let mut a = Array2::<T>::zeros((m, m));
    for i in 0..m {
      for j in 0..m {
        a[[i, j]] = self.q_matrix[[i, j]] * dt;
      }
    }
    matrix_exp_real(&a)
  }

  fn sample_next_regime<R: rand::Rng + ?Sized>(
    &self,
    current: usize,
    p_matrix: &Array2<T>,
    rng: &mut R,
  ) -> usize {
    let u: f64 = rng.random();
    let m = self.vols.len();
    let mut cum = T::zero();
    for j in 0..m {
      cum += p_matrix[[current, j]];
      if T::from_f64_fast(u) <= cum {
        return j;
      }
    }
    m - 1
  }

  fn fill_paths(&mut self, s_path: &mut [T], z_path: &mut [T]) {
    if self.n == 0 {
      return;
    }

    s_path[0] = self.s0;
    z_path[0] = T::from_usize_(self.initial_state);

    if self.n <= 1 {
      return;
    }

    let n_inc = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_inc);
    let sqrt_dt = dt.sqrt();

    let p_matrix = self.transition_prob_matrix(dt);

    let mut dw = Array1::<T>::zeros(n_inc);
    let dw_slice = dw.as_slice_mut().unwrap();
    let mut normal = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed);
    normal.fill_slice(dw_slice);

    let mut rng = self.seed.rng();
    let mut state = self.initial_state;

    for i in 1..self.n {
      let sigma = self.vols[state];
      let half = T::from_f64_fast(0.5);

      let log_inc = (self.mu - half * sigma * sigma) * dt + sigma * dw[i - 1];
      s_path[i] = s_path[i - 1] * log_inc.exp();

      state = self.sample_next_regime(state, &p_matrix, &mut rng);
      z_path[i] = T::from_usize_(state);
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for RegimeSwitchingDiffusionSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for RegimeSwitchingDiffusionSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, z] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("RegimeSwitching output must be contiguous"),
      z.as_slice_mut()
        .expect("RegimeSwitching output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s_path = Array1::<T>::zeros(self.n);
    let mut z_path = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s_path.as_slice_mut().expect("contiguous"),
      z_path.as_slice_mut().expect("contiguous"),
    );
    [s_path, z_path]
  }
}

#[cfg(test)]
mod tests;
