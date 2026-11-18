//! # Lmm — drift-coupled LIBOR Market Model under the spot-LIBOR measure
//!
//! $$
//! \frac{dL_n(t)}{L_n(t)} \;=\;
//!   \sigma_n(t)\!\!\sum_{j=\eta(t)}^{n}\!\!
//!     \frac{\delta_j\,\rho_{n,j}\,\sigma_j(t)\,L_j(t)}{1+\delta_j L_j(t)}\,dt
//!   \;+\;\sigma_n(t)\,dW_n^{\!*}(t)
//! $$
//!
//! Forward-LIBOR market model (BGM/Jamshidian/Brace-Gatarek-Musiela) with
//! cross-forward drift coupling under the **spot-LIBOR measure**
//! `B^*(t)` of Glasserman (2003) §3.7.
//!
//! Numerical scheme: **log-Euler with frozen-at-step drift** (Glasserman
//! 2003 §3.7.1; Joshi 2003). Each step approximates
//!
//! ```text
//! L_n(t+Δt) = L_n(t)·exp[(μ_n(t) − ½σ_n²) Δt + σ_n √Δt · Z_n]
//! ```
//!
//! where `μ_n(t)` is the drift summation evaluated at the start of the
//! step and `Z_n = Σ_k chol(ρ)_{n,k} ε_k` with `ε ~ N(0, I)`. Log-Euler
//! preserves positivity (forwards stay positive, unlike a forward-Euler
//! step) and is the standard discretization choice for caplet / swaption
//! pricing.
//!
//! ## Tenor structure
//!
//! Tenor `T_0 < T_1 < … < T_M` (length `M+1`) defines the M forward Libors
//! `L_n` for `n=0..M-1`, where `L_n(t)` is the simple-compounded forward
//! rate over `[T_n, T_{n+1}]` observed at time `t ≤ T_n`. Day-count
//! fractions `δ_n = T_{n+1} − T_n`. The active index function
//! `η(t) = min{ n : T_n > t }` is the next reset date; only Libors with
//! `n ≥ η(t)` are alive.
//!
//! ## Output
//!
//! `Array2<T>` of shape `(M, n_steps)`. Row `n` stores the path
//! `L_n(t_0), L_n(t_1), …, L_n(t_{n_steps-1})`. After `t_k > T_n`,
//! `L_n` is **frozen** at its reset value `L_n(T_n^−)` (a Libor that has
//! already reset cannot evolve further — its value is locked in).
//!
//! ## Scope and references
//!
//! - Brace, Gatarek, Musiela (1997), "The Market Model of Interest Rate
//!   Dynamics", Mathematical Finance 7(2), 127-155.
//! - Jamshidian (1997), "LIBOR and swap market models and measures",
//!   Finance and Stochastics 1(4), 293-330.
//! - Glasserman (2003), "Monte Carlo Methods in Financial Engineering",
//!   Springer, §3.7 ("LIBOR market model").
//! - Glasserman, Zhao (2000), "Arbitrage-free discretization of lognormal
//!   forward LIBOR and swap rate models", Finance and Stochastics 4(1),
//!   35-68 — the original predictor-corrector scheme; this implementation
//!   uses the simpler frozen-drift log-Euler.
//! - Hula (2011), "Discrete LIBOR Market Model Analogy", arXiv:1108.4260
//!   — discrete construction whose continuous-time limit is the LMM.
//!
//! Volatilities `σ_n(t)` are taken **time-homogeneous** (constant per
//! Libor) for v0; piecewise-constant vol can be added later by extending
//! `sigma` to `Array2<T>` of shape `(M, n_steps)`.
//!
//! Correlations `ρ` default to the identity (independent factors per
//! Libor); supply a custom positive-definite matrix via `with_correlation`.
//!

use ndarray::Array1;
use ndarray::Array2;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::linalg::cholesky_lower;
use crate::linalg::validate_correlation;
use crate::precondition::assert_entries;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Drift-coupled LIBOR Market Model (BGM/Jamshidian) under the spot-LIBOR measure; the module
/// header gives the scope, the dynamics and the references.
#[derive(Debug, Clone)]
pub struct Lmm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Tenor dates `T_0 < T_1 < … < T_M`, length `M+1`. Strictly increasing.
  pub tenor: Array1<T>,
  /// Initial forward Libor curve `L_n(0)` for `n=0..M-1`. Length `M`.
  /// Each entry must be positive.
  pub l0: Array1<T>,
  /// Per-Libor volatilities `σ_n` (constant in time for v0). Length `M`.
  pub sigma: Array1<T>,
  /// Lower-triangular Cholesky factor `L` of the forward correlation matrix
  /// `ρ = L Lᵀ`. Square `M × M`. `None` ⇒ identity (independent factors).
  pub chol: Option<Array2<T>>,
  /// Grid points from `t_0 = 0` to the horizon; `dt = horizon / (n − 1)`.
  pub n: usize,
  /// Simulation horizon `t_horizon`. Defaults to `T_M` (last tenor date)
  /// when `None`. Must satisfy `0 < t_horizon ≤ T_M`.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Lmm<T, S> {
  /// Construct an LMM with independent forward factors (identity correlation).
  pub fn new(
    tenor: Array1<T>,
    l0: Array1<T>,
    sigma: Array1<T>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    check_n(n);
    check_t(t);
    Self::check_tenor(&tenor);
    Self::check_l0(&l0);
    Self::check_sigma(&sigma);
    Self::check_lengths(&tenor, &l0, &sigma);
    Self {
      backend: Cpu,
      tenor,
      l0,
      sigma,
      chol: None,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Lmm<T, S, B> {
  fn check_tenor(tenor: &Array1<T>) {
    assert_entries("tenor", "tenor[i].is_finite()", tenor, |v| v.is_finite());
    assert!(
      tenor.len() >= 2,
      "tenor.len() must satisfy `tenor.len() >= 2`, got tenor.len() = {:?}",
      tenor.len()
    );
    for i in 0..tenor.len() - 1 {
      assert!(
        tenor[i + 1] > tenor[i],
        "tenor[{}] must satisfy `tenor[i + 1] > tenor[i]`, got tenor[{}] = {:?}, tenor[{i}] = {:?}",
        i + 1,
        i + 1,
        tenor[i + 1],
        tenor[i]
      );
    }
  }

  fn check_l0(l0: &Array1<T>) {
    assert_entries("l0", "l0[i].is_finite()", l0, |v| v.is_finite());
    assert_entries("l0", "l0[i] > T::zero()", l0, |v| v > T::zero());
  }

  fn check_sigma(sigma: &Array1<T>) {
    assert_entries("sigma", "sigma[i].is_finite()", sigma, |v| v.is_finite());
    assert_entries("sigma", "sigma[i] >= T::zero()", sigma, |v| v >= T::zero());
  }

  /// The joint shape, which no setter checks: a chain of them changes the Libor count one array
  /// at a time.
  fn check_lengths(tenor: &Array1<T>, l0: &Array1<T>, sigma: &Array1<T>) {
    assert!(
      l0.len() == tenor.len() - 1,
      "l0.len() must satisfy `l0.len() == tenor.len() - 1`, got l0.len() = {:?}",
      l0.len()
    );
    assert!(
      sigma.len() == tenor.len() - 1,
      "sigma.len() must satisfy `sigma.len() == tenor.len() - 1`, got sigma.len() = {:?}",
      sigma.len()
    );
  }

  /// Every array check again, where a sampler or a launch reads arrays set one at a time.
  fn validate_lmm_inputs(&self) {
    Self::check_tenor(&self.tenor);
    Self::check_l0(&self.l0);
    Self::check_sigma(&self.sigma);
    Self::check_lengths(&self.tenor, &self.l0, &self.sigma);
  }

  /// Sets the forwards' correlation `rho` (`M × M`) and factors it. Panics unless `rho` is a
  /// finite, symmetric, unit-diagonal, positive-definite matrix with `M = l0.len()` rows.
  pub fn with_correlation(mut self, rho: Array2<T>) -> Self {
    assert!(
      rho.nrows() == self.l0.len(),
      "rho.nrows() must satisfy `rho.nrows() == l0.len()`, got rho.nrows() = {:?}",
      rho.nrows()
    );
    assert_entries("rho", "rho[(i, j)].is_finite()", &rho, |v| v.is_finite());
    validate_correlation(&rho);
    self.chol = Some(cholesky_lower(&rho));
    self
  }

  /// Replace `tenor`. Panics if the new tenor has fewer than two dates or
  /// is not strictly increasing — the same checks `new(...)` runs on its
  /// own `tenor` argument. Does not cross-check `tenor.len() - 1` against
  /// `l0`/`sigma`'s *current* length: eagerly doing so would make it
  /// impossible to ever change the Libor count via chained setters (every
  /// ordering of `.with_tenor(..).with_l0(..).with_sigma(..)` has an
  /// intermediate state where exactly one of the three hasn't caught up
  /// yet). A leftover mismatch by the time `.sample()` runs still panics
  /// there, same as indexing a too-short array inside `Lmm::new`'s own
  /// equivalent misuse would.
  pub fn with_tenor(mut self, tenor: Array1<T>) -> Self {
    Self::check_tenor(&tenor);
    self.tenor = tenor;
    self
  }

  /// Replace `l0`. Panics if any entry is not strictly positive. If the new
  /// `l0` has a different length than the current one, the cached
  /// correlation Cholesky factor `chol` (if any) is no longer
  /// shape-compatible and cannot be recomputed from public fields alone —
  /// only the original `rho` matrix determines it, and that matrix is not
  /// retained after [`with_correlation`](Lmm::with_correlation) decomposes
  /// it — so it is invalidated (reset to `None`, the identity-correlation
  /// default) rather than left stale. A same-length replacement leaves
  /// `chol` untouched, since its value never depended on `l0`'s contents,
  /// only its length.
  pub fn with_l0(mut self, l0: Array1<T>) -> Self {
    Self::check_l0(&l0);
    let shape_changed = l0.len() != self.l0.len();
    self.l0 = l0;
    if shape_changed {
      self.chol = None;
    }
    self
  }

  /// Replace `sigma`. Panics if any entry is negative.
  pub fn with_sigma(mut self, sigma: Array1<T>) -> Self {
    Self::check_sigma(&sigma);
    self.sigma = sigma;
    self
  }

  /// Replace the number of simulation grid points `n`, all else unchanged.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Lmm<T, S, B> {
  /// The horizon, the last tenor date when omitted.
  fn horizon(&self) -> T {
    self.t.unwrap_or(self.tenor[self.l0.len()])
  }

  /// The grid spacing, the horizon itself for a single point.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.horizon() / T::from_usize_(self.n - 1)
    } else {
      self.horizon()
    }
  }

  /// The number of reset dates passed at each step's start — the host loop's
  /// `eta` — tabulated for the step that produces each grid point, with the
  /// time accumulated step by step exactly as the host accumulates it, so a
  /// reset landing on a grid time is crossed at the same step on both. The
  /// rates below it are frozen; the drift sums start one below it.
  fn reset_count_curve(&self) -> Vec<T> {
    let m = self.l0.len();
    let dt = self.dt();
    let mut t_now = T::zero();
    (0..self.n)
      .map(|i| {
        if i > 1 {
          t_now += dt;
        }
        let mut eta = 0usize;
        while eta < m && self.tenor[eta] <= t_now {
          eta += 1;
        }
        T::from_usize_(eta)
      })
      .collect()
  }
}

/// The Euler engine's view of the curve: the rows padded into the four slots
/// of one launch, an absent rate travelling with a zero volatility and an
/// identity row of the factor so it neither moves nor enters a drift.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct LmmLaunch<'a, T: FloatExt, S: SeedExt, B>(&'a Lmm<T, S, B>);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for LmmLaunch<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for LmmLaunch<'_, T, S, B>
{
  type Output = [Array1<T>; 4];
  type Sampler<'s>
    = LmmLaunchSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> LmmLaunchSampler<T, S> {
    LmmLaunchSampler {
      inner: <Lmm<T, S, B> as ProcessExt<T>>::sampler(self.0),
    }
  }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct LmmLaunchSampler<T: FloatExt, S: SeedExt> {
  inner: LmmSampler<T, S>,
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for LmmLaunchSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for LmmLaunchSampler<T, S> {
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
  for LmmLaunch<'_, T, S, B>
{
  /// Padded into the four slots. A wider curve never reaches a launch —
  /// [`ProcessExt::sample`] keeps it on the host — so asking here is a caller
  /// bypassing that guard; the horizon is held to the host's own bounds.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let p = self.0;
    p.validate_lmm_inputs();
    assert!(
      p.device_ready(),
      "Lmm: a launch carries at most {} forwards; sample through `ProcessExt`, which keeps a wider \
       curve on the host",
      crate::euler::CORRELATED_STREAMS
    );
    let m = p.l0.len();
    let (horizon, t_max) = (p.horizon(), p.tenor[m]);
    assert!(
      horizon > T::zero() && horizon <= t_max,
      "horizon must satisfy `horizon > T::zero() && horizon <= t_max`, got horizon = {horizon:?}, t_max = {t_max:?}"
    );
    let mut l = [T::zero(); 10];
    let mut at = 0;
    for row in 0..4 {
      for col in 0..=row {
        l[at] = match &p.chol {
          Some(chol) if row < m && col < m => chol[(row, col)],
          _ => {
            if row == col {
              T::one()
            } else {
              T::zero()
            }
          }
        };
        at += 1;
      }
    }
    crate::euler::EulerSpec::LiborMarket4 {
      sigma: std::array::from_fn(|i| if i < m { p.sigma[i] } else { T::zero() }),
      delta: std::array::from_fn(|i| {
        if i < m {
          p.tenor[i + 1] - p.tenor[i]
        } else {
          T::zero()
        }
      }),
      l,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    let p = self.0;
    std::array::from_fn(|i| if i < p.l0.len() { p.l0[i] } else { T::zero() })
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.horizon()
  }

  fn time_step(&self) -> T {
    self.0.dt()
  }

  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.0.reset_count_curve()])
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

/// The launch's four slots back into the process's own `m × n` matrix.
fn rows_to_matrix<T: FloatExt>(slots: [Array1<T>; 4], m: usize) -> Array2<T> {
  let n = slots[0].len();
  let mut out = Array2::<T>::zeros((m, n));
  for (i, row) in slots.iter().take(m).enumerate() {
    out.row_mut(i).assign(row);
  }
  out
}

backend_switch!([T: FloatExt, S: SeedExt] Lmm<T, S> { tenor, l0, sigma, chol, n, t, seed } via euler);

mod sampler;

#[doc(hidden)]
pub use sampler::LmmSampler;

#[cfg(test)]
mod tests;
