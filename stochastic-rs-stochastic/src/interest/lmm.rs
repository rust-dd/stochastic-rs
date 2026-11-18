//! Forward-LIBOR market model under the spot-LIBOR measure of Glasserman (2003) §3.7, stepped by
//! log-Euler with the drift frozen at each step's start (§3.7.1; Joshi 2003), so forwards stay > 0.
//!
//! $$
//! \frac{dL_n(t)}{L_n(t)} \;=\;
//!   \sigma_n(t)\!\!\sum_{j=\eta(t)}^{n}\!\!
//!     \frac{\delta_j\,\rho_{n,j}\,\sigma_j(t)\,L_j(t)}{1+\delta_j L_j(t)}\,dt
//!   \;+\;\sigma_n(t)\,dW_n^{\!*}(t)
//! $$
//!
//! ```text
//! L_n(t+Δt) = L_n(t)·exp[(μ_n(t) − ½σ_n²) Δt + σ_n √Δt · Z_n]
//! ```
//!
//! Reference: Brace, Gatarek & Musiela (1997), Math. Finance 7(2), 127–155
//! Reference: Jamshidian (1997), Finance Stoch. 1(4), 293–330
//! Reference: Glasserman (2003), "Monte Carlo Methods in Financial Engineering", Springer, §3.7
//! Reference: Glasserman & Zhao (2000), Finance Stoch. 4(1), 35–68, the predictor-corrector scheme
//! Reference: Hula (2011), "Discrete LIBOR Market Model Analogy", arXiv:1108.4260
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

/// Drift-coupled LIBOR Market Model under the spot-LIBOR measure. `sample` returns `(M, n)`:
/// row `n` is `L_n` on the grid, frozen at its reset value once the grid passes `T_n`.
#[derive(Debug, Clone)]
pub struct Lmm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Tenor dates `T_0 < T_1 < … < T_M`, strictly increasing: `L_n` is the simple forward over
  /// `[T_n, T_{n+1}]`, `δ_n = T_{n+1} − T_n`, alive while `n ≥ η(t) = min{n : T_n > t}`.
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
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Lmm<T, S> {
  /// Independent forward factors (identity correlation). Panics unless `n >= 2`, a given `t` is
  /// positive, `tenor` increases strictly, and `l0 > 0`, `sigma >= 0` hold `tenor.len() - 1` rates.
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

  /// Replace `tenor`; panics unless it holds two or more finite, strictly increasing dates. Its
  /// length is checked against `l0` and `sigma` at sampling, so chained setters can change it.
  pub fn with_tenor(mut self, tenor: Array1<T>) -> Self {
    Self::check_tenor(&tenor);
    self.tenor = tenor;
    self
  }

  /// Replace `l0`; panics unless every entry is positive and finite. A new length resets `chol`
  /// to `None` (identity), since the `rho` it factored is not kept; the same length keeps it.
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

  /// Replace the number of simulation grid points `n`; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
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

  /// The host loop's `eta`, the reset dates passed at each step's start, with time accumulated
  /// step by step as the host does, so a reset on a grid time is crossed at the same step on both.
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

/// The Euler engine's view: the rows padded into one launch's four slots; an absent rate has a
/// zero volatility and an identity row of the factor, so it neither moves nor enters a drift.
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
  /// Padded into the four slots, the horizon held to the host's bounds. [`ProcessExt::sample`]
  /// keeps a wider curve on the host, so a launch that asks here bypassed that guard.
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
