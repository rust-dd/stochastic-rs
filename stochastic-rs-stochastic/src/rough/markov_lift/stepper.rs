use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;

use super::simd::RoughSimd;
use crate::rough::kernel::RlKernel;
use crate::traits::FloatExt;
use crate::volterra::lift::VolterraLift;

/// Single-path and batch Markov-lift stepper for $f,g$-driven RL-Volterra SDEs.
///
/// A thin, backward-compatible wrapper over the kernel-generic
/// [`VolterraLift`]: `MarkovLift<T>` is `VolterraLift<T, RlKernel<T>>`, with
/// `f`/`g` coefficients that take only the state (RL-Volterra SDEs in this
/// crate are time-homogeneous) rather than `VolterraLift`'s `(t, x)` pair.
#[derive(Debug, Clone)]
pub struct MarkovLift<T: FloatExt> {
  inner: VolterraLift<T, RlKernel<T>>,
}

impl<T: FloatExt> MarkovLift<T> {
  /// Build a stepper for the given kernel and step size $\delta t > 0$.
  #[must_use]
  pub fn new(kernel: RlKernel<T>, dt: T) -> Self {
    Self {
      inner: VolterraLift::new(kernel, dt),
    }
  }
}

impl<T: FloatExt + RoughSimd> MarkovLift<T> {
  /// Integrate a single path. `dw` carries Brownian increments on the same
  /// grid as the output (length $n{-}1$).
  pub fn simulate<F, G>(&self, x0: T, f: F, g: G, dw: &[T]) -> Array1<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    self.inner.simulate(x0, |_t, x| f(x), |_t, x| g(x), dw)
  }

  /// Integrate $m$ independent paths driven by the given Brownian increment
  /// matrix `dw` of shape $(m, n{-}1)$. Returns an $(m, n)$ path matrix. See
  /// [`VolterraLift::simulate_batch`] for the cache-tiling strategy.
  pub fn simulate_batch<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    self
      .inner
      .simulate_batch(x0, |_t, x| f(x), |_t, x| g(x), dw)
  }

  /// Same as [`simulate_batch`](Self::simulate_batch) but parallelises the
  /// outer tile loop with rayon — combines per-core SIMD path-batching with
  /// multi-core scheduling. Requires `f` and `g` to be `Send + Sync`.
  pub fn simulate_batch_par<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T) -> T + Send + Sync,
    G: Fn(T) -> T + Send + Sync,
  {
    self
      .inner
      .simulate_batch_par(x0, |_t, x| f(x), |_t, x| g(x), dw)
  }
}

/// Path block size for [`MarkovLift::simulate_batch`]. Re-exported from
/// [`VolterraLift`]'s own tiling constant, which now owns the definition.
pub use crate::volterra::lift::BATCH_TILE;
