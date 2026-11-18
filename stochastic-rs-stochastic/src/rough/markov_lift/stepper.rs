use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;

use crate::rough::kernel::RlKernel;
use crate::traits::FloatExt;
use crate::volterra::lift::VolterraLift;

/// The Markov-lift stepper for Riemann–Liouville Volterra SDEs: [`VolterraLift`] at [`RlKernel`],
/// with `f` and `g` taking the state alone, since these SDEs are time-homogeneous.
#[derive(Debug, Clone)]
pub struct MarkovLift<T: FloatExt> {
  inner: VolterraLift<T, RlKernel<T>>,
}

impl<T: FloatExt> MarkovLift<T> {
  /// A stepper for `kernel` at step $\delta t$; panics unless `dt > 0`.
  #[must_use]
  pub fn new(kernel: RlKernel<T>, dt: T) -> Self {
    Self {
      inner: VolterraLift::new(kernel, dt),
    }
  }

  /// The lift's per-node constants and boundary terms, which a device launch ships instead of the
  /// kernel: the device runs the recursion of [`VolterraLift::simulate`].
  pub(crate) fn lift(&self) -> &VolterraLift<T, RlKernel<T>> {
    &self.inner
  }

  /// Integrate a single path. `dw` carries Brownian increments on the same
  /// grid as the output (length $n{-}1$).
  pub fn simulate<F, G>(&self, x0: T, f: F, g: G, dw: &[T]) -> Array1<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    self.inner.simulate(x0, |_t, x| f(x), |_t, x| g(x), dw)
  }

  /// Integrate $m$ independent paths from the Brownian increments `dw` of shape $(m, n{-}1)$ into
  /// an $(m, n)$ matrix; [`VolterraLift::simulate_batch`] has the cache tiling.
  pub fn simulate_batch<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    self
      .inner
      .simulate_batch(x0, |_t, x| f(x), |_t, x| g(x), dw)
  }

  /// [`simulate_batch`](Self::simulate_batch) with the outer tile loop on rayon; `f` and `g` must
  /// be `Send + Sync`.
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
