//! A stateless law bound to one seeded SIMD stream, and the stream-state pieces its kernels share.

use std::fmt;
use std::fmt::Debug;

use num_traits::Zero;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_core::simd_rng::derive_fork_seed;
use stochastic_rs_core::simd_rng::derive_seed;

use crate::traits::distribution::DistributionSampler;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Values per `sample_matrix` worker: a fork plus a rayon task cost a few hundred nanoseconds.
const MIN_PAR_CHUNK: usize = 16 * 1024;

/// `sample_matrix`'s worker count, a function of `total` alone: the fork sequence must not see the pool size.
pub(crate) fn worker_count(total: usize) -> usize {
  total.div_ceil(MIN_PAR_CHUNK).max(1).min(total)
}

/// A stateless law bound to one seeded SIMD stream; draws take `&mut self` and a clone is a snapshot.
#[derive(Clone, Debug)]
pub struct Seeded<D: SimdDistribution, R: SimdRngExt = SimdRng> {
  dist: D,
  state: D::State<R>,
  basis: u64,
}

impl<D: SimdDistribution, R: SimdRngExt> Seeded<D, R> {
  /// Draws the law's sub-streams from `seed` in the law's fixed order.
  pub fn new<S: SeedExt>(dist: D, seed: &S) -> Self {
    let (state, basis) = dist.init::<R, S>(seed);
    Self { dist, state, basis }
  }

  /// The law this stream draws.
  pub fn dist(&self) -> &D {
    &self.dist
  }

  /// The law, with the stream state dropped.
  pub fn into_dist(self) -> D {
    self.dist
  }

  /// The stream state, for crate kernels that draw through one of its parts directly.
  pub(crate) fn state_mut(&mut self) -> &mut D::State<R> {
    &mut self.state
  }

  /// An independent child stream for worker `stream_idx`; advances this stream's fork basis.
  pub fn fork(&mut self, stream_idx: u64) -> Self {
    let call_basis = derive_seed(&mut self.basis);
    let child = derive_fork_seed(call_basis, stream_idx);
    Self::new(self.dist.clone(), &Deterministic::new(child))
  }
}

impl<D: SimdKernel, R: SimdRngExt> Seeded<D, R> {
  /// One draw from the stream's single-draw buffer, which the bulk kernel refills when empty.
  #[inline]
  pub fn sample(&mut self) -> D::Item {
    self.dist.next(&mut self.state)
  }
}

impl<D: SimdKernel, R: SimdRngExt> Iterator for Seeded<D, R> {
  type Item = D::Item;

  #[inline]
  fn next(&mut self) -> Option<D::Item> {
    Some(self.sample())
  }
}

impl<D: SimdDistribution + Default, R: SimdRngExt> Default for Seeded<D, R> {
  fn default() -> Self {
    Self::new(D::default(), &Unseeded)
  }
}

impl<D: SimdDistribution, R: SimdRngExt> Sealed for Seeded<D, R> {}

impl<D: SimdKernel, R: SimdRngExt> DistributionSampler<D::Item> for Seeded<D, R> {
  #[inline]
  fn fill_slice(&mut self, out: &mut [D::Item]) {
    self.dist.fill(&mut self.state, out);
  }

  #[inline]
  fn fork(&mut self, stream_idx: u64) -> Self {
    Seeded::fork(self, stream_idx)
  }
}

/// Fixed-size single-draw buffer; `idx == N` means empty.
#[doc(hidden)]
#[derive(Clone)]
pub struct Buffered<T: Copy + Zero, const N: usize> {
  buf: [T; N],
  idx: usize,
}

impl<T: Copy + Zero, const N: usize> Buffered<T, N> {
  pub fn new() -> Self {
    Self {
      buf: [T::zero(); N],
      idx: N,
    }
  }

  /// Pops one value, refilling the whole buffer through `refill` when it is empty.
  #[inline]
  pub fn pop(&mut self, refill: impl FnOnce(&mut [T])) -> T {
    if self.idx >= N {
      self.refill(refill);
    }
    let x = self.buf[self.idx];
    self.idx += 1;
    x
  }

  #[inline]
  fn refill(&mut self, refill: impl FnOnce(&mut [T])) {
    refill(&mut self.buf);
    self.idx = 0;
  }
}

impl<T: Copy + Zero, const N: usize> Default for Buffered<T, N> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: Copy + Zero, const N: usize> Debug for Buffered<T, N> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Buffered")
      .field("n", &N)
      .field("idx", &self.idx)
      .finish()
  }
}

/// One engine plus one single-draw buffer, the state of every leaf kernel.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct StreamState<T: Copy + Zero, R: SimdRngExt, const N: usize> {
  pub rng: R,
  pub buf: Buffered<T, N>,
}
