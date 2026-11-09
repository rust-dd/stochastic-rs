//! The GH law's bulk kernel and buffered draw, over its GIG-clock and normal sub-streams.

use std::mem::MaybeUninit;

use stochastic_rs_core::simd_rng::SimdRngExt;

use super::GhState;
use super::SimdGeneralizedHyperbolic;
use crate::SimdFloatExt;
use crate::normal::SimdNormal;
use crate::seeded::StreamState;
use crate::traits::distribution::SimdKernel;
use crate::traits::distribution::as_uninit_mut;

const SMALL_GH_THRESHOLD: usize = 16;

impl<T: SimdFloatExt> SimdGeneralizedHyperbolic<T> {
  fn fill_parts<R: SimdRngExt>(
    &self,
    gig: &mut StreamState<T, R, 16>,
    normal: &mut StreamState<T, R, 64>,
    out: &mut [MaybeUninit<T>],
  ) {
    if out.len() < SMALL_GH_THRESHOLD {
      for x in out.iter_mut() {
        let w = self.gig.next(gig);
        let z = SimdNormal::<T>::standard().next(normal);
        x.write(self.mu + self.beta * w + w.sqrt() * z);
      }
      return;
    }
    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);
    let mut wbuf = [T::zero(); 64];
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gig.fill(gig, as_uninit_mut(&mut wbuf));
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf);
      for (sub, (w8, z8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        wbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(zbuf.as_chunks::<8>().0.iter()),
      ) {
        let w = T::simd_from_array(*w8);
        let z = T::simd_from_array(*z8);
        *sub = T::simd_to_array(mu + beta * w + T::simd_sqrt(w) * z).map(MaybeUninit::new);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gig.fill(gig, as_uninit_mut(&mut wbuf[..n]));
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      for i in 0..n {
        rem[i].write(self.mu + self.beta * wbuf[i] + wbuf[i].sqrt() * zbuf[i]);
      }
    }
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGeneralizedHyperbolic<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut GhState<T, R>, out: &mut [MaybeUninit<T>]) {
    self.fill_parts(&mut state.gig, &mut state.normal, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GhState<T, R>) -> T {
    let GhState { gig, normal, buf } = state;
    buf.pop(|b| self.fill_parts(gig, normal, b))
  }
}
