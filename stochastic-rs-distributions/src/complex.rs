//! # Complex
//!
//! $$
//! \mathbb P(X\in A)=\int_A f_X(x)dx\ \text{or}\ \sum_{x\in A}p_X(x)
//! $$
//!
use num_complex::Complex;
use num_traits::Num;
use num_traits::Zero;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexDistribution<Re, Im = Re> {
  re: Re,
  im: Im,
}

impl<Re, Im> ComplexDistribution<Re, Im> {
  /// Creates a complex distribution by pairing two independent
  /// sub-distributions, one per component.
  ///
  /// - `re` — distribution sampled for the real part of the output.
  /// - `im` — distribution sampled for the imaginary part of the output.
  ///
  /// Unlike every other type in this crate, `re`/`im` are not shape or
  /// scale parameters — they are full sub-distributions composed
  /// together, sampled independently on every draw.
  pub fn new(re: Re, im: Im) -> Self {
    ComplexDistribution { re, im }
  }
}

impl<T, Re, Im> Distribution<Complex<T>> for ComplexDistribution<Re, Im>
where
  T: Num + Clone,
  Re: Distribution<T>,
  Im: Distribution<T>,
{
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
    Complex::new(self.re.sample(rng), self.im.sample(rng))
  }
}

impl<Re: SimdDistribution, Im: SimdDistribution> Sealed for ComplexDistribution<Re, Im> {}

impl<Re: SimdDistribution, Im: SimdDistribution> SimdDistribution for ComplexDistribution<Re, Im> {
  type State<R: SimdRngExt> = (Re::State<R>, Im::State<R>);

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> ((Re::State<R>, Im::State<R>), u64) {
    let (re, basis) = self.re.init::<R, S>(seed);
    let (im, _) = self.im.init::<R, S>(seed);
    ((re, im), basis)
  }
}

impl<Re, Im> SimdKernel for ComplexDistribution<Re, Im>
where
  Re: SimdKernel,
  Im: SimdKernel<Item = Re::Item>,
  Re::Item: Num,
{
  type Item = Complex<Re::Item>;

  fn fill<R: SimdRngExt>(
    &self,
    state: &mut (Re::State<R>, Im::State<R>),
    out: &mut [Complex<Re::Item>],
  ) {
    let mut re = [Re::Item::zero(); 64];
    let mut im = [Re::Item::zero(); 64];
    for chunk in out.chunks_mut(64) {
      let n = chunk.len();
      self.re.fill(&mut state.0, &mut re[..n]);
      self.im.fill(&mut state.1, &mut im[..n]);
      for (c, (r, i)) in chunk.iter_mut().zip(re.iter().zip(im.iter())) {
        *c = Complex::new(*r, *i);
      }
    }
  }

  fn next<R: SimdRngExt>(&self, state: &mut (Re::State<R>, Im::State<R>)) -> Complex<Re::Item> {
    Complex::new(self.re.next(&mut state.0), self.im.next(&mut state.1))
  }
}
