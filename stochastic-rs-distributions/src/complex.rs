//! # Complex
//!
//! $$
//! \mathbb P(X\in A)=\int_A f_X(x)dx\ \text{or}\ \sum_{x\in A}p_X(x)
//! $$
//!
use num_complex::Complex;
use num_traits::Num;
use rand::Rng;
use rand::distr::Distribution;

#[derive(Clone, Copy, Debug)]
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
