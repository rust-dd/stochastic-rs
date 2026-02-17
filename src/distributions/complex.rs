//! # Complex
//!
//! $$
//! \mathbb P(X\in A)=\int_A f_X(x)dx\ \text{or}\ \sum_{x\in A}p_X(x)
//! $$
//!
use num_complex::Complex;
use num_traits::Num;
use rand::distr::Distribution;
use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct ComplexDistribution<Re, Im = Re> {
  re: Re,
  im: Im,
}

impl<Re, Im> ComplexDistribution<Re, Im> {
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