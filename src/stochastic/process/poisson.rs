//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use rand_distr::Distribution;

use crate::distributions::exp::SimdExp;
use crate::simd_rng::SimdRng;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Poisson<T: FloatExt> {
  /// Jump intensity (expected arrivals per unit time).
  pub lambda: T,
  /// Optional fixed number of sampled events.
  /// If set, the process is generated with exactly `n` increments.
  pub n: Option<usize>,
  /// Optional terminal time for horizon-based sampling.
  /// If set (and `n` is `None`), events are sampled up to `t_max`.
  pub t_max: Option<T>,
}

impl<T: FloatExt> Poisson<T> {
  pub fn new(lambda: T, n: Option<usize>, t_max: Option<T>) -> Self {
    Poisson { lambda, n, t_max }
  }
}

impl<T: FloatExt> ProcessExt<T> for Poisson<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let distr = SimdExp::new(self.lambda);

    if let Some(n) = self.n {
      let mut poisson = Array1::<T>::zeros(n);
      if n <= 1 {
        return poisson;
      }

      let mut tail_view = poisson.slice_mut(s![1..]);
      let tail = tail_view
        .as_slice_mut()
        .expect("Poisson output tail must be contiguous");
      let mut rng = SimdRng::new();
      distr.fill_slice(&mut rng, tail);

      let mut acc = T::zero();
      for x in tail.iter_mut() {
        acc += *x;
        *x = acc;
      }
      poisson
    } else if let Some(t_max) = self.t_max {
      let expected = if t_max > T::zero() {
        (self.lambda * t_max).to_f64().unwrap_or(0.0)
      } else {
        0.0
      };
      let cap = if expected.is_finite() && expected > 0.0 {
        (expected.ceil() as usize).saturating_add(1)
      } else {
        1
      };
      let mut poisson = Vec::with_capacity(cap);
      poisson.push(T::zero());
      if t_max <= T::zero() {
        return Array1::from(poisson);
      }

      let mut t = T::zero();
      let mut rng = SimdRng::new();

      while t < t_max {
        t += distr.sample(&mut rng);

        if t < t_max {
          poisson.push(t);
        }
      }

      Array1::from(poisson)
    } else {
      panic!("n or t_max must be provided");
    }
  }
}

py_process_1d!(PyPoisson, Poisson,
  sig: (lambda_, n=None, t_max=None, dtype=None),
  params: (lambda_: f64, n: Option<usize>, t_max: Option<f64>)
);
