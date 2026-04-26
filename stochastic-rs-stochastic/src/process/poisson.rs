//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use rand_distr::Distribution;

use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Poisson<T: FloatExt, S: SeedExt = Unseeded> {
  /// Jump intensity (expected arrivals per unit time).
  pub lambda: T,
  /// Optional fixed number of sampled events.
  pub n: Option<usize>,
  /// Optional terminal time for horizon-based sampling.
  pub t_max: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Poisson<T> {
  pub fn new(lambda: T, n: Option<usize>, t_max: Option<T>) -> Self {
    Poisson {
      lambda,
      n,
      t_max,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Poisson<T, Deterministic> {
  pub fn seeded(lambda: T, n: Option<usize>, t_max: Option<T>, seed: u64) -> Self {
    Poisson {
      lambda,
      n,
      t_max,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> Poisson<T, S> {
  /// Sample with an explicit seed, used by callers like CompoundPoisson.
  pub fn sample_with_seed(&self, seed: u64) -> Array1<T> {
    self.sample_impl(Deterministic(seed))
  }

  /// Core sampling — monomorphised per seed strategy, zero runtime branching.
  pub(crate) fn sample_impl<S2: SeedExt>(&self, mut seed: S2) -> Array1<T> {
    let distr = SimdExp::from_seed_source(self.lambda, &mut seed);

    if let Some(n) = self.n {
      let mut poisson = Array1::<T>::zeros(n);
      if n <= 1 {
        return poisson;
      }

      let mut tail_view = poisson.slice_mut(s![1..]);
      let tail = tail_view
        .as_slice_mut()
        .expect("Poisson output tail must be contiguous");
      let mut rng = seed.rng();
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
      let mut rng = seed.rng();

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

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Poisson<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.sample_impl(self.seed)
  }
}

py_process_1d!(PyPoisson, Poisson,
  sig: (lambda_, n=None, t_max=None, seed=None, dtype=None),
  params: (lambda_: f64, n: Option<usize>, t_max: Option<f64>)
);
