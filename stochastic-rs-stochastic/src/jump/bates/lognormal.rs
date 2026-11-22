use std::any::Any;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::traits::FloatExt;

/// Bates' (1996) relative jump `Y − 1 = e^Z − 1`, `Z ~ N(nu, omega²)`, so a jump never takes the
/// price through zero; [`mean`](Self::mean) is the `k` that compensates it.
#[derive(Clone, Copy, Debug)]
pub struct LogNormalJump<T> {
  nu: T,
  omega: T,
  standard: SimdNormal<T>,
}

impl<T: FloatExt> LogNormalJump<T> {
  /// The mean `nu` and deviation `omega` of `Z`; panics unless both are finite and `omega ≥ 0`.
  pub fn new(nu: T, omega: T) -> Self {
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      omega.is_finite(),
      "omega must satisfy `omega.is_finite()`, got omega = {omega:?}"
    );
    assert!(
      omega >= T::zero(),
      "omega must satisfy `omega >= T::zero()`, got omega = {omega:?}"
    );
    Self {
      nu,
      omega,
      standard: SimdNormal::default(),
    }
  }

  /// The mean `nu` of the log jump size `Z`.
  pub fn nu(&self) -> T {
    self.nu
  }

  /// The deviation `omega` of the log jump size `Z`.
  pub fn omega(&self) -> T {
    self.omega
  }

  /// `E[e^Z − 1] = e^{nu + omega²/2} − 1`.
  pub fn mean(&self) -> T {
    (self.nu + T::from_f64_fast(0.5) * self.omega * self.omega).exp_m1()
  }

  /// `law` when it is this law, recognised by type as the device recognises the laws it draws.
  pub(crate) fn of<D: Any>(law: &D) -> Option<&Self> {
    (law as &dyn Any).downcast_ref()
  }

  /// The jump sums of one path on a grid of `points`, the counts' stream seeded from `seed` before
  /// the sizes'; none is seeded on a single point or at a zero rate.
  pub(crate) fn jump_sums<S: SeedExt>(
    &self,
    lambda: T,
    seed: &S,
    points: usize,
    dt: T,
  ) -> LogJumpSums<'_, T> {
    let lambda_dt = (lambda * dt).to_f64().unwrap();
    let draws = if points <= 1 || lambda_dt <= 0.0 {
      None
    } else {
      Some((SimdPoisson::<u32>::new(lambda_dt).seeded(seed), seed.rng()))
    };
    LogJumpSums { law: self, draws }
  }
}

/// Each step's `ΣZ_j` over a `Poisson(lambda·dt)` count `n`, one `N(n·nu, n·omega²)` draw as the
/// kernels draw it, taken as the path steps so no per-path buffer holds them.
pub(crate) struct LogJumpSums<'a, T> {
  law: &'a LogNormalJump<T>,
  draws: Option<(Seeded<SimdPoisson<u32>>, SimdRng)>,
}

impl<T: FloatExt> LogJumpSums<'_, T> {
  pub(crate) fn next_sum(&mut self) -> T {
    let Some((count, rng)) = &mut self.draws else {
      return T::zero();
    };
    let n = count.sample();
    if n == 0 {
      return T::zero();
    }
    let n = T::from_usize_(n as usize);
    self.law.nu * n + self.law.omega * n.sqrt() * self.law.standard.sample(rng)
  }
}

impl<T: FloatExt> Distribution<T> for LogNormalJump<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    (self.nu + self.omega * self.standard.sample(rng)).exp_m1()
  }
}
