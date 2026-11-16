use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_distributions::normal::SimdNormal;

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
}

impl<T: FloatExt> Distribution<T> for LogNormalJump<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    (self.nu + self.omega * self.standard.sample(rng)).exp_m1()
  }
}
