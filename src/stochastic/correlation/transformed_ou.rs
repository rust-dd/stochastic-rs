//! General transformed OU stochastic correlation (Section 2.1).

use ndarray::Array1;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Bounded transformation mapping ℝ → (−1, 1) for constructing
/// stochastic correlation processes from an underlying OU process.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transformation {
  /// ρ = tanh(x).  Steeper near 0; boundaries approached exponentially.
  Tanh,
  /// ρ = (2/π) arctan(πx/2).  Shallower / heavier-tailed than tanh
  /// (Section 2.2, Eq. 11 in Teng et al. 2016).
  Arctan,
}

impl Transformation {
  /// Apply the forward transformation: x ↦ ρ ∈ (−1, 1).
  #[inline]
  pub fn forward<T: FloatExt>(self, x: T) -> T {
    match self {
      Self::Tanh => x.tanh(),
      Self::Arctan => {
        let pi = T::from_f64_fast(std::f64::consts::PI);
        let two = T::from_f64_fast(2.0);
        (two / pi) * (pi / two * x).atan()
      }
    }
  }

  /// Apply the inverse transformation: ρ ↦ x ∈ ℝ.
  #[inline]
  pub fn inverse<T: FloatExt>(self, rho: T) -> T {
    let lo = T::from_f64_fast(-0.9999);
    let hi = T::from_f64_fast(0.9999);
    let r = rho.clamp(lo, hi);
    match self {
      Self::Tanh => r.atanh(),
      Self::Arctan => {
        let pi = T::from_f64_fast(std::f64::consts::PI);
        let two = T::from_f64_fast(2.0);
        (pi / two * r).tan() / (pi / two)
      }
    }
  }
}

/// General transformed OU stochastic correlation (Section 2.1).
///
/// Simulates the Ornstein-Uhlenbeck process in X-space:
///
/// dX_t = κ(μ − X_t) dt + σ dW_t
///
/// and maps to correlation via ρ_t = f(X_t) where f is a
/// [`Transformation`] (tanh or arctan).
pub struct TransformedOU<T: FloatExt, S: SeedExt = Unseeded> {
  pub kappa: T,
  pub mu: T,
  pub sigma: T,
  pub rho0: T,
  pub transform: Transformation,
  pub n: usize,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> TransformedOU<T> {
  pub fn new(
    kappa: T,
    mu: T,
    sigma: T,
    rho0: T,
    transform: Transformation,
    n: usize,
    t: Option<T>,
  ) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      rho0,
      transform,
      n,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> TransformedOU<T, Deterministic> {
  pub fn seeded(
    kappa: T,
    mu: T,
    sigma: T,
    rho0: T,
    transform: Transformation,
    n: usize,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      rho0,
      transform,
      n,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for TransformedOU<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let n_steps = self.n.saturating_sub(1);
    if self.n == 0 {
      return Array1::zeros(0);
    }
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    let sqrt_dt = dt.sqrt();

    let mut gn = Array1::<T>::zeros(n_steps);
    if let Some(slice) = gn.as_slice_mut() {
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
      normal.fill_slice_fast(slice);
    }

    let mut rho = Array1::zeros(self.n);
    let mut x = self.transform.inverse(self.rho0);
    rho[0] = self.rho0;

    for i in 1..self.n {
      x = x + self.kappa * (self.mu - x) * dt + self.sigma * gn[i - 1];
      rho[i] = self.transform.forward(x);
    }

    rho
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tanh_stays_bounded() {
    let scp = TransformedOU::seeded(
      5.0_f64,
      0.0,
      1.5,
      0.3,
      Transformation::Tanh,
      2000,
      Some(2.0),
      77,
    );
    let path = scp.sample();
    assert_eq!(path.len(), 2000);
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }

  #[test]
  fn arctan_stays_bounded() {
    let scp = TransformedOU::seeded(
      5.0_f64,
      0.0,
      1.5,
      -0.5,
      Transformation::Arctan,
      2000,
      Some(2.0),
      88,
    );
    let path = scp.sample();
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }

  #[test]
  fn transformation_roundtrip() {
    for &x in &[-2.0_f64, -1.0, -0.5, 0.0, 0.3, 1.0, 3.0] {
      let rho_t = Transformation::Tanh.forward(x);
      let x_back: f64 = Transformation::Tanh.inverse(rho_t);
      assert!(
        (x - x_back).abs() < 1e-10,
        "tanh roundtrip failed for x={x}"
      );

      let rho_a = Transformation::Arctan.forward(x);
      let x_back_a: f64 = Transformation::Arctan.inverse(rho_a);
      assert!(
        (x - x_back_a).abs() < 1e-8,
        "arctan roundtrip failed for x={x}"
      );
    }
  }
}
