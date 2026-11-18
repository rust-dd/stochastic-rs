//! Transient impact propagators: the mid-price change is past signed flow convolved with a decaying
//! kernel $G$: on trade signs at $\delta=0$, on signed volume at $\delta=1$ (Almgren et al. 2005).
//!
//! $$
//! p_t - p_0 = \sum_{s\le t} G(t - s)\,\varepsilon_s\,|v_s|^\delta + \xi_t
//! $$
//!
//! Reference: Bouchaud et al. (2004), Quant. Finance 4(2), 176–190, DOI 10.1080/14697680400000022
//! Reference: Gatheral (2010), Quantitative Finance 10(7), 749–759, DOI 10.1080/14697680903373692

use std::fmt::Display;

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::RealExt;

/// Propagator-kernel family.
#[derive(Default, Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ImpactKernel<T: RealExt> {
  /// Pure power-law decay $G(\ell) = G_0\,(1 + \ell)^{-\beta}$ for
  /// $\beta \in (0, 1)$ (Bouchaud et al. 2004).
  #[default]
  PowerLaw,
  /// Exponential decay $G(\ell) = G_0\,e^{-\beta\,\ell}$ (Obizhaeva-Wang
  /// resilience).
  Exponential,
  /// Custom user-supplied kernel function.
  Custom(fn(T) -> T),
}

impl<T: RealExt> Display for ImpactKernel<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::PowerLaw => write!(f, "Power-law"),
      Self::Exponential => write!(f, "Exponential"),
      Self::Custom(_) => write!(f, "Custom"),
    }
  }
}

impl<T: RealExt> ImpactKernel<T> {
  /// Evaluate the kernel weight at non-negative lag `lag`.
  pub fn evaluate(&self, lag: T, g0: T, beta: T) -> T {
    match self {
      Self::PowerLaw => g0 * (T::one() + lag).powf(-beta),
      Self::Exponential => g0 * (-beta * lag).exp(),
      Self::Custom(f) => f(lag),
    }
  }
}

/// Propagator impact at the latest tick; `signed_volumes[i]` is the signed flow of interval
/// `i` on the kernel's time grid.
pub fn propagator_price_impact<T: RealExt>(
  signed_volumes: ArrayView1<T>,
  kernel: ImpactKernel<T>,
  g0: T,
  beta: T,
) -> T {
  let n = signed_volumes.len();
  if n == 0 {
    return T::zero();
  }
  let mut acc = T::zero();
  for s in 0..n {
    let lag = T::from_usize_(n - 1 - s);
    let weight = kernel.evaluate(lag, g0, beta);
    acc += weight * signed_volumes[s];
  }
  acc
}

/// The impact-driven mid-price path $\{p_t\}_{t \ge 0}$, $p_0 = 0$: the signed-volume stream
/// convolved with the propagator kernel.
pub fn propagator_impact_path<T: RealExt>(
  signed_volumes: ArrayView1<T>,
  kernel: ImpactKernel<T>,
  g0: T,
  beta: T,
) -> Array1<T> {
  let n = signed_volumes.len();
  let mut out = Array1::<T>::zeros(n);
  for t in 0..n {
    let mut acc = T::zero();
    for s in 0..=t {
      let lag = T::from_usize_(t - s);
      acc += kernel.evaluate(lag, g0, beta) * signed_volumes[s];
    }
    out[t] = acc;
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn power_law_kernel_starts_at_g0() {
    let k = ImpactKernel::<f64>::PowerLaw;
    assert!(approx(k.evaluate(0.0, 0.5, 0.7), 0.5, 1e-12));
  }

  #[test]
  fn exponential_kernel_decays_as_expected() {
    let k = ImpactKernel::<f64>::Exponential;
    let v = k.evaluate(2.0, 1.0, 0.5);
    assert!(approx(v, (-1.0_f64).exp(), 1e-12));
  }

  #[test]
  fn custom_kernel_passes_through() {
    let k = ImpactKernel::<f64>::Custom(|x| x * x);
    assert!(approx(k.evaluate(3.0, 0.0, 0.0), 9.0, 1e-12));
  }

  #[test]
  fn impact_path_matches_per_step_total() {
    let v = ndarray::array![1.0_f64, -1.0, 1.0, 1.0];
    let path = propagator_impact_path(v.view(), ImpactKernel::PowerLaw, 1.0, 0.5);
    for t in 0..v.len() {
      let expected =
        propagator_price_impact(v.slice(ndarray::s![..=t]), ImpactKernel::PowerLaw, 1.0, 0.5);
      assert!(approx(path[t], expected, 1e-12));
    }
  }

  #[test]
  fn empty_sequence_yields_zero_impact() {
    let v = Array1::<f64>::zeros(0);
    let p = propagator_price_impact(v.view(), ImpactKernel::PowerLaw, 1.0, 0.5);
    assert!(approx(p, 0.0, 1e-12));
  }
}
