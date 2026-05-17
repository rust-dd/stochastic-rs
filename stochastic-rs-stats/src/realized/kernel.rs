//! Realized kernels (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).
//!
//! A consistent, noise-robust estimator of integrated variance under
//! microstructure noise:
//!
//! $$
//! K(X) = \gamma_0(X) + \sum_{h=1}^{H} k\!\left(\tfrac{h-1}{H}\right)\bigl(\gamma_h(X)+\gamma_{-h}(X)\bigr),
//! $$
//!
//! where $\gamma_h(X) = \sum_{j=|h|+1}^{n} r_j r_{j-|h|}$ is the lag-$h$ return
//! autocovariance and $k:[0,1]\to[0,1]$ is the chosen kernel weight.
//!
//! Reference: Barndorff-Nielsen, Hansen, Lunde, Shephard, "Designing Realised
//! Kernels to Measure the Ex-Post Variation of Equity Prices in the Presence
//! of Noise", Econometrica, 76(6), 1481-1536 (2008). DOI: 10.3982/ECTA6495

use std::fmt::Display;

use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Kernel weight family.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
  /// $k(x) = 1 - x$ (linear; zero asymptotic bias rate, slow rate of consistency).
  Bartlett,
  /// Parzen kernel — recommended workhorse in BNHLS (2008):
  /// $k(x) = 1 - 6x^2 + 6x^3$ for $x \le 1/2$ and $2(1-x)^3$ otherwise.
  #[default]
  Parzen,
  /// $k(x) = \tfrac{1+\cos(\pi x)}{2}$.
  TukeyHanning,
  /// Tukey-Hanning$_2$: $k(x) = \sin^2\!\bigl(\tfrac{\pi}{2}(1-x)^2\bigr)$.
  TukeyHanning2,
  /// Cubic kernel: $k(x) = 1 - 3x^2 + 2x^3$.
  Cubic,
  /// Quadratic-spectral (Andrews 1991):
  /// $k(x) = \tfrac{25}{12\pi^2 x^2}\!\left(\tfrac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)\right)$.
  QuadraticSpectral,
}

impl Display for KernelType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Bartlett => write!(f, "Bartlett"),
      Self::Parzen => write!(f, "Parzen"),
      Self::TukeyHanning => write!(f, "Tukey-Hanning"),
      Self::TukeyHanning2 => write!(f, "Tukey-Hanning2"),
      Self::Cubic => write!(f, "Cubic"),
      Self::QuadraticSpectral => write!(f, "Quadratic-Spectral"),
    }
  }
}

impl KernelType {
  /// Evaluate the kernel weight at $x \in [0, 1]$ (or the full real line for
  /// the quadratic-spectral kernel).
  pub fn weight<T: FloatExt>(&self, x: T) -> T {
    match self {
      Self::Bartlett => {
        let one = T::one();
        if x >= one { T::zero() } else { one - x }
      }
      Self::Parzen => {
        if x >= T::one() {
          T::zero()
        } else if x <= T::from_f64_fast(0.5) {
          let x2 = x * x;
          T::one() - T::from_f64_fast(6.0) * x2 + T::from_f64_fast(6.0) * x2 * x
        } else {
          let one_minus = T::one() - x;
          T::from_f64_fast(2.0) * one_minus * one_minus * one_minus
        }
      }
      Self::TukeyHanning => {
        if x >= T::one() {
          T::zero()
        } else {
          let pi = T::from_f64_fast(std::f64::consts::PI);
          (T::one() + (pi * x).cos()) / T::from_f64_fast(2.0)
        }
      }
      Self::TukeyHanning2 => {
        if x >= T::one() {
          T::zero()
        } else {
          let pi_half = T::from_f64_fast(std::f64::consts::FRAC_PI_2);
          let arg = pi_half * (T::one() - x) * (T::one() - x);
          arg.sin() * arg.sin()
        }
      }
      Self::Cubic => {
        if x >= T::one() {
          T::zero()
        } else {
          let x2 = x * x;
          T::one() - T::from_f64_fast(3.0) * x2 + T::from_f64_fast(2.0) * x2 * x
        }
      }
      Self::QuadraticSpectral => {
        let xf = x.to_f64().unwrap();
        if xf.abs() < 1e-12 {
          T::one()
        } else {
          let z = 6.0 * std::f64::consts::PI * xf / 5.0;
          let val =
            (25.0 / (12.0 * std::f64::consts::PI.powi(2) * xf * xf)) * (z.sin() / z - z.cos());
          T::from_f64_fast(val)
        }
      }
    }
  }
}

/// Suggested bandwidth $H^*$ for a flat-top realized kernel
/// (BNHLS 2008, Table 4 — Parzen "rule of thumb"): $H = c \cdot n^{3/5}$
/// with $c \approx 3.5134\,\xi^{4/5}$ and the noise-to-signal ratio $\xi^2$
/// estimated from the data. Falls back to $\xi = 1$ for the default heuristic.
pub fn parzen_default_bandwidth(n: usize, xi: f64) -> usize {
  let nn = n as f64;
  let h = 3.5134_f64 * xi.abs().powf(0.8) * nn.powf(0.6);
  h.round().max(1.0) as usize
}

/// Realized kernel estimator with the chosen weight family and bandwidth `h`.
///
/// `returns` are intraday log-returns. `h` should typically be set via
/// [`parzen_default_bandwidth`] or the optimal-bandwidth analytical result of
/// BNHLS (2008).
pub fn realized_kernel<T: FloatExt>(returns: ArrayView1<T>, kernel: KernelType, h: usize) -> T {
  let n = returns.len();
  if n == 0 {
    return T::zero();
  }
  let gamma_0 = autocovariance(returns, 0);
  let mut acc = gamma_0;
  if h == 0 {
    return acc;
  }
  let h_t = T::from_usize_(h);
  for lag in 1..=h {
    let g = autocovariance(returns, lag);
    let arg = T::from_usize_(lag - 1) / h_t;
    let w = kernel.weight(arg);
    acc += T::from_f64_fast(2.0) * w * g;
  }
  acc
}

fn autocovariance<T: FloatExt>(returns: ArrayView1<T>, lag: usize) -> T {
  let n = returns.len();
  if lag >= n {
    return T::zero();
  }
  let mut acc = T::zero();
  for j in lag..n {
    acc += returns[j] * returns[j - lag];
  }
  acc
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn iid_normal(seed: u64, n: usize, std: f64) -> Array1<f64> {
    let dist = SimdNormal::<f64>::new(0.0, std, &stochastic_rs_core::simd_rng::Deterministic::new(seed));
    let mut out = Array1::<f64>::zeros(n);
    dist.fill_slice_fast(out.as_slice_mut().unwrap());
    out
  }

  fn noisy_efficient_path(seed: u64, n: usize, sigma: f64, omega: f64) -> Array1<f64> {
    let dx = SimdNormal::<f64>::new(0.0, sigma, &stochastic_rs_core::simd_rng::Deterministic::new(seed));
    let dn = SimdNormal::<f64>::new(0.0, omega, &stochastic_rs_core::simd_rng::Deterministic::new(seed.wrapping_add(1)));
    let mut steps = vec![0.0_f64; n];
    dx.fill_slice_fast(&mut steps);
    let mut noise = vec![0.0_f64; n + 1];
    dn.fill_slice_fast(&mut noise);
    let mut y = vec![0.0_f64; n + 1];
    y[0] = noise[0];
    for i in 1..=n {
      y[i] = y[i - 1] - noise[i - 1] + steps[i - 1] + noise[i];
    }
    Array1::from(y)
  }

  #[test]
  fn weight_at_zero_is_one() {
    for k in [
      KernelType::Bartlett,
      KernelType::Parzen,
      KernelType::TukeyHanning,
      KernelType::TukeyHanning2,
      KernelType::Cubic,
      KernelType::QuadraticSpectral,
    ] {
      let w: f64 = k.weight(0.0_f64);
      assert!((w - 1.0).abs() < 1e-9, "kernel {k:?} k(0)={w}");
    }
  }

  #[test]
  fn weight_at_one_is_zero() {
    for k in [
      KernelType::Bartlett,
      KernelType::Parzen,
      KernelType::TukeyHanning,
      KernelType::TukeyHanning2,
      KernelType::Cubic,
    ] {
      let w: f64 = k.weight(1.0_f64);
      assert!(w.abs() < 1e-9, "kernel {k:?} k(1)={w}");
    }
  }

  #[test]
  fn h_zero_recovers_realized_variance() {
    let r = iid_normal(31, 1_000, 0.01);
    let rk = realized_kernel(r.view(), KernelType::Parzen, 0);
    let rv: f64 = r.iter().map(|x| x * x).sum();
    assert!((rk - rv).abs() < 1e-12);
  }

  #[test]
  fn kernel_close_to_rv_for_iid_returns() {
    let r = iid_normal(41, 10_000, 0.01);
    let rv: f64 = r.iter().map(|x| x * x).sum();
    let h = parzen_default_bandwidth(10_000, 0.5);
    let rk = realized_kernel(r.view(), KernelType::Parzen, h);
    assert!((rk - rv).abs() / rv < 0.2);
  }

  #[test]
  fn kernel_corrects_first_order_noise_bias() {
    let n = 20_000;
    let sigma = 0.005_f64;
    let omega = 0.003_f64;
    let y = noisy_efficient_path(53, n, sigma, omega);
    let dy = Array1::from_iter((1..=n).map(|i| y[i] - y[i - 1]));
    let iv = (n as f64) * sigma.powi(2);
    let rv: f64 = dy.iter().map(|v| v * v).sum();
    let h = parzen_default_bandwidth(n, omega / sigma);
    let rk = realized_kernel(dy.view(), KernelType::Parzen, h);
    assert!((rk - iv).abs() < (rv - iv).abs());
  }
}
