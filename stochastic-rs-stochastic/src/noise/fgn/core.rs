//! # Core
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
use std::sync::Arc;

use ndarray::prelude::*;
use ndrustfft::FftHandler;
use ndrustfft::ndfft_inplace_par;
use num_complex::Complex;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::traits::FloatExt;

pub struct Fgn<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Internal FFT length (power-of-two padded).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Model parameter controlling process dynamics.
  pub offset: usize,
  pub(crate) out_len: usize,
  pub(crate) scale: T,
  /// Model parameter controlling process dynamics.
  pub sqrt_eigenvalues: Arc<Array1<T>>,
  /// Model parameter controlling process dynamics.
  pub fft_handler: Arc<FftHandler<T>>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Fgn<T, S> {
  pub fn dt(&self) -> T {
    let step_count = self.out_len.max(1);
    self.t.unwrap_or(T::one()) / T::from_usize_(step_count)
  }
}

impl<T: FloatExt> Fgn<T> {
  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    if !(T::zero()..=T::one()).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let out_len = n;
    let n = n.next_power_of_two();
    let circ_len = 2 * n;
    let f2h = T::from_usize_(2) * hurst;
    let half = T::from_f64_fast(0.5);

    let mut buf = Array1::<Complex<T>>::zeros(circ_len);
    let buf_slice = buf.as_slice_mut().unwrap();
    buf_slice[0] = Complex::new(T::one(), T::zero());
    for k in 1..=n {
      let kf = T::from_usize_(k);
      let val = half
        * ((kf + T::one()).powf(f2h) - T::from_usize_(2) * kf.powf(f2h)
          + (kf - T::one()).powf(f2h));
      buf_slice[k] = Complex::new(val, T::zero());
      if k > 0 && k < n {
        buf_slice[circ_len - k] = Complex::new(val, T::zero());
      }
    }

    let fft_handler = Arc::new(FftHandler::new(circ_len));
    let mut buf_view = buf.view_mut();
    ndfft_inplace_par(&mut buf_view, &*fft_handler, 0);

    let norm = T::from_usize_(circ_len);
    let mut sqrt_eigenvalues = Array1::<T>::uninit(circ_len);
    let eig_slice =
      unsafe { std::slice::from_raw_parts_mut(sqrt_eigenvalues.as_mut_ptr() as *mut T, circ_len) };
    let buf_slice = buf.as_slice().unwrap();
    for (dst, src) in eig_slice.iter_mut().zip(buf_slice.iter()) {
      let lambda = src.re / norm;
      *dst = if lambda > T::zero() {
        lambda.sqrt()
      } else {
        T::zero()
      };
    }
    let sqrt_eigenvalues = unsafe { sqrt_eigenvalues.assume_init() };

    let scale_n = out_len.max(1);

    Self {
      hurst,
      n,
      offset,
      out_len,
      t,
      scale: T::from_usize_(scale_n).powf(-hurst) * t.unwrap_or(T::one()).powf(hurst),
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      fft_handler,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Fgn<T, Deterministic> {
  pub fn seeded(hurst: T, n: usize, t: Option<T>, seed: u64) -> Self {
    let base = Fgn::<T>::new(hurst, n, t);
    Self {
      hurst: base.hurst,
      n: base.n,
      offset: base.offset,
      out_len: base.out_len,
      t: base.t,
      scale: base.scale,
      sqrt_eigenvalues: base.sqrt_eigenvalues,
      fft_handler: base.fft_handler,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> Fgn<T, S> {
  /// Sample fGn using a specific deterministic seed.
  pub fn sample_cpu_with_seed(&self, seed: u64) -> Array1<T> {
    self.sample_cpu_impl(&Deterministic::new(seed))
  }

  pub(crate) fn sample_cpu(&self) -> Array1<T> {
    self.sample_cpu_impl(&self.seed)
  }

  /// Core fGn sampling — monomorphised per seed strategy, zero runtime branching.
  #[inline]
  pub(crate) fn sample_cpu_impl<S2: SeedExt>(&self, seed: &S2) -> Array1<T> {
    let len = 2 * self.n;
    let mut fgn = Array1::<T>::zeros(self.out_len);

    T::with_fgn_complex_scratch(len, |rnd| {
      // SAFETY: Complex<T> is repr(C) with layout {re: T, im: T}, identical to [T; 2]
      let flat = unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * len) };
      let normal = stochastic_rs_distributions::normal::SimdNormal::<T>::from_seed_source(
        T::zero(),
        T::one(),
        seed,
      );
      normal.fill_slice_fast(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for (dst, c) in fgn.iter_mut().zip(src.iter()) {
        *dst = c.re * self.scale;
      }
    });

    fgn
  }

  /// Sample a pair of independent fGn paths using a specific deterministic seed.
  pub(crate) fn sample_pair_cpu_with_seed(&self, seed: u64) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu_impl(&Deterministic::new(seed))
  }

  pub(crate) fn sample_pair_cpu(&self) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu_impl(&self.seed)
  }

  /// Two independent fGn paths per FFT call. Re and Im of the circulant
  /// output are independent zero-mean Gaussians with the same target
  /// covariance — Dietrich & Newsam (1997), Kroese & Botev (2013 §2.2
  /// Step 4, MATLAB listing "two independent fields").
  #[inline]
  pub(crate) fn sample_pair_cpu_impl<S2: SeedExt>(&self, seed: &S2) -> (Array1<T>, Array1<T>) {
    let len = 2 * self.n;
    let mut fgn_re = Array1::<T>::zeros(self.out_len);
    let mut fgn_im = Array1::<T>::zeros(self.out_len);

    T::with_fgn_complex_scratch(len, |rnd| {
      // SAFETY: Complex<T> is repr(C) with layout {re: T, im: T}, identical to [T; 2]
      let flat = unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * len) };
      let normal = stochastic_rs_distributions::normal::SimdNormal::<T>::from_seed_source(
        T::zero(),
        T::one(),
        seed,
      );
      normal.fill_slice_fast(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for ((r, i), c) in fgn_re.iter_mut().zip(fgn_im.iter_mut()).zip(src.iter()) {
        *r = c.re * self.scale;
        *i = c.im * self.scale;
      }
    });

    (fgn_re, fgn_im)
  }
}

#[cfg(test)]
mod tests {
  use super::Fgn;

  fn generate_fgn_paths(h: f64, n: usize, t: f64, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f64>::new(h, n, Some(t));
    let mut out = Vec::with_capacity(m);
    for _ in 0..m {
      out.push(fgn.sample_cpu().to_vec());
    }
    out
  }

  fn unit_lag_covariance(h: f64, k: usize) -> f64 {
    if k == 0 {
      1.0
    } else {
      0.5
        * (((k + 1) as f64).powf(2.0 * h) - 2.0 * (k as f64).powf(2.0 * h)
          + ((k - 1) as f64).powf(2.0 * h))
    }
  }

  fn lag_covariance(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for p in paths {
      for i in 0..(p.len() - lag) {
        s += (p[i] - mean) * (p[i + lag] - mean);
        c += 1;
      }
    }
    s / c as f64
  }

  fn nearest_quantile(sorted: &[f64], p: f64) -> f64 {
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx]
  }

  #[test]
  fn dt_and_scale_use_requested_length_not_fft_padding() {
    let hs = [0.2_f64, 0.7_f64];
    let ns = [3_usize, 17, 1000, 4095];
    let ts = [0.7_f64, 2.0_f64];

    for &h in &hs {
      for &n in &ns {
        for &t in &ts {
          let fgn = Fgn::<f64>::new(h, n, Some(t));

          // Internal FFT length is padded, but dt/scale must follow requested n.
          assert!(fgn.n >= n && fgn.n.is_power_of_two());
          assert!((fgn.dt() - (t / n as f64)).abs() < 1e-15);

          let expected_scale = (n as f64).powf(-h) * t.powf(h);
          assert!((fgn.scale - expected_scale).abs() < 1e-15);
        }
      }
    }
  }

  #[test]
  fn fgn_marginal_distribution_and_covariance_match_theory() {
    let h = 0.72_f64;
    let n = 2048_usize;
    let t = 1.0_f64;
    let m = 1024_usize;
    let paths = generate_fgn_paths(h, n, t, m);

    let mut values = Vec::with_capacity(m * n);
    for p in &paths {
      values.extend_from_slice(p);
    }

    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let var = values
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d
      })
      .sum::<f64>()
      / count;
    let std = var.sqrt();

    let m3 = values
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d * d
      })
      .sum::<f64>()
      / count;
    let m4 = values
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d * d * d
      })
      .sum::<f64>()
      / count;
    let skew = m3 / std.powi(3);
    let excess_kurtosis = m4 / std.powi(4) - 3.0;

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q025 = (nearest_quantile(&sorted, 0.025) - mean) / std;
    let q975 = (nearest_quantile(&sorted, 0.975) - mean) / std;

    let dt = t / n as f64;
    let var_theory = dt.powf(2.0 * h);
    let cov1_theory = var_theory * unit_lag_covariance(h, 1);
    let cov4_theory = var_theory * unit_lag_covariance(h, 4);

    let cov1_emp = lag_covariance(&paths, mean, 1);
    let cov4_emp = lag_covariance(&paths, mean, 4);

    assert!(mean.abs() < 5e-4, "mean too far from zero: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.05,
      "variance mismatch: emp={var}, theory={var_theory}"
    );
    assert!(
      (skew.abs() < 0.05) && (excess_kurtosis.abs() < 0.10),
      "non-Gaussian marginals: skew={skew}, excess_kurtosis={excess_kurtosis}"
    );
    assert!(
      (q025 + 1.96).abs() < 0.10 && (q975 - 1.96).abs() < 0.10,
      "quantile mismatch: q025={q025}, q975={q975}"
    );
    assert!(
      ((cov1_emp / cov1_theory) - 1.0).abs() < 0.05,
      "lag-1 covariance mismatch: emp={cov1_emp}, theory={cov1_theory}"
    );
    assert!(
      ((cov4_emp / cov4_theory) - 1.0).abs() < 0.05,
      "lag-4 covariance mismatch: emp={cov4_emp}, theory={cov4_theory}"
    );
  }

  #[test]
  fn fgn_lag1_correlation_sign_matches_hurst_regime() {
    let n = 2048_usize;
    let t = 1.0_f64;
    let m = 192_usize;

    let low_h = 0.25_f64;
    let high_h = 0.80_f64;

    let low_paths = generate_fgn_paths(low_h, n, t, m);
    let high_paths = generate_fgn_paths(high_h, n, t, m);

    let low_vals: Vec<f64> = low_paths.iter().flatten().copied().collect();
    let high_vals: Vec<f64> = high_paths.iter().flatten().copied().collect();

    let low_mean = low_vals.iter().sum::<f64>() / low_vals.len() as f64;
    let high_mean = high_vals.iter().sum::<f64>() / high_vals.len() as f64;

    let low_var = low_vals
      .iter()
      .map(|x| {
        let d = *x - low_mean;
        d * d
      })
      .sum::<f64>()
      / low_vals.len() as f64;
    let high_var = high_vals
      .iter()
      .map(|x| {
        let d = *x - high_mean;
        d * d
      })
      .sum::<f64>()
      / high_vals.len() as f64;

    let low_cov1 = lag_covariance(&low_paths, low_mean, 1);
    let high_cov1 = lag_covariance(&high_paths, high_mean, 1);

    let low_rho1 = low_cov1 / low_var;
    let high_rho1 = high_cov1 / high_var;

    assert!(
      low_rho1 < -0.10,
      "expected negative lag-1 correlation, got {low_rho1}"
    );
    assert!(
      high_rho1 > 0.10,
      "expected positive lag-1 correlation, got {high_rho1}"
    );
  }

  // `fbm_hurst_and_fractal_dimension_from_fgn_increments` lives in
  // `stochastic-rs-stats/tests/fractal_dim_validation.rs` because it exercises
  // the `FractalDim` estimator from the stats crate.

  fn cross_covariance(a: &[Vec<f64>], b: &[Vec<f64>], mean_a: f64, mean_b: f64) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for (pa, pb) in a.iter().zip(b.iter()) {
      for i in 0..pa.len() {
        s += (pa[i] - mean_a) * (pb[i] - mean_b);
        c += 1;
      }
    }
    s / c as f64
  }

  /// Primary and secondary paths of `sample_pair` must each satisfy the
  /// target marginal law and be mutually independent. The latter is the
  /// Dietrich–Newsam (1997) / Kroese–Botev (2013 §2.2) claim — here we
  /// check the sample cross-correlation vanishes to within Monte-Carlo SE.
  #[test]
  fn sample_pair_independent_paths_match_theory() {
    let h = 0.68_f64;
    let n = 2048_usize;
    let t = 1.0_f64;
    let pairs_count = 1024_usize;

    let fgn = Fgn::<f64>::new(h, n, Some(t));
    let mut prim = Vec::with_capacity(pairs_count);
    let mut sec = Vec::with_capacity(pairs_count);
    for _ in 0..pairs_count {
      let (a, b) = fgn.sample_pair_cpu();
      prim.push(a.to_vec());
      sec.push(b.to_vec());
    }

    let dt = t / n as f64;
    let var_theory = dt.powf(2.0 * h);
    let cov1_theory = var_theory * unit_lag_covariance(h, 1);

    for (label, paths) in [("primary", &prim), ("secondary", &sec)] {
      let values: Vec<f64> = paths.iter().flatten().copied().collect();
      let count = values.len() as f64;
      let mean = values.iter().sum::<f64>() / count;
      let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
      let cov1 = lag_covariance(paths, mean, 1);

      assert!(mean.abs() < 5e-4, "{label}: mean drift {mean}");
      assert!(
        ((var / var_theory) - 1.0).abs() < 0.05,
        "{label}: variance mismatch emp={var} theory={var_theory}"
      );
      assert!(
        ((cov1 / cov1_theory) - 1.0).abs() < 0.05,
        "{label}: lag-1 covariance mismatch emp={cov1} theory={cov1_theory}"
      );
    }

    let mean_p = prim.iter().flatten().sum::<f64>() / (prim.len() * n) as f64;
    let mean_s = sec.iter().flatten().sum::<f64>() / (sec.len() * n) as f64;
    let var_p = prim
      .iter()
      .flatten()
      .map(|x| (*x - mean_p).powi(2))
      .sum::<f64>()
      / (prim.len() * n) as f64;

    let xcov = cross_covariance(&prim, &sec, mean_p, mean_s);
    let correlation = xcov / var_p;
    assert!(
      correlation.abs() < 0.02,
      "primary/secondary correlation {correlation} too large (expected ≈ 0)"
    );
  }

  /// `sample_pair` with an explicit seed must be fully deterministic and
  /// bit-for-bit match two separate `sample_cpu_with_seed` invocations —
  /// the second one using a seed derived by replaying the same SplitMix64
  /// step the single-path variant would have produced. Here we check the
  /// simpler determinism-across-identical-seed property.
  #[test]
  fn sample_pair_is_deterministic_with_seed() {
    let fgn = Fgn::<f64>::new(0.55, 1024, Some(1.0));
    let (a1, b1) = fgn.sample_pair_cpu_with_seed(7);
    let (a2, b2) = fgn.sample_pair_cpu_with_seed(7);
    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
  }
}
