//! # fBM
//!
//! $$
//! \mathbb E[B_t^H B_s^H]=\tfrac12\left(t^{2H}+s^{2H}-|t-s|^{2H}\right)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::gamma;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::malliavin::Malliavin;
use crate::noise::fgn::Fgn;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Brownian motion. Fields are private, since the fGN driver caches a spectrum of
/// `hurst`, `n` and `t`: read them through getters, change them through `with_*`.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::process::fbm::Fbm;
/// let mut p = Fbm::<f64>::new(0.7, 10, None, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Debug, Clone)]
pub struct Fbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  n: usize,
  t: Option<T>,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

/// Every field has a `with_*` setter (`Fbm::default().with_hurst(0.3)`). The embedded `fgn` is
/// `Unseeded` and lends only its FFT cache; the randomness comes from the outer `self.seed`.
impl<T: FloatExt, S: SeedExt> Fbm<T, S, Cpu> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  /// Panics unless a given `t` is positive and finite, `n >= 2`, and `hurst` is finite.
  pub fn new(hurst: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self::check_hurst(hurst);
    check_t(t);
    check_n(n);

    Self {
      hurst,
      n,
      t,
      seed,
      fgn: Fgn::for_grid(hurst, n, t),
    }
  }

  /// Replace `hurst` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `hurst` is finite.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self.fgn = Fgn::for_grid(hurst, self.n, self.t);
    self
  }

  /// Replace `n`; panics if `n < 2`.
  /// Rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.fgn = Fgn::for_grid(self.hurst, n, self.t);
    self
  }

  /// Replace `t` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.fgn = Fgn::for_grid(self.hurst, self.n, t);
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Fbm<T, S, B> {
  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Hurst parameter (`0 < H < 1`) controlling roughness and memory.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Number of discrete time points in the generated path.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Simulation horizon [0, t] for the path (defaults to `1` if `None`).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

/// H=0.7, t=1, a textbook Fbm; 252 points, a trading year of daily closes.
impl<T: FloatExt> Default for Fbm<T, Unseeded, Cpu> {
  fn default() -> Self {
    Self::new(T::from_f64_fast(0.7), 252, Some(T::one()), Unseeded)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::traits::Sealed for Fbm<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fbm<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FbmSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler on the inner [`Fgn`]'s shared FFT plan, with a Gaussian source seeded from
  /// `self.seed.derive()`; a GPU backend batches through [`sample_par`](Self::sample_par).
  fn sampler(&self) -> FbmSampler<'_, T, S, B> {
    FbmSampler {
      fbm: self,
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed.derive()),
    }
  }

  /// The same launch as [`sample_par`](Self::sample_par), mapped; the trait's defaults would map
  /// a batch on the host sampler, a different stream from the device's `sample_par`.
  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    crate::euler::EulerBackend::euler_paths_map(&self.fgn.backend, self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    crate::euler::EulerBackend::euler_paths_map_view(&self.fgn.backend, self, m, f)
  }

  /// One batched fGN call, then a cumulative sum per path. `self.seed` drives the batch on
  /// every backend, not the embedded `fgn`'s `Unseeded`; [`FgnBackend`] states each guarantee.
  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Self::Output>, DeviceError> {
    crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`Fbm`] sampling state: it borrows the process for its inner [`Fgn`] and owns the
/// Gaussian source. The path is the cumulative sum of an fGn vector, with `B_0^H = 0`.
#[doc(hidden)]
#[derive(Debug)]
pub struct FbmSampler<'a, T: FloatExt, S: SeedExt, B> {
  fbm: &'a Fbm<T, S, B>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FbmSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = T::zero();
    if out.len() == 1 {
      return;
    }
    let mut fgn = Array1::<T>::zeros(self.fbm.fgn.n());
    self
      .fbm
      .fgn
      .fill_cpu(&mut self.normal, fgn.as_slice_mut().unwrap());
    let mut acc = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      acc += *inc;
      *dst = acc;
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed for FbmSampler<'_, T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FbmSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fbm output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fbm.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

/// The Euler engine's view: the additive family, whose step is the increment itself, so one
/// kernel accumulates the fGN increments.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Fbm<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Additive
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }

  /// The pipeline that produces this process's increments: the device runs it
  /// and keeps the result in its own buffer.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.fgn.fgn_spec(1))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fbm<T, S> { hurst, n, t, seed } via fgn euler);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> Fbm<T, S, B> {
  /// The path and `D_{t_i} B^H_T = √(Γ(2H+1) sin πH) (T − t_i)^{H−½} / Γ(H+½)` (Mishura 2008, §1.3)
  /// with respect to the Mandelbrot–Van Ness driver; the last entry is ∞ for H < ½.
  pub fn malliavin(&self) -> Malliavin<T, Array1<T>> {
    let dt = self.fgn.dt();
    let h = self.hurst.to_f64().unwrap();
    let g = gamma(h + 0.5);
    let c = (gamma(2.0 * h + 1.0) * (std::f64::consts::PI * h).sin()).sqrt();
    let scale = T::from_f64_fast(c / g);
    let derivative = Array1::from_shape_fn(self.n, |i| {
      scale * (T::from_usize_(self.n - 1 - i) * dt).powf(self.hurst - T::from_f64_fast(0.5))
    });
    Malliavin {
      paths: self.sample(),
      derivative,
    }
  }
}

py_process_1d!(PyFbm, Fbm,
  sig: (hurst, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, n: usize, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::special::erf;

  use super::*;

  #[test]
  fn the_malliavin_derivative_is_the_mandelbrot_van_ness_kernel_of_a_standard_fbm() {
    // √(Γ(2H+1) sin πH) / Γ(H+½), by mpmath at 40 digits.
    for (h, c_h) in [(0.7, 1.0918091308839126), (0.3, 0.730282934079923)] {
      let fbm = || Fbm::<f64, _>::new(h, 17, Some(1.0), Deterministic::new(5));
      let Malliavin {
        paths, derivative, ..
      } = fbm().malliavin();
      assert_eq!(paths, fbm().sample());
      for (i, d) in derivative.iter().take(16).enumerate() {
        let expected = c_h * (1.0 - i as f64 / 16.0).powf(h - 0.5);
        assert!(
          (d - expected).abs() <= 1e-14 * expected,
          "H = {h}, t_{i}: {d} against {expected}"
        );
      }
      assert_eq!(derivative[16], if h < 0.5 { f64::INFINITY } else { 0.0 });
    }
  }

  fn nearest_quantile(sorted: &[f64], p: f64) -> f64 {
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx]
  }

  fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
  }

  fn regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
      num += (x - x_mean) * (y - y_mean);
      den += (x - x_mean) * (x - x_mean);
    }
    num / den
  }

  #[test]
  fn fbm_terminal_marginal_is_gaussian_with_correct_scale() {
    let h = 0.72_f64;
    let t = 1.0_f64;
    let n = 2048_usize;
    let m = 6000_usize;
    // A seeded `Fbm::sample` advances the seed state on each call, so one instance yields `m`
    // independent paths deterministically.
    let fbm = Fbm::new(h, n, Some(t), Deterministic::new(0xFBC0_FFEE_u64));

    let mut endpoints = Vec::with_capacity(m);
    for _ in 0..m {
      let x = fbm.sample();
      endpoints.push(x[n - 1]);
    }

    let mean = endpoints.iter().sum::<f64>() / m as f64;
    let var = endpoints
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d
      })
      .sum::<f64>()
      / m as f64;
    let std = var.sqrt();
    let var_theory = t.powf(2.0 * h);

    let mut sorted = endpoints.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q025 = (nearest_quantile(&sorted, 0.025) - mean) / std;
    let q975 = (nearest_quantile(&sorted, 0.975) - mean) / std;
    let mut ks = 0.0_f64;
    for (i, x) in sorted.iter().enumerate() {
      let z = (*x - mean) / std;
      let f = standard_normal_cdf(z);
      let e1 = ((i + 1) as f64 / m as f64 - f).abs();
      let e2 = (i as f64 / m as f64 - f).abs();
      ks = ks.max(e1.max(e2));
    }

    assert!(mean.abs() < 0.05, "terminal mean too far from 0: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.05,
      "terminal variance mismatch: emp={var}, theory={var_theory}"
    );
    assert!(
      (q025 + 1.96).abs() < 0.1 && (q975 - 1.96).abs() < 0.1,
      "terminal quantile mismatch: q025={q025}, q975={q975}"
    );
    assert!(ks < 0.05, "KS distance too large: {ks}");
  }

  #[test]
  fn fbm_covariance_kernel_matches_theory() {
    let h = 0.72_f64;
    let t_max = 1.0_f64;
    let n = 2048_usize;
    let m = 5000_usize;
    let fbm = Fbm::new(h, n, Some(t_max), Unseeded);
    let dt = t_max / (n as f64 - 1.0);
    let idxs = [n / 4, n / 2, 3 * n / 4, n - 1];

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(m); idxs.len()];
    for _ in 0..m {
      let path = fbm.sample();
      for (j, &idx) in idxs.iter().enumerate() {
        samples[j].push(path[idx]);
      }
    }

    let means: Vec<f64> = samples
      .iter()
      .map(|v| v.iter().sum::<f64>() / v.len() as f64)
      .collect();

    let mut off_diag_rel_sum = 0.0_f64;
    let mut off_diag_count = 0usize;

    for i in 0..idxs.len() {
      for j in i..idxs.len() {
        let mut cov = 0.0;
        for k in 0..m {
          cov += (samples[i][k] - means[i]) * (samples[j][k] - means[j]);
        }
        cov /= m as f64;

        let ti = idxs[i] as f64 * dt;
        let tj = idxs[j] as f64 * dt;
        let cov_theory =
          0.5 * (ti.powf(2.0 * h) + tj.powf(2.0 * h) - (ti - tj).abs().powf(2.0 * h));
        let rel_err = ((cov / cov_theory) - 1.0).abs();
        if i == j {
          assert!(
            rel_err < 0.08,
            "variance mismatch at ({i},{j}): emp={cov}, theory={cov_theory}, rel_err={rel_err}"
          );
        } else {
          off_diag_rel_sum += rel_err;
          off_diag_count += 1;
        }
      }
    }

    let off_diag_mean_rel_err = off_diag_rel_sum / off_diag_count as f64;
    assert!(
      off_diag_mean_rel_err < 0.08,
      "off-diagonal mean relative covariance error too large: {off_diag_mean_rel_err}"
    );
  }

  #[test]
  fn fbm_hurst_scaling_matches_theory() {
    let h = 0.72_f64;
    let t_max = 1.0_f64;
    let n = 2048_usize;
    let m = 2200_usize;
    let fbm = Fbm::new(h, n, Some(t_max), Unseeded);
    let dt = t_max / (n as f64 - 1.0);
    let idxs = [n / 16, n / 8, n / 4, n / 2, n - 1];

    let mut buckets: Vec<Vec<f64>> = vec![Vec::with_capacity(m); idxs.len()];
    for _ in 0..m {
      let path = fbm.sample();
      for (j, &idx) in idxs.iter().enumerate() {
        buckets[j].push(path[idx]);
      }
    }

    let mut xs = Vec::with_capacity(idxs.len());
    let mut ys = Vec::with_capacity(idxs.len());
    for (j, &idx) in idxs.iter().enumerate() {
      let vals = &buckets[j];
      let mean = vals.iter().sum::<f64>() / vals.len() as f64;
      let var = vals
        .iter()
        .map(|x| {
          let d = *x - mean;
          d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
      xs.push((idx as f64 * dt).ln());
      ys.push(var.ln());
    }

    let h_est = 0.5 * regression_slope(&xs, &ys);
    assert!(
      (h_est - h).abs() < 0.05,
      "hurst mismatch from scaling: h_est={h_est}, h={h}"
    );
  }

  // `fbm_fractal_dimension_matches_theory` is in the stats crate's
  // `tests/fractal_dim_validation.rs`, since it exercises the `FractalDim` estimator.
}
