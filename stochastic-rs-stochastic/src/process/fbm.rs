//! # fBM
//!
//! $$
//! \mathbb E[B_t^H B_s^H]=\tfrac12\left(t^{2H}+s^{2H}-|t-s|^{2H}\right)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Fbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst parameter (`0 < H < 1`) controlling roughness and memory.
  pub hurst: T,
  /// Number of discrete time points in the generated path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Fbm::default().with_hurst(0.3)`.
///
/// **Cache note**: the embedded `fgn: Fgn<T, Unseeded, Cpu>` holds the
/// expensive FFT/eigenvalue cache and is always constructed with the
/// literal `Unseeded` — never consulted for randomness (`FbmSampler`
/// draws through a Gaussian source built from the *outer* `self.seed`
/// and only borrows `fgn` for its cached FFT plan/eigenvalues), so unlike
/// [`Vasicek`](crate::interest::vasicek::Vasicek)'s embedded `Ou` there is
/// no seed-derivation subtlety here: every setter that feeds `fgn`
/// rebuilds it with the exact expression `new()` itself uses.
impl<T: FloatExt, S: SeedExt> Fbm<T, S, Cpu> {
  pub fn new(hurst: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      n,
      t,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }

  /// Replace `hurst`; rebuilds the embedded `fgn`.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    self.hurst = hurst;
    self.fgn = Fgn::new(hurst, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the embedded
  /// `fgn`. Panics if `n < 2`, matching `new()`'s own assertion.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.fgn = Fgn::new(self.hurst, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the embedded `fgn`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.fgn = Fgn::new(self.hurst, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// H=0.7, t=1 — a textbook Fbm parameterization. n=252 — one trading year
/// of daily steps (this crate's `Default` convention).
impl<T: FloatExt> Default for Fbm<T, Unseeded, Cpu> {
  fn default() -> Self {
    Self::new(T::from_f64_fast(0.7), 252, Some(T::one()), Unseeded)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fbm<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FbmSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler: it reuses the inner [`Fgn`]'s `Arc`-shared FFT plan and
  /// eigenvalues and owns a Gaussian source seeded from `self.seed.derive()`,
  /// matching the legacy `sample()` stream on the first call. As with
  /// [`Fgn::sampler`](crate::noise::fgn::Fgn), even GPU backends sample on the
  /// CPU here — batch through [`sample_par`](Self::sample_par) for the GPU.
  fn sampler(&self) -> FbmSampler<'_, T, S, B> {
    FbmSampler {
      fbm: self,
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed.derive()),
    }
  }

  /// The `m` fGN noises are generated in one batched backend call, then each
  /// path is assembled (cumulative sum) on the host across all cores.
  ///
  /// **Reproducibility on `Cpu`/`Accelerate`.** Same guarantee as
  /// [`Fgn::sample_par`](crate::noise::fgn::Fgn::sample_par) on those two
  /// backends (`Cpu`: bit-identical; `Accelerate`: thread-count-independent
  /// seed consumption, but not bit-identical — see
  /// [`FgnBackend`]'s doc), with one added wrinkle this
  /// override alone has to get right: the embedded `self.fgn` is always
  /// [`Unseeded`] (never consulted for randomness — see this type's own
  /// doc), so the batch is driven by `self.seed` passed in explicitly here,
  /// not `self.fgn.noise_batch`'s own (dead) seed. Getting that backwards
  /// was the actual bug this fixes: passing `self.fgn`'s seed silently
  /// ignored `self.seed` entirely, so a `Deterministic`-seeded
  /// `Fbm::sample_par` used to draw fresh randomness on every call
  /// regardless of the pinned seed.
  ///
  /// **On the GPU backends the same seed reaches the launch.** The device
  /// `FgnBackend` impls draw their launch seed from the `seed: &S2` handed
  /// to `noise_batch` — `self.seed` here, not the embedded `fgn`'s dead
  /// `Unseeded` one — so two `Deterministic`-seeded `Fbm`s built from the
  /// same seed value produce the same device paths (and consecutive calls on
  /// one advance the stream, as on the host), subject to the cross-driver
  /// caveat in [`FgnBackend`]'s table.
  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Self::Output>, DeviceError> {
    crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`Fbm`] sampling state: borrows the process for its inner [`Fgn`]
/// (FFT plan + eigenvalues) and owns the Gaussian source, so a Monte-Carlo
/// loop pays the `SimdNormal` setup once. The path is the cumulative sum of an
/// fGn increment vector, with `B_0^H = 0`.
#[doc(hidden)]
pub struct FbmSampler<'a, T: FloatExt, S: SeedExt, B> {
  fbm: &'a Fbm<T, S, B>,
  normal: SimdNormal<T>,
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
    let mut fgn = Array1::<T>::zeros(self.fbm.fgn.out_len);
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

/// The Euler engine's view of fBm: the additive family, whose step is the
/// increment itself, so accumulating fGN increments is one kernel rather than
/// a host pass over the batch.
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
    Some(crate::euler::FgnSpec {
      sqrt_eigenvalues: self.fgn.sqrt_eigenvalues.as_slice().expect("contiguous"),
      n: self.fgn.n,
      offset: self.fgn.offset,
      hurst: self.fgn.hurst.to_f64().unwrap_or(0.5),
      t: self.fgn.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
    })
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fbm<T, S> { hurst, n, t, seed } via fgn euler);

impl<T: FloatExt, S: SeedExt, B> Fbm<T, S, B> {
  /// Calculate the Malliavin derivative
  ///
  /// The Malliavin derivative of the fractional Brownian motion is given by:
  /// D_s B^H_t = 1 / Γ(H + 1/2) (t - s)^{H - 1/2}
  ///
  /// where B^H_t is the fractional Brownian motion with Hurst parameter H in Mandelbrot-Van Ness representation as
  /// B^H_t = 1 / Γ(H + 1/2) ∫_0^t (t - s)^{H - 1/2} dW_s
  /// which is a truncated Wiener integral.
  pub fn malliavin(&self) -> Array1<T> {
    let dt = self.fgn.dt();
    let mut m = Array1::zeros(self.n);
    let g = stochastic_rs_distributions::special::gamma(self.hurst.to_f64().unwrap() + 0.5);

    for i in 0..self.n {
      m[i] = T::one() / T::from_f64_fast(g)
        * (T::from_usize_(i) * dt).powf(self.hurst - T::from_f64_fast(0.5));
    }

    m
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
    // Seeded `Fbm::sample(&self)` advances the internal atomic seed state
    // each call, so a single instance yields `m` independent paths
    // deterministically.
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

  // `fbm_fractal_dimension_matches_theory` lives in
  // `stochastic-rs-stats/tests/fractal_dim_validation.rs` because it exercises
  // the `FractalDim` estimator from the stats crate.
}
