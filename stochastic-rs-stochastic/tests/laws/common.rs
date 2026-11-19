//! What a law case needs: a statistic taken per path, the standard error of
//! its mean across paths, and the band a published closed form is held to.

use ndarray::Array1;
use num_complex::Complex64;

/// Paths per case. Each draws its own stream, so the spread across them is the error bar of
/// every assertion, whatever the dependence inside a path.
pub(crate) const PATHS: usize = 256;

/// Points per path, long enough that the O(1/n) bias of a sample autocorrelation stays under
/// the band: for an AR(1) at φ = 0.6 it is ≈ 2.8/n, about 0.6 of one standard error.
pub(crate) const N: usize = 16_384;

/// Points dropped from the head of every path: a sampler starts from zero or the unconditional
/// variance, not a stationary draw, and 1024 outlasts any persistence here (0.85^1024 ≈ 10⁻⁷²).
pub(crate) const BURN: usize = 1_024;

/// A statistic estimated from independent paths.
pub(crate) struct Estimate {
  /// The mean of the per-path values.
  pub(crate) mean: f64,
  /// The standard error of that mean: `sd / √paths`.
  pub(crate) se: f64,
}

/// `stat` applied to every path past the burn-in, as a mean and its standard
/// error.
pub(crate) fn across_paths(paths: &[Array1<f64>], stat: impl Fn(&[f64]) -> f64) -> Estimate {
  estimate(paths, BURN, stat)
}

fn estimate(paths: &[Array1<f64>], burn: usize, stat: impl Fn(&[f64]) -> f64) -> Estimate {
  let values: Vec<f64> = paths
    .iter()
    .map(|p| stat(&p.as_slice().expect("contiguous")[burn..]))
    .collect();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  Estimate {
    mean,
    se: (var / n).sqrt(),
  }
}

/// The estimate covers `theory` within five standard errors. Five, not three: many assertions
/// on two architectures with different SIMD streams need a false-failure rate near 10⁻⁶.
pub(crate) fn holds(estimate: Estimate, theory: f64, what: &str) {
  let Estimate { mean, se } = estimate;
  let z = (mean - theory) / se;
  assert!(
    z.abs() < 5.0,
    "{what}: sampled {mean} ± {se}, the law says {theory} ({z:.1} standard errors away)"
  );
}

/// [`holds`] for the best of several estimates, one per pinned seed: a right law fails only when
/// every seed misses, at the band's rate raised to the number of seeds.
pub(crate) fn holds_on_one_of(estimates: &[Estimate], theory: f64, what: &str) {
  let z = |e: &Estimate| ((e.mean - theory) / e.se).abs();
  let best = estimates.iter().map(z).fold(f64::INFINITY, f64::min);
  assert!(
    best < 5.0,
    "{what}: the law says {theory}, the best of {} seeds is {best:.1} standard errors away: {:?}",
    estimates.len(),
    estimates.iter().map(|e| (e.mean, e.se)).collect::<Vec<_>>()
  );
}

/// Each path's last point — one independent draw of the law at the horizon.
pub(crate) fn terminals(paths: &[Array1<f64>]) -> Vec<f64> {
  paths.iter().map(|p| p[p.len() - 1]).collect()
}

/// The mean of independent per-path values and its standard error.
pub(crate) fn over_paths(values: &[f64]) -> Estimate {
  let n = values.len() as f64;
  let m = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
  Estimate {
    mean: m,
    se: (var / n).sqrt(),
  }
}

/// The variance of independent per-path values and its standard error from the sample's own
/// fourth moment, `Var(s²) = (m₄ − m₂²)/P`, which assumes no normality.
pub(crate) fn spread_over_paths(values: &[f64]) -> Estimate {
  let n = values.len() as f64;
  let m = values.iter().sum::<f64>() / n;
  let m2 = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
  let m4 = values.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
  Estimate {
    mean: m2,
    se: ((m4 - m2 * m2) / n).sqrt(),
  }
}

/// The estimate covers `theory` within five standard errors widened by `bias`, the computed
/// difference between the law of the chain the code steps and the diffusion's.
pub(crate) fn holds_within(estimate: Estimate, theory: f64, bias: f64, what: &str) {
  let Estimate { mean, se } = estimate;
  let band = 5.0 * se + bias.abs();
  assert!(
    (mean - theory).abs() < band,
    "{what}: sampled {mean} ± {se}, the law says {theory} \
     (band {band}, of which {} is the scheme's own bias)",
    bias.abs()
  );
}

/// The sample mean.
pub(crate) fn mean(x: &[f64]) -> f64 {
  x.iter().sum::<f64>() / x.len() as f64
}

/// The sample variance about the sample mean.
pub(crate) fn variance(x: &[f64]) -> f64 {
  let m = mean(x);
  x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

/// The sample autocorrelation at lag `k`, normalised by the series length as the textbook
/// estimator is, so it is the quantity the published ACF formulas name.
pub(crate) fn autocorr(x: &[f64], k: usize) -> f64 {
  let m = mean(x);
  let n = x.len();
  let c0 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
  let ck = (k..n).map(|t| (x[t] - m) * (x[t - k] - m)).sum::<f64>() / n as f64;
  ck / c0
}

/// The sample kurtosis, the fourth moment about the mean over the squared
/// variance. Three for a normal law.
pub(crate) fn kurtosis(x: &[f64]) -> f64 {
  let m = mean(x);
  let v = variance(x);
  x.iter().map(|s| (s - m).powi(4)).sum::<f64>() / x.len() as f64 / v.powi(2)
}

/// The sample autocovariance at lag `k` about a known mean, over the number of products formed:
/// unbiased, where the correlation ratio's `1/n` bias is six standard errors at `ρ = 0.994`.
pub(crate) fn autocovariance(x: &[f64], k: usize, centre: f64) -> f64 {
  (k..x.len())
    .map(|t| (x[t] - centre) * (x[t - k] - centre))
    .sum::<f64>()
    / (x.len() - k) as f64
}

/// The autocorrelation of the squared series, the statistic a conditional variance model is
/// read through: a GARCH's level series is white noise.
pub(crate) fn squared_autocorr(x: &[f64], k: usize) -> f64 {
  let squares: Vec<f64> = x.iter().map(|v| v * v).collect();
  autocorr(&squares, k)
}

/// Every increment of every path, pooled: a Lévy process's `paths · (n − 1)` increments are
/// independent draws of the one-step law, a far sharper sample than `paths` terminal values.
pub(crate) fn increments(paths: &[Array1<f64>]) -> Vec<f64> {
  paths
    .iter()
    .flat_map(|p| {
      let x = p.as_slice().expect("contiguous");
      x.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>()
    })
    .collect()
}

/// The sample mean of `f` against the law's value, within five standard errors. A bounded `f`
/// (`cos(ux)`, `sin(ux)`, `exp(−ux)`) needs no moment, so it reaches a law with no finite mean.
pub(crate) fn transform_holds(sample: &[f64], f: impl Fn(f64) -> f64, theory: f64, what: &str) {
  let values: Vec<f64> = sample.iter().map(|&x| f(x)).collect();
  let n = values.len() as f64;
  let m = values.iter().sum::<f64>() / n;
  let se = (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
  let z = (m - theory) / se;
  assert!(
    z.abs() < 5.0,
    "{what}: sampled {m} ± {se}, the law says {theory} ({z:.1} standard errors away)"
  );
}

/// Both parts of the empirical characteristic function at `u`, against the
/// value the law gives it.
pub(crate) fn characteristic_function_holds(sample: &[f64], u: f64, theory: Complex64, what: &str) {
  transform_holds(
    sample,
    |x| (u * x).cos(),
    theory.re,
    &format!("{what}: Re φ({u})"),
  );
  transform_holds(
    sample,
    |x| (u * x).sin(),
    theory.im,
    &format!("{what}: Im φ({u})"),
  );
}
