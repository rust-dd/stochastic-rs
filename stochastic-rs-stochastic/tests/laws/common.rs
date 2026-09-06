//! What a law case needs: a statistic taken per path, the standard error of
//! its mean across paths, and the band a published closed form is held to.

use ndarray::Array1;
use num_complex::Complex64;

/// Paths per case. They are independent — each draws its own stream — so the
/// spread across them is the error bar every assertion below is built from,
/// whatever the dependence *inside* a path is.
pub(crate) const PATHS: usize = 256;

/// Points per path. Long enough that the O(1/n) bias of a sample
/// autocorrelation stays well under the band: for an AR(1) at φ = 0.6 that
/// bias is ≈ 2.8/n, against a standard error of √((1 − ρ²)/(n · PATHS)), so
/// at these sizes it is about 0.6 of one standard error.
pub(crate) const N: usize = 16_384;

/// Points dropped from the head of every path. A sampler starts from zero or
/// from the unconditional variance, not from a draw of the stationary law, so
/// the first points are a transient; 1024 is past 10⁻¹⁰ of any persistence
/// used here (0.85^1024 ≈ 10⁻⁷²).
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
  let values: Vec<f64> = paths
    .iter()
    .map(|p| stat(&p.as_slice().expect("contiguous")[BURN..]))
    .collect();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  Estimate {
    mean,
    se: (var / n).sqrt(),
  }
}

/// The estimate covers `theory` within five standard errors.
///
/// Five rather than three: a file of this many assertions run on two
/// architectures — the SIMD stream differs between them, so a seed verified
/// on one is an unverified draw on the other — needs a per-assertion false
/// failure rate near 10⁻⁶, and a wrong closed form is off by tens of percent,
/// hundreds of standard errors away.
pub(crate) fn holds(estimate: Estimate, theory: f64, what: &str) {
  let Estimate { mean, se } = estimate;
  let z = (mean - theory) / se;
  assert!(
    z.abs() < 5.0,
    "{what}: sampled {mean} ± {se}, the law says {theory} ({z:.1} standard errors away)"
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

/// The sample autocorrelation at lag `k`, normalised by the series length as
/// the textbook estimator is, so it is the quantity the published ACF
/// formulas name.
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

/// The autocorrelation of the squared series, the statistic a conditional
/// variance model is read through: the level series of a GARCH is white
/// noise, and everything the model says lives in the squares.
pub(crate) fn squared_autocorr(x: &[f64], k: usize) -> f64 {
  let squares: Vec<f64> = x.iter().map(|v| v * v).collect();
  autocorr(&squares, k)
}

/// Every increment of every path, pooled.
///
/// A Lévy process has independent, identically distributed increments, so the
/// `paths · (n − 1)` differences are that many independent draws of the law
/// at one step — a far sharper sample than the terminal values alone, which
/// number only `paths`.
pub(crate) fn increments(paths: &[Array1<f64>]) -> Vec<f64> {
  paths
    .iter()
    .flat_map(|p| {
      let x = p.as_slice().expect("contiguous");
      x.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>()
    })
    .collect()
}

/// The sample mean of `f` over independent draws, against the value the law
/// gives it, within five standard errors of the sample's own spread.
///
/// The transforms below are what this exists for: `f = cos(ux)` and
/// `sin(ux)` estimate a characteristic function, `f = exp(−ux)` a Laplace
/// transform. All three are bounded, so the standard error is the sample's
/// and needs no moment condition on the process itself — which is what makes
/// this the one statistic that reaches a law with no finite mean.
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
pub(crate) fn characteristic_function_holds(
  sample: &[f64],
  u: f64,
  theory: Complex64,
  what: &str,
) {
  transform_holds(sample, |x| (u * x).cos(), theory.re, &format!("{what}: Re φ({u})"));
  transform_holds(sample, |x| (u * x).sin(), theory.im, &format!("{what}: Im φ({u})"));
}
