//! The truncated laws' analytics against mpmath at two precisions: far tails, widths 1e-2…1e-6, open and singular ends.

use stochastic_rs_distributions::traits::DistributionExt;

mod beta;
mod ends;
mod exp;
mod gamma;
mod held_bounds;
mod mgf_frame;
mod narrow_laws;
mod near_unit_shapes;
mod normal;
mod offsets;
mod parents;
mod range_ends;
mod rungs;
mod slopes;
mod survey;

/// Reference values of one interval; `None` where the reference run printed none.
pub struct Want {
  pub mean: f64,
  pub variance: f64,
  pub skewness: f64,
  pub kurtosis: f64,
  pub pdf_mid: Option<f64>,
  pub cdf_mid: Option<f64>,
  pub median: Option<f64>,
  pub entropy: Option<f64>,
  pub mgf_1: Option<f64>,
}

pub fn rel(got: f64, want: f64) -> f64 {
  ((got - want) / want).abs()
}

/// Relative `rtol` or absolute `atol`, whichever is looser; kurtosis carries 1e-11 against a worst measured 2.2e-12.
pub fn close(name: &str, what: &str, got: f64, want: f64, rtol: f64, atol: f64) {
  assert!(
    (got - want).abs() <= atol || rel(got, want) <= rtol,
    "{name}: {what} = {got:e}, expected {want:e}"
  );
}

/// The levels next to 1 whose quantiles count the mass past an upper clip: the largest double below 1, and `1 − 1e-12`.
pub const NEXT_TO_ONE: [f64; 2] = [1.0 - f64::EPSILON / 2.0, 1.0 - 1e-12];

/// The four moments at `check`'s bars; a reference that overflowed or underflowed is its rounding, matched exactly.
pub fn moments_match(name: &str, law: &impl DistributionExt, want: [f64; 4]) {
  let got = [
    law.mean().unwrap(),
    law.variance().unwrap(),
    law.skewness().unwrap(),
    law.kurtosis().unwrap(),
  ];
  for ((what, got), (want, (rtol, atol))) in ["mean", "variance", "skewness", "kurtosis"]
    .into_iter()
    .zip(got)
    .zip(
      want
        .into_iter()
        .zip([(1e-12, 0.0), (1e-12, 0.0), (1e-12, 1e-14), (1e-11, 0.0)]),
    )
  {
    if want.is_infinite() {
      assert_eq!(got, want, "{name}: {what}");
    } else {
      close(name, what, got, want, rtol, atol);
    }
  }
}

/// A law with a singular lower end at 0: its four moments, its entropy and its cdf at 1e-200, 1e-20 and ½.
pub fn singular_row(name: &str, law: &impl DistributionExt, want: [f64; 8]) {
  moments_match(name, law, [want[0], want[1], want[2], want[3]]);
  close(name, "entropy", law.entropy().unwrap(), want[4], 1e-12, 0.0);
  for (x, cdf) in [1e-200, 1e-20, 0.5].into_iter().zip(&want[5..]) {
    close(
      name,
      &format!("cdf({x:e})"),
      law.cdf(x).unwrap(),
      *cdf,
      1e-12,
      0.0,
    );
  }
}

/// `x` never rises as `p` falls from 0.1 to 1e-323 in quarter decades.
pub fn quantile_is_monotone(name: &str, law: &impl DistributionExt) {
  let mut prev = f64::INFINITY;
  for i in 0..1290 {
    let p = 0.1 * 10_f64.powf(-0.25 * i as f64);
    if p < 1e-323 {
      break;
    }
    let x = law.quantile(p).unwrap();
    assert!(
      x <= prev,
      "{name}: quantile({p:e}) = {x:e} rises past {prev:e}"
    );
    prev = x;
  }
}

/// `mid` is where the reference evaluated pdf and cdf, `scale` the interval's width (or `σ`/`θ` for an open end),
/// `points` three interior points for the quantile round trip, each with `1e-10 < cdf < 1 − 1e-10`.
pub fn check(
  name: &str,
  law: &impl DistributionExt,
  mid: f64,
  scale: f64,
  points: [f64; 3],
  want: &Want,
) {
  close(
    name,
    "mean",
    law.mean().unwrap(),
    want.mean,
    1e-12,
    1e-14 * scale,
  );
  close(
    name,
    "variance",
    law.variance().unwrap(),
    want.variance,
    1e-12,
    0.0,
  );
  close(
    name,
    "skewness",
    law.skewness().unwrap(),
    want.skewness,
    1e-12,
    1e-14,
  );
  close(
    name,
    "kurtosis",
    law.kurtosis().unwrap(),
    want.kurtosis,
    1e-11,
    0.0,
  );
  if let Some(pdf) = want.pdf_mid {
    close(name, "pdf(mid)", law.pdf(mid).unwrap(), pdf, 1e-12, 0.0);
  }
  if let Some(cdf) = want.cdf_mid {
    let atol = if cdf > 1.0 - 1e-12 { 1e-13 } else { 0.0 };
    close(name, "cdf(mid)", law.cdf(mid).unwrap(), cdf, 1e-12, atol);
  }
  if let Some(median) = want.median {
    close(
      name,
      "median",
      law.median().unwrap(),
      median,
      1e-12,
      1e-14 * scale,
    );
  }
  if let Some(entropy) = want.entropy {
    close(name, "entropy", law.entropy().unwrap(), entropy, 1e-12, 0.0);
  }
  if let Some(mgf) = want.mgf_1 {
    close(
      name,
      "mgf(1)",
      law.moment_generating_function(1.0).unwrap(),
      mgf,
      1e-12,
      0.0,
    );
  }
  // The level of a double next to x moves by pdf·ulp(x): the round trip allows that rounding.
  for p in [0.1, 0.5, 0.9] {
    let x = law.quantile(p).unwrap();
    let back = law.cdf(x).unwrap();
    let ulps = 2.0 * law.pdf(x).unwrap() * f64::EPSILON * x.abs();
    assert!(
      (back - p).abs() <= 1e-13 + ulps,
      "{name}: cdf(quantile({p})) = {back}"
    );
  }
  for x in points {
    let p = law.cdf(x).unwrap();
    assert!(
      (1e-10..=1.0 - 1e-10).contains(&p),
      "{name}: point {x} has cdf {p}, not interior"
    );
    let back = law.quantile(p).unwrap();
    assert!(
      (back - x).abs() <= 1e-12 * scale,
      "{name}: quantile(cdf({x})) = {back}"
    );
  }
}
