//! Truncated betas and gammas with a shape beside 1, whose mode sits beside a bound, against mpmath
//! at 60 and 120 digits (`near_unit_shapes.txt`, one law a row).

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

use crate::rel;

const TABLE: &str = include_str!("near_unit_shapes.txt");

/// The quantities after the law's four parameters, as the table's header names them.
const NAMES: [&str; 8] = [
  "median", "cdf(q50)", "cdf(q10)", "cdf(q90)", "pdf(q25)", "pdf(q75)", "mean", "variance",
];

/// Each row's errors, relative, or 0 where the law answers the reference's double.
fn errors(law: &impl DistributionExt, v: &[f64]) -> [f64; 8] {
  let err = |got: f64, want: f64| if got == want { 0.0 } else { rel(got, want) };
  [
    err(law.median().unwrap(), v[4]),
    err(law.cdf(v[4]).unwrap(), v[5]),
    err(law.cdf(v[6]).unwrap(), v[7]),
    err(law.cdf(v[8]).unwrap(), v[9]),
    err(law.pdf(v[10]).unwrap(), v[11]),
    err(law.pdf(v[12]).unwrap(), v[13]),
    err(law.mean().unwrap(), v[14]),
    err(law.variance().unwrap(), v[15]),
  ]
}

#[test]
fn shapes_beside_one_match_mpmath() {
  let mut worst: [(f64, String); 8] = Default::default();
  let mut failed = Vec::new();
  for line in TABLE.lines().filter(|l| !l.starts_with('#')) {
    let mut f = line.split_whitespace();
    let family = f.next().unwrap();
    let v: Vec<f64> = f.map(|s| s.parse().unwrap()).collect();
    let name = format!(
      "T{family}({:e}, {:e}, [{:e}, {:e}])",
      v[0], v[1], v[2], v[3]
    );
    let errs = match family {
      "B" => errors(&SimdTruncatedBeta::<f64>::new(v[0], v[1], v[2], v[3]), &v),
      _ => errors(&SimdTruncatedGamma::<f64>::new(v[0], v[1], v[2], v[3]), &v),
    };
    for (i, e) in errs.into_iter().enumerate() {
      if e > worst[i].0 || e.is_nan() {
        worst[i] = (e, name.clone());
      }
      if e > 1e-12 || e.is_nan() {
        failed.push(format!("{name}: {} off by {e:e}", NAMES[i]));
      }
    }
  }
  for (i, (e, law)) in worst.iter().enumerate() {
    println!("{:>8}: worst {e:.1e} at {law}", NAMES[i]);
  }
  assert!(failed.is_empty(), "{}", failed.join("\n"));
}
