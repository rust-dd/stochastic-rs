//! The fit to the returns times `unit` read in the returns' own units.

use ndarray::Array1;
use ndarray::Array2;

use super::GarchKind;
use super::GarchSpec;
use super::MeanSpec;
use crate::StatsError;
use crate::stationarity::common::per_unit;

/// `ω`'s place in the flat layout, after `μ` where the mean is estimated.
fn omega_index(spec: &GarchSpec) -> usize {
  usize::from(spec.mean == MeanSpec::Constant)
}

/// Parameter `i`'s power of `unit`: `μ` one, a level recursion's `ω`, a variance, two.
fn power(spec: &GarchSpec, i: usize) -> usize {
  if spec.mean == MeanSpec::Constant && i == 0 {
    1
  } else if i == omega_index(spec) && spec.kind != GarchKind::Egarch {
    2
  } else {
    0
  }
}

/// EGARCH's `ω` moves by `−2 ln unit (1 − Σβ)`, as each `ln σ²` moves by `−2 ln unit`.
fn egarch_shift(unit: f64) -> f64 {
  2.0 * unit.ln()
}

/// `natural`, fitted to the returns times `unit`, in the returns' units; `Numerical` where one
/// leaves the double range there, or a nonzero one that scales rounds to zero.
pub(super) fn params(spec: &GarchSpec, natural: &[f64], unit: f64) -> Result<Vec<f64>, StatsError> {
  let mut out = natural
    .iter()
    .enumerate()
    .map(|(i, &v)| per_unit(v, unit, power(spec, i)))
    .collect::<Vec<_>>();
  if spec.kind == GarchKind::Egarch {
    let beta = natural[natural.len() - spec.q..].iter().sum::<f64>();
    out[omega_index(spec)] -= egarch_shift(unit) * (1.0 - beta);
  }
  let lost =
    |i: usize| !out[i].is_finite() || (power(spec, i) > 0 && out[i] == 0.0 && natural[i] != 0.0);
  if (0..out.len()).any(lost) {
    return Err(StatsError::Numerical {
      stage: "data scale",
    });
  }
  Ok(out)
}

/// The covariance `J C Jᵀ` in the returns' units and its standard errors, rooted at unit scale so
/// that each stays finite where its variance leaves the double range.
pub(super) fn covariance(
  spec: &GarchSpec,
  c: &Array2<f64>,
  unit: f64,
) -> (Array2<f64>, Array1<f64>) {
  let k = c.nrows();
  let mut c = c.clone();
  if spec.kind == GarchKind::Egarch {
    // `∂ω/∂β_j = 2 ln unit`: each `β` row joins `ω`'s, then each `β` column.
    let (w, shift) = (omega_index(spec), egarch_shift(unit));
    let betas = k - spec.q..k;
    for x in 0..k {
      let from_betas = betas.clone().map(|b| c[[b, x]]).sum::<f64>();
      c[[w, x]] += shift * from_betas;
    }
    for x in 0..k {
      let from_betas = betas.clone().map(|b| c[[x, b]]).sum::<f64>();
      c[[x, w]] += shift * from_betas;
    }
  }
  let scaled = Array2::from_shape_fn((k, k), |(i, j)| {
    per_unit(
      per_unit(c[[i, j]], unit, power(spec, i)),
      unit,
      power(spec, j),
    )
  });
  let std_errors =
    Array1::from_iter((0..k).map(|i| per_unit(c[[i, i]].sqrt(), unit, power(spec, i))));
  (scaled, std_errors)
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;

  use super::covariance;
  use super::params;
  use crate::garch::GarchSpec;

  /// EGARCH's `ω` takes `2 ln unit` of each `β`'s variance and covariance, `J C Jᵀ` exactly.
  #[test]
  fn egarch_omega_takes_the_betas_share_of_the_covariance() {
    let spec = GarchSpec::egarch(1, 1);
    let unit = 0.25_f64;
    let c = Array2::from_shape_fn((5, 5), |(i, j)| if i == j { 1.0 + i as f64 } else { 0.1 });
    let (scaled, std_errors) = covariance(&spec, &c, unit);
    let s = 2.0 * unit.ln();
    let mut j = Array2::<f64>::eye(5);
    j[[0, 0]] = 1.0 / unit;
    j[[1, 4]] = s;
    let want = j.dot(&c).dot(&j.t());
    for a in 0..5 {
      for b in 0..5 {
        let (got, w) = (scaled[[a, b]], want[[a, b]]);
        assert!(
          (got - w).abs() <= 1e-12 * w.abs().max(1.0),
          "({a}, {b}): {got} against {w}"
        );
      }
    }
    assert!((std_errors[1] - want[[1, 1]].sqrt()).abs() <= 1e-12 * want[[1, 1]].sqrt());
    let mapped = params(&spec, &[0.5, -0.1, 0.1, 0.0, 0.9], unit).unwrap();
    assert_eq!(mapped[0], 2.0);
    assert!((mapped[1] - (-0.1 - s * (1.0 - 0.9))).abs() < 1e-15);
  }
}
