//! Closed-form checks of the exposure profile and the adjustments on a
//! Brownian mark-to-market, and the Hull–White swap engine.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;

use super::irs::HullWhiteSwapExposure;
use super::*;
use crate::credit::survival_curve::HazardInterpolation;
use crate::curves::InterpolationMethod;

fn flat_discount(r: f64) -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![r; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn flat_hazard(h: f64) -> SurvivalCurve<f64> {
  SurvivalCurve::from_hazard_rates(
    &Array1::from_vec(vec![1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![h, h, h]),
    HazardInterpolation::PiecewiseConstantHazard,
  )
}

/// Brownian MtM `V_t = σ W_t` on the dates `times`, `paths` rows, pinned seed.
fn brownian_mtm(sigma: f64, times: &[f64], paths: usize, seed: u64) -> Array2<f64> {
  let normal = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
  let mut z = vec![0.0; paths * times.len()];
  normal.fill_slice(&mut z);
  let mut mtm = Array2::<f64>::zeros((paths, times.len()));
  for p in 0..paths {
    let mut w = 0.0;
    let mut prev = 0.0;
    for (c, &t) in times.iter().enumerate() {
      w += (t - prev).sqrt() * z[p * times.len() + c];
      prev = t;
      mtm[(p, c)] = sigma * w;
    }
  }
  mtm
}

const TIMES: [f64; 4] = [0.5, 1.0, 1.5, 2.0];

/// `E[max(σW_t, 0)] = σ√t / √(2π)` and the PFE quantile is `σ√t Φ⁻¹(q)`.
#[test]
fn brownian_profile_matches_the_closed_forms() {
  let sigma = 10.0;
  let best = [1_u64, 2, 3]
    .into_iter()
    .map(|seed| {
      let profile = ExposureProfile::from_mtm(
        &brownian_mtm(sigma, &TIMES, 40_000, seed),
        TIMES.to_vec(),
        0.95,
      );
      let mut worst = 0.0_f64;
      for (i, &t) in TIMES.iter().enumerate() {
        let scale = sigma * t.sqrt();
        let epe = scale / (2.0 * std::f64::consts::PI).sqrt();
        worst = worst.max((profile.epe[i] - epe).abs() / epe);
        worst = worst.max((profile.ene[i] - epe).abs() / epe);
        worst = worst.max((profile.pfe[i] - scale * ndtri(0.95)).abs() / (scale * ndtri(0.95)));
      }
      worst
    })
    .fold(f64::INFINITY, f64::min);
  assert!(best < 0.03, "worst relative deviation {best}");
}

/// CVA with flat hazard and flat discounting against the analytic EPE plugged
/// into the same rectangle rule; DVA mirrors it and the funding adjustments
/// cancel for a symmetric exposure.
#[test]
fn adjustments_match_their_closed_forms_on_a_symmetric_exposure() {
  let (sigma, h, r, lgd) = (10.0, 0.02, 0.03, 0.6);
  let epe: Vec<f64> = TIMES
    .iter()
    .map(|t| sigma * t.sqrt() / (2.0 * std::f64::consts::PI).sqrt())
    .collect();
  let profile = ExposureProfile::from_expected(TIMES.to_vec(), epe.clone(), epe.clone());
  let discount = flat_discount(r);
  let survival = flat_hazard(h);
  let mut want = 0.0;
  let mut prev = 0.0_f64;
  for (i, &t) in TIMES.iter().enumerate() {
    want += (-r * t).exp() * epe[i] * ((-h * prev).exp() - (-h * t).exp());
    prev = t;
  }
  want *= lgd;
  let got = cva(&profile, &survival, &discount, lgd);
  assert!(
    (got - want).abs() < 1e-9 * want.max(1.0),
    "cva {got} vs {want}"
  );
  assert!((dva(&profile, &survival, &discount, lgd) - got).abs() < 1e-12);
  assert!(fva(&profile, &discount, 0.01).abs() < 1e-12);
  assert!(
    fca(&profile, &discount, 0.01) > 0.0
      && (fca(&profile, &discount, 0.01) - fba(&profile, &discount, 0.01)).abs() < 1e-12
  );
  assert!(bilateral_cva(&profile, &survival, &flat_hazard(0.0), &discount, lgd) - got < 1e-12);
  assert!(bilateral_cva(&profile, &survival, &survival, &discount, lgd) < got);
  assert_eq!(cva(&profile, &flat_hazard(0.0), &discount, lgd), 0.0);
  assert_eq!(cva(&profile, &survival, &discount, 0.0), 0.0);
  assert!(
    profile.peak_epe() > 0.0
      && profile.average_epe() > 0.0
      && profile.average_epe() < profile.peak_epe()
  );
}

/// The Hull–White engine starts a par swap near zero, builds exposure and
/// closes it at maturity; CVA rises with the counterparty hazard.
#[test]
fn hull_white_swap_exposure_is_shaped_like_a_swap() {
  let curve = flat_discount(0.03);
  let times: Vec<f64> = (1..=5).map(|i| i as f64).collect();
  let mut swap = HullWhiteSwapExposure::new(0.1, 0.01, 1_000_000.0, 0.0, times.clone(), 1.0)
    .with_steps_per_year(12);
  swap.fixed_rate = swap.par_rate(&curve);
  assert!(
    swap.value_at(&curve, 0.0, 0.03).abs() < 1e-6 * swap.notional,
    "par swap at inception"
  );
  let profile = swap.profile(&curve, 4_000, 0.95, Deterministic::new(5));
  assert!(profile.epe[0] > 0.0 && profile.epe[1] > profile.epe[0] * 0.5);
  assert_eq!(profile.epe[4], 0.0);
  assert!(profile.pfe[1] >= profile.epe[1]);
  let discount = flat_discount(0.03);
  let low = cva(&profile, &flat_hazard(0.01), &discount, 0.6);
  let high = cva(&profile, &flat_hazard(0.05), &discount, 0.6);
  assert!(low > 0.0 && high > low, "cva {low} -> {high}");
  let again = swap.profile(&curve, 4_000, 0.95, Deterministic::new(5));
  assert_eq!(profile, again);
}

#[test]
#[should_panic(expected = "one MtM column per exposure date")]
fn rejects_mismatched_dates() {
  let _ = ExposureProfile::from_mtm(&Array2::<f64>::zeros((3, 2)), vec![1.0, 2.0, 3.0], 0.95);
}
