// docs: quant#cds-index-and-cdo-tranches
//! Backs the CDS index and tranche example on the quant catalog page.

use chrono::NaiveDate;
use ndarray::Array1;
use stochastic_rs::quant::credit::index::CdsIndex;
use stochastic_rs::quant::credit::index::flat_survival;
use stochastic_rs::quant::credit::tranche::CdoTranche;
use stochastic_rs::quant::credit::tranche::PoolName;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;

#[test]
fn index_upfront_and_tranche_spreads() {
  let discount = DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let date = |y: i32, m: u32, d: u32| NaiveDate::from_ymd_opt(y, m, d).unwrap();

  // A 125-name index at a 100 bp coupon: fair spread and the ISDA upfront for a 120 bp quote.
  let index = CdsIndex::homogeneous(
    vec![flat_survival(0.02); 125],
    0.4,
    0.01,
    10_000_000.0,
    date(2026, 3, 20),
    date(2031, 6, 20),
  );
  let fair = index.fair_spread(date(2026, 3, 20), &discount);
  let upfront = index.isda_upfront(date(2026, 3, 20), &discount, 0.012, 0.4);
  assert!(fair > 0.0 && upfront > 0.0);

  // Equity and mezzanine tranches on the same pool under a 30 % Gaussian copula correlation.
  let pool: Vec<PoolName> = (0..125)
    .map(|_| PoolName {
      weight: 1.0 / 125.0,
      recovery: 0.4,
      survival: flat_survival(0.02),
    })
    .collect();
  let times: Vec<f64> = (1..=5).map(|i| i as f64).collect();
  let equity =
    CdoTranche::new(0.0, 0.03, 0.05, times.clone(), 1.0, 0.3).valuation(&pool, &discount);
  let mezz = CdoTranche::new(0.03, 0.07, 0.01, times, 1.0, 0.3).valuation(&pool, &discount);
  assert!(equity.fair_spread > mezz.fair_spread && mezz.fair_spread > 0.0);
}
