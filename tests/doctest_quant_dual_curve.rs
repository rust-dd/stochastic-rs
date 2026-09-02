// docs: quant#dual-curve-bootstrap-ois-discounting-tenor-forecasting
//! Backs the dual-curve bootstrap example on the quant catalog page.

use ndarray::Array1;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::MultiCurve;
use stochastic_rs::quant::curves::dual_curve::ForecastInstrument;
use stochastic_rs::quant::curves::dual_curve::bootstrap_forecast;

#[test]
fn dual_curve_bootstrap_recovers_the_tenor_basis() {
  // Exogenous OIS discount curve (flat 2 %) and a 3M tenor trading 50 bp above it.
  let ois = DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.25, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.02; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let tenor_df = |t: f64| (-0.025_f64 * t).exp();

  // Market quotes implied by that tenor curve: 3M deposit, 3×6 FRA, par swaps
  // against 3M with annual fixed and quarterly floating payments.
  let mut quotes = vec![
    ForecastInstrument::Deposit {
      maturity: 0.25,
      rate: (1.0 / tenor_df(0.25) - 1.0) / 0.25,
    },
    ForecastInstrument::Fra {
      start: 0.25,
      end: 0.5,
      rate: (tenor_df(0.25) / tenor_df(0.5) - 1.0) / 0.25,
    },
  ];
  for years in [1_usize, 2, 3, 5] {
    let fixed_times: Vec<f64> = (1..=years).map(|i| i as f64).collect();
    let float_times: Vec<f64> = (1..=4 * years).map(|i| 0.25 * i as f64).collect();
    let float_pv: f64 = float_times
      .iter()
      .scan(0.0, |prev, &t| {
        let leg = ois.discount_factor(t) * (tenor_df(*prev) / tenor_df(t) - 1.0);
        *prev = t;
        Some(leg)
      })
      .sum();
    let annuity: f64 = fixed_times
      .iter()
      .scan(0.0, |prev, &t| {
        let leg = (t - *prev) * ois.discount_factor(t);
        *prev = t;
        Some(leg)
      })
      .sum();
    quotes.push(ForecastInstrument::Swap {
      rate: float_pv / annuity,
      fixed_times,
      float_times,
    });
  }

  // Forecast curve against OIS discounting: pseudo-discount factors come back exactly.
  let forecast = bootstrap_forecast(
    &quotes,
    &ois,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0] {
    assert!((forecast.discount_factor(t) - tenor_df(t)).abs() < 1e-9);
  }

  // The multi-curve container reads the 50 bp tenor basis back off the two curves.
  let mut multi = MultiCurve::new(ois);
  multi.add_forecast("3M", forecast);
  let basis = multi.basis_spread("3M", 1.0, 1.25).unwrap();
  assert!((basis - 0.005).abs() < 2e-4, "basis {basis}");
}
