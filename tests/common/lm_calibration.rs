use ndarray::Array1;
use stochastic_rs::prelude::OptionType;
use stochastic_rs::quant::OptionQuote;
use stochastic_rs::quant::QuantError;
use stochastic_rs::quant::calibration::bsm::BsmCalibrator;
use stochastic_rs::quant::calibration::bsm::BsmParams;
use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::calibration::heston::HestonJacobianMethod;
use stochastic_rs::quant::calibration::levy::LevyCalibrator;
use stochastic_rs::quant::calibration::levy::LevyModelType;
use stochastic_rs::quant::calibration::levy::MarketSlice;
use stochastic_rs::quant::calibration::sabr::SabrCalibrator;
use stochastic_rs::quant::pricing::bsm::BsmCoc;
use stochastic_rs::quant::pricing::bsm::BsmPricer;
use stochastic_rs::quant::pricing::heston::HestonParams;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::quant::pricing::sabr::SabrParams;
use stochastic_rs::quant::pricing::sabr::SabrPricer;
use stochastic_rs::quant::vol_surface::ssvi::SsviParams;
use stochastic_rs::quant::vol_surface::ssvi::SsviSlice;
use stochastic_rs::quant::vol_surface::ssvi::calibrate_ssvi;
use stochastic_rs::quant::vol_surface::svi::SviRawParams;
use stochastic_rs::quant::vol_surface::svi::calibrate_svi;
use stochastic_rs::traits::CalibrationResult;
use stochastic_rs::traits::Calibrator;
use stochastic_rs::traits::ModelPricer;

pub struct Case {
  pub name: &'static str,
  pub run: Box<dyn Fn() -> (Option<bool>, f64)>,
  pub max_rmse: f64,
}

fn calibration<C>(name: &'static str, calibrator: C, max_rmse: f64) -> Case
where
  C: Calibrator<Error = QuantError> + 'static,
{
  Case {
    name,
    run: Box::new(move || {
      let result = calibrator.calibrate(None).unwrap();
      (Some(result.converged()), result.rmse())
    }),
    max_rmse,
  }
}

fn calls(
  strikes: &Array1<f64>,
  maturities: &[f64],
  price: impl Fn(f64, f64) -> f64,
) -> Vec<OptionQuote> {
  maturities
    .iter()
    .flat_map(|&tau| strikes.iter().map(move |&k| (k, tau)))
    .map(|(k, tau)| OptionQuote::new(k, tau, price(k, tau), OptionType::Call))
    .collect()
}

pub fn cases() -> Vec<Case> {
  let strikes = Array1::from_iter((0..9).map(|i| 80.0 + 5.0 * i as f64));
  let mut cases = Vec::new();
  for (name, start) in [("bsm/low_start", 0.1), ("bsm/high_start", 0.5)] {
    let model = BsmPricer::new(0.2, BsmCoc::Bsm1973);
    let quotes = calls(&strikes, &[1.0], |k, tau| {
      model.price_call(100.0, k, 0.03, 0.0, tau)
    });
    cases.push(calibration(
      name,
      BsmCalibrator::new(100.0, 0.03, 0.0, quotes).with_start(BsmParams { sigma: start }),
      1e-8,
    ));
  }
  let heston = HestonPricer::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.6,
  });
  let heston_quotes = calls(&strikes, &[0.25, 1.0, 2.0], |k, tau| {
    heston.price_call(100.0, k, 0.03, 0.0, tau)
  });
  for (name, method) in [
    ("heston/numeric", HestonJacobianMethod::NumericFiniteDiff),
    ("heston/analytic", HestonJacobianMethod::CuiAnalytic),
  ] {
    let cal = HestonCalibrator::new(100.0, 0.03, 0.0, heston_quotes.clone())
      .with_start(HestonParams {
        v0: 0.06,
        kappa: 2.0,
        theta: 0.05,
        sigma: 0.25,
        rho: -0.4,
      })
      .with_jacobian_method(method);
    // The analytic and numerical Heston pricers use different quadratures.
    // The pre-migration analytic fit has a price RMSE of 2.10e-5.
    let max_rmse = if method == HestonJacobianMethod::CuiAnalytic {
      2.5e-5
    } else {
      1e-8
    };
    cases.push(calibration(name, cal, max_rmse));
  }
  let sabr = SabrPricer::new(SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  });
  let sabr_quotes = strikes
    .iter()
    .map(|&k| {
      let price = sabr.call_put(100.0, k, 0.03, 0.0, 1.0).0;
      OptionQuote::new(k, 1.0, price, OptionType::Call)
    })
    .collect();
  cases.push(calibration(
    "sabr",
    SabrCalibrator::new(100.0, 0.03, 0.0, sabr_quotes).with_start(SabrParams {
      alpha: 0.25,
      beta: 1.0,
      rho: 0.0,
      nu: 0.8,
    }),
    1e-5,
  ));
  for (name, model, prices) in [
    (
      "levy/vg",
      LevyModelType::VarianceGamma,
      [
        25.056158, 20.941767, 17.091301, 13.572373, 10.453503, 7.795810, 5.640544, 3.991334,
        2.793823,
      ],
    ),
    (
      "levy/merton",
      LevyModelType::MertonJd,
      [
        24.537096, 20.322713, 16.420092, 12.915553, 9.877019, 7.340385, 5.303578, 3.729825,
        2.557790,
      ],
    ),
  ] {
    cases.push(calibration(
      name,
      LevyCalibrator::new(
        model,
        100.0,
        0.05,
        0.0,
        vec![MarketSlice {
          strikes: strikes.to_vec(),
          prices: prices.to_vec(),
          is_call: vec![true; strikes.len()],
          tau: 1.0,
        }],
      ),
      1e-4,
    ));
  }
  let ks = (0..21).map(|i| -0.5 + 0.05 * i as f64).collect::<Vec<_>>();
  let truth = SviRawParams::new(0.04, 0.2, -0.4, 0.05, 0.15);
  let ws = ks
    .iter()
    .map(|&k| truth.total_variance(k))
    .collect::<Vec<_>>();
  let svi_ks = ks.clone();
  cases.push(Case {
    name: "svi",
    max_rmse: 1e-9,
    run: Box::new(move || {
      let fit = calibrate_svi(&svi_ks, &ws, None);
      let rmse = (svi_ks
        .iter()
        .zip(&ws)
        .map(|(&k, &w)| (fit.total_variance(k) - w).powi(2))
        .sum::<f64>()
        / ws.len() as f64)
        .sqrt();
      // Surface calibration returns parameters without a termination report.
      (None, rmse)
    }),
  });
  let truth = SsviParams::new(-0.4, 0.7, 0.4);
  let slices = [0.02, 0.04, 0.08].map(|theta| SsviSlice {
    log_moneyness: ks.clone(),
    total_variance: ks.iter().map(|&k| truth.total_variance(k, theta)).collect(),
    theta,
  });
  cases.push(Case {
    name: "ssvi",
    max_rmse: 1e-9,
    run: Box::new(move || {
      let fit = calibrate_ssvi(&slices, None);
      let squared = slices
        .iter()
        .flat_map(|s| {
          s.log_moneyness
            .iter()
            .zip(&s.total_variance)
            .map(move |(&k, &w)| (fit.total_variance(k, s.theta) - w).powi(2))
        })
        .sum::<f64>();
      let rmse = (squared / (slices.len() * slices[0].log_moneyness.len()) as f64).sqrt();
      (None, rmse)
    }),
  });
  cases
}
