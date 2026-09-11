use nalgebra::DVector;
use stochastic_rs::prelude::OptionType;
use stochastic_rs::quant::calibration::bsm::BSMCalibrator;
use stochastic_rs::quant::calibration::bsm::BSMParams;
use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::calibration::heston::HestonJacobianMethod;
use stochastic_rs::quant::calibration::heston::HestonParams;
use stochastic_rs::quant::calibration::levy::LevyCalibrator;
use stochastic_rs::quant::calibration::levy::LevyModelType;
use stochastic_rs::quant::calibration::levy::MarketSlice;
use stochastic_rs::quant::calibration::sabr::SabrCalibrator;
use stochastic_rs::quant::calibration::sabr::SabrParams;
use stochastic_rs::quant::pricing::bsm::BSMCoc;
use stochastic_rs::quant::pricing::bsm::BSMPricer;
use stochastic_rs::quant::pricing::heston::HestonPricer;
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
  C: Calibrator<Error = anyhow::Error> + 'static,
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

pub fn cases() -> Vec<Case> {
  let strikes = DVector::from_iterator(9, (0..9).map(|i| 80.0 + 5.0 * i as f64));
  let spot = DVector::from_element(strikes.len(), 100.0);
  let mut cases = Vec::new();
  for (name, start) in [("bsm/low_start", 0.1), ("bsm/high_start", 0.5)] {
    let model = BSMPricer::new(0.2, BSMCoc::Bsm1973);
    let prices = strikes.map(|k| model.price_call(100.0, k, 0.03, 0.0, 1.0));
    cases.push(calibration(
      name,
      BSMCalibrator::new(
        BSMParams { v: start },
        prices,
        spot.clone(),
        strikes.clone(),
        0.03,
        None,
        None,
        None,
        1.0,
        OptionType::Call,
      ),
      1e-8,
    ));
  }
  let heston = HestonPricer::new(0.04, -0.6, 1.5, 0.04, 0.3, Some(0.0));
  let slices = [0.25, 1.0, 2.0].map(|tau| MarketSlice {
    strikes: strikes.as_slice().to_vec(),
    prices: strikes
      .iter()
      .map(|&k| heston.price_call(100.0, k, 0.03, 0.0, tau))
      .collect(),
    is_call: vec![true; strikes.len()],
    tau,
  });
  for (name, method) in [
    ("heston/numeric", HestonJacobianMethod::NumericFiniteDiff),
    ("heston/analytic", HestonJacobianMethod::CuiAnalytic),
  ] {
    let mut cal = HestonCalibrator::from_slices(
      Some(HestonParams {
        v0: 0.06,
        kappa: 2.0,
        theta: 0.05,
        sigma: 0.25,
        rho: -0.4,
      }),
      &slices,
      100.0,
      0.03,
      Some(0.0),
      OptionType::Call,
      false,
    );
    cal.set_jacobian_method(method);
    // The analytic and numerical Heston pricers use different quadratures.
    // The pre-migration analytic fit has a price RMSE of 2.10e-5.
    let max_rmse = if method == HestonJacobianMethod::CuiAnalytic {
      2.5e-5
    } else {
      1e-8
    };
    cases.push(calibration(name, cal, max_rmse));
  }
  let sabr = SabrPricer::new(0.2, 1.0, 0.6, -0.3);
  cases.push(calibration(
    "sabr",
    SabrCalibrator::new(
      Some(SabrParams {
        alpha: 0.25,
        beta: 1.0,
        nu: 0.8,
        rho: 0.0,
      }),
      strikes.map(|k| sabr.call_put(100.0, k, 0.03, 0.0, 1.0).0),
      spot,
      strikes.clone(),
      0.03,
      None,
      1.0,
      OptionType::Call,
      false,
    ),
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
      LevyModelType::MertonJD,
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
          strikes: strikes.as_slice().to_vec(),
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
