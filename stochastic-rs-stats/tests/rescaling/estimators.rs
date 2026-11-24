use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::econometrics::cointegration::fit_vecm;
use stochastic_rs_stats::econometrics::cointegration::johansen_test;
use stochastic_rs_stats::econometrics::cusum;
use stochastic_rs_stats::econometrics::pelt;
use stochastic_rs_stats::fractal_dim::FdDiagnostic;
use stochastic_rs_stats::fractal_dim::FractalDimEstimator;
use stochastic_rs_stats::fractal_dim::Higuchi;
use stochastic_rs_stats::fractal_dim::Variogram;
use stochastic_rs_stats::hurst::Gph;
use stochastic_rs_stats::hurst::HurstDiagnostic;
use stochastic_rs_stats::hurst::RescaledRange;
use stochastic_rs_stats::hurst::Variations;
use stochastic_rs_stats::hurst::Wavelet;
use stochastic_rs_stats::realized::HarRv;
use stochastic_rs_stats::spectral::PeriodogramConfig;
use stochastic_rs_stats::spectral::SpectrumSearchConfig;
use stochastic_rs_stats::spectral::fft_spectrum_search;
use stochastic_rs_stats::spectral::periodogram_fft;
use stochastic_rs_stats::stationarity::andrews_ploberger::AndrewsPlobergerConfig;
use stochastic_rs_stats::stationarity::andrews_ploberger::andrews_ploberger_test;
use stochastic_rs_stats::stationarity::cusum::CusumConfig;
use stochastic_rs_stats::stationarity::cusum::CusumVariant;
use stochastic_rs_stats::stationarity::cusum::cusum_test;
use stochastic_rs_stats::stationarity::reset::ResetConfig;
use stochastic_rs_stats::stationarity::reset::reset_test;
use stochastic_rs_stats::traits::HurstEstimator;

use crate::assert_no_bit_moves;
use crate::normals;
use crate::twins;

const DATA_SCALE: StatsError = StatsError::Numerical {
  stage: "data scale",
};

/// The exponent and the fit's `r²` move no bit; the intercept and the logged statistics move by
/// `shift`, to the rounding of one logarithm and one difference each.
fn assert_log_log_moves_by(here: HurstDiagnostic<f64>, there: HurstDiagnostic<f64>, shift: f64) {
  let parts = |d: HurstDiagnostic<f64>| match d {
    HurstDiagnostic::LogLogRegression {
      slope,
      intercept,
      r_squared,
      log_scales,
      log_stats,
    } => ((slope, r_squared, log_scales), intercept, log_stats),
    other => panic!("{other:?}"),
  };
  let ((free, intercept, logs), (free_there, intercept_there, logs_there)) =
    (parts(here), parts(there));
  assert_eq!(format!("{free:?}"), format!("{free_there:?}"));
  let moved = std::iter::once((intercept, intercept_there)).chain(logs.into_iter().zip(logs_there));
  for (a, b) in moved {
    let bound = 4.0 * f64::EPSILON * a.abs().max(b.abs()).max(shift.abs());
    assert!((b - a - shift).abs() <= bound, "{a} + {shift} against {b}");
  }
}

#[test]
fn the_log_log_hurst_estimators_answer_one_way_at_every_power_of_two() {
  let noise = normals(18, 2048, 1.0);
  for (at, twin, power) in twins(&noise, 1000) {
    let pair = |e: &dyn HurstEstimator<f64>| {
      (
        e.estimate(at.view()).unwrap(),
        e.estimate(twin.view()).unwrap(),
      )
    };
    let cases: [(&dyn HurstEstimator<f64>, f64); 3] = [
      (&Gph::default(), 2.0 * power.ln()),
      (&RescaledRange::default(), 0.0),
      (&Wavelet::default(), 2.0 * power.log2()),
    ];
    for (estimator, shift) in cases {
      let (here, there) = pair(estimator);
      assert_eq!(
        here.hurst.to_bits(),
        there.hurst.to_bits(),
        "at {:e}",
        at[0]
      );
      assert_log_log_moves_by(here.diagnostic, there.diagnostic, shift);
    }
    let fit =
      |x: &Array1<f64>| FractalDimEstimator::estimate(&Higuchi::default(), x.view()).unwrap();
    let (here, there) = (fit(&at), fit(&twin));
    assert_eq!(
      here.d.to_bits(),
      there.d.to_bits(),
      "Higuchi at {:e}",
      at[0]
    );
    let (
      FdDiagnostic::LogLogRegression { intercept: a, .. },
      FdDiagnostic::LogLogRegression { intercept: b, .. },
    ) = (here.diagnostic, there.diagnostic)
    else {
      panic!("Higuchi's diagnostic is its regression");
    };
    let shift = power.ln();
    assert!((b - a - shift).abs() <= 4.0 * f64::EPSILON * a.abs().max(b.abs()).max(shift.abs()));
  }
}

/// A variation is the path's square, so the pairs stop where it leaves the doubles; past that the
/// estimator names the scale, not a constant path.
#[test]
fn the_variation_estimators_answer_one_way_at_every_power_of_two_their_sums_fit_in() {
  let mut path = normals(19, 1024, 1.0);
  for t in 1..path.len() {
    path[t] += path[t - 1];
  }
  for (at, twin, power) in twins(&path, 200) {
    let fit = |x: &Array1<f64>| Variations::default().estimate(x.view()).unwrap();
    let (here, there) = (fit(&at), fit(&twin));
    assert_eq!(
      here.hurst.to_bits(),
      there.hurst.to_bits(),
      "at {:e}",
      at[0]
    );
    let sums = |d: HurstDiagnostic<f64>| match d {
      HurstDiagnostic::Variations { v_short, v_long } => (v_short, v_long),
      other => panic!("{other:?}"),
    };
    let ((short, long), there) = (sums(here.diagnostic), sums(there.diagnostic));
    assert_eq!((short * power * power, long * power * power), there);
    let fit =
      |x: &Array1<f64>| FractalDimEstimator::estimate(&Variogram::default(), x.view()).unwrap();
    let (here, there) = (fit(&at), fit(&twin));
    assert_eq!(
      here.d.to_bits(),
      there.d.to_bits(),
      "variogram at {:e}",
      at[0]
    );
  }
  for size in [1e-200, 1e300] {
    let far = Array1::from_iter(path.iter().map(|v| v * size));
    let variations = Variations::default().estimate(far.view()).map(|f| f.hurst);
    assert_eq!(variations.unwrap_err(), DATA_SCALE, "{size:e}");
    let variogram = FractalDimEstimator::estimate(&Variogram::default(), far.view()).map(|f| f.d);
    assert_eq!(variogram.unwrap_err(), DATA_SCALE, "{size:e}");
  }
}

/// `y = 1 + 0.8 x + ε` with a level shift at row 250, and its design.
fn regression() -> (Vec<f64>, Vec<f64>) {
  let (x, e) = (normals(21, 400, 1.0), normals(22, 400, 0.5));
  let shift = |i: usize| if i > 250 { 1.5 } else { 0.0 };
  let y = (0..400).map(|i| 1.0 + 0.8 * x[i] + shift(i) + e[i]);
  (y.collect(), x)
}

fn design(x: &[f64], power: f64) -> Array2<f64> {
  Array2::from_shape_fn((x.len(), 2), |(i, j)| [1.0, power * x[i]][j])
}

#[test]
fn the_regression_tests_answer_one_way_at_every_power_of_two_of_y_and_of_a_regressor() {
  let (y, x) = regression();
  for (at, twin, power) in twins(&y, 1000) {
    for (here_x, there_x) in [
      (design(&x, 1.0), design(&x, 1.0)),
      (design(&x, 1.0), design(&x, power)),
    ] {
      for variant in [CusumVariant::Cusum, CusumVariant::Cusumq] {
        let cfg = CusumConfig {
          variant,
          ..CusumConfig::default()
        };
        let (here, there) = (
          cusum_test(at.view(), here_x.view(), cfg).unwrap(),
          cusum_test(twin.view(), there_x.view(), cfg).unwrap(),
        );
        let free = |t: &stochastic_rs_stats::stationarity::cusum::CusumTest| {
          format!(
            "{:?}",
            (t.statistic, t.rejects_null, t.breakpoint, &t.cusum_path)
          )
        };
        assert_eq!(free(&here), free(&there), "{variant:?} at {:e}", at[0]);
        let lifted = here.recursive_residuals.iter().map(|w| w * power);
        assert_eq!(lifted.collect::<Vec<_>>(), there.recursive_residuals);
      }
      let breaks = |y: &Array1<f64>, x: &Array2<f64>| {
        let cfg = AndrewsPlobergerConfig::default();
        format!("{:?}", andrews_ploberger_test(y.view(), x.view(), cfg))
      };
      assert_eq!(
        breaks(&at, &here_x),
        breaks(&twin, &there_x),
        "at {:e}",
        at[0]
      );
      let reset = |y: &Array1<f64>, x: &Array2<f64>| {
        format!(
          "{:?}",
          reset_test(y.view(), x.view(), ResetConfig::default())
        )
      };
      assert_eq!(
        reset(&at, &here_x),
        reset(&twin, &there_x),
        "at {:e}",
        at[0]
      );
    }
  }
}

#[test]
fn the_cusum_chart_alarms_at_the_same_points_at_every_power_of_two() {
  let mut series = normals(25, 400, 0.5);
  series[200..].iter_mut().for_each(|v| *v += 2.0);
  assert_no_bit_moves(&series, "CUSUM chart", |x| {
    format!("{:?}", cusum(x.view(), 0.5, 5.0))
  });
}

#[test]
fn har_answers_one_way_at_every_power_of_two_its_sum_of_squares_fits_in() {
  let shocks = normals(24, 300, 0.3);
  let mut rv = vec![1.0_f64; 300];
  for t in 1..300 {
    rv[t] = (0.7 * rv[t - 1].ln() + shocks[t]).exp();
  }
  for (at, twin, power) in twins(&rv, 200) {
    let (here, there) = (
      HarRv::fit(at.view()).unwrap(),
      HarRv::fit(twin.view()).unwrap(),
    );
    let free = |h: &HarRv| (h.fit.beta_d, h.fit.beta_w, h.fit.beta_m, h.fit.r_squared);
    assert_eq!(
      format!("{:?}", free(&here)),
      format!("{:?}", free(&there)),
      "at {:e}",
      at[0]
    );
    assert_eq!(here.fit.intercept * power, there.fit.intercept);
    assert_eq!(here.fit.rss * power * power, there.fit.rss);
  }
  for size in [1e-200, 1e300] {
    let far = Array1::from_iter(rv.iter().map(|v| v * size));
    assert_eq!(
      HarRv::fit(far.view()).map(|h| h.fit.nobs).unwrap_err(),
      DATA_SCALE
    );
  }
}

#[test]
fn the_periodogram_follows_a_power_of_two_exactly_where_its_power_fits_in() {
  let noise = normals(26, 512, 0.2);
  let wave = (0..512).map(|i| (0.3 * i as f64).sin() + 0.5 * (1.1 * i as f64).sin() + noise[i]);
  let wave = wave.collect::<Vec<_>>();
  for (at, twin, power) in twins(&wave, 200) {
    let spectrum =
      |x: &Array1<f64>| periodogram_fft(x.view(), PeriodogramConfig::default()).unwrap();
    let lifted = spectrum(&at)
      .spectrum
      .iter()
      .map(|p| p * power * power)
      .collect::<Vec<_>>();
    assert_eq!(lifted, spectrum(&twin).spectrum, "at {:e}", at[0]);
    let peaks = |x: &Array1<f64>| {
      let found = fft_spectrum_search(x.view(), SpectrumSearchConfig::default()).unwrap();
      found.iter().map(|p| p.index).collect::<Vec<_>>()
    };
    assert_eq!(peaks(&at), peaks(&twin), "at {:e}", at[0]);
  }
  for size in [1e-200, 1e300] {
    let far = Array1::from_iter(wave.iter().map(|v| v * size));
    let answer = periodogram_fft(far.view(), PeriodogramConfig::default()).map(|p| p.nfft);
    assert_eq!(answer.unwrap_err(), DATA_SCALE, "{size:e}");
  }
}

/// Each series at its own power of two: the eigenvalues and statistics move no bit, a cointegrating
/// vector's row by its series' power inverted, and the loadings, `Π` and `Ω` as their units say.
#[test]
fn johansen_and_the_vecm_answer_one_way_at_every_power_of_two_of_each_series() {
  let (steps, e) = (normals(23, 400, 1.0), normals(22, 400, 0.5));
  let mut walk = vec![0.0; 400];
  for t in 1..400 {
    walk[t] = walk[t - 1] + steps[t];
  }
  let pair = |a: f64, b: f64| {
    Array2::from_shape_fn((400, 2), |(i, j)| {
      [a * (1.0 + 1.2 * walk[i] + 0.3 * e[i]), b * walk[i]][j]
    })
  };
  for (size, k) in [(1.0, 3_i32), (1.0, -200), (1e-200, 664), (1e300, -996)] {
    let power = 2f64.powi(k);
    let (at, twin) = (pair(size, 1.0), pair(size * power, 1.0));
    let (here, there) = (
      johansen_test(at.view(), 2).unwrap(),
      johansen_test(twin.view(), 2).unwrap(),
    );
    let free = |t: &stochastic_rs_stats::econometrics::cointegration::JohansenTest| {
      format!(
        "{:?}",
        (
          &t.eigenvalues,
          &t.trace_statistics,
          &t.max_eig_statistics,
          t.rank_trace
        )
      )
    };
    assert_eq!(free(&here), free(&there), "{size:e} × 2^{k}");
    let mut lifted = here.eigenvectors.clone();
    lifted.row_mut(0).mapv_inplace(|v| v / power);
    assert_eq!(lifted, there.eigenvectors, "{size:e} × 2^{k}");
  }
  let power = 2f64.powi(-40);
  let (here, there) = (
    fit_vecm(pair(1.0, 1.0).view(), 2, 1).unwrap(),
    fit_vecm(pair(power, 1.0).view(), 2, 1).unwrap(),
  );
  let units = [power, 1.0];
  let lifted = |m: &Array2<f64>, row: i32, column: i32| {
    Array2::from_shape_fn(m.dim(), |(i, j)| {
      m[[i, j]] * units[i].powi(row) * units[j].powi(column)
    })
  };
  assert_eq!(lifted(&here.beta, -1, 0), there.beta);
  assert_eq!(lifted(&here.alpha, 1, 0), there.alpha);
  assert_eq!(lifted(&here.pi, 1, -1), there.pi);
  assert_eq!(lifted(&here.gamma[0], 1, -1), there.gamma[0]);
  assert_eq!(lifted(&here.sigma, 1, 1), there.sigma);
  assert_eq!(
    &here.intercept * &Array1::from_vec(units.to_vec()),
    there.intercept
  );
  let scaled = Array2::from_shape_fn(here.residuals.dim(), |(t, j)| {
    here.residuals[[t, j]] * units[j]
  });
  assert_eq!(scaled, there.residuals);
  assert_eq!(
    fit_vecm(pair(1e300, 1e300).view(), 2, 1)
      .map(|f| f.nobs)
      .unwrap_err(),
    DATA_SCALE
  );
}

/// `c + k ulp(c)` is exact, so a series of neighbouring doubles is its integers moved and scaled:
/// the break tests, the chart and HAR's fit, free of both, answer for the integers to `n² ε`.
#[test]
fn the_break_tests_and_har_read_neighbouring_doubles_as_their_integers() {
  let n = 512;
  let bound = (n * n) as f64 * f64::EPSILON * 20.0;
  let mut state = 0.0;
  let steps = normals(28, n, 1.0);
  let steps = steps.iter().map(|z| {
    state = 0.4 * state + z;
    (20.0 + 6.0 * state).round().clamp(0.0, 40.0)
  });
  let steps = steps.collect::<Vec<_>>();
  let regressor = normals(29, n, 1.0);
  let answers = |y: &Array1<f64>| {
    let x = design(&regressor, 1.0);
    let chart = cusum(y.view(), 0.5, 5.0).unwrap();
    let har = HarRv::fit(y.view()).unwrap().fit;
    [
      chart.alarms.len() as f64,
      cusum_test(y.view(), x.view(), CusumConfig::default())
        .unwrap()
        .statistic,
      andrews_ploberger_test(y.view(), x.view(), AndrewsPlobergerConfig::default())
        .unwrap()
        .statistic,
      har.r_squared,
      har.beta_d,
    ]
  };
  let theirs = answers(&Array1::from_vec(steps.clone()));
  for level in [1.0_f64, 3.0, 1e8, -7e5] {
    let ulp = level.abs().next_up() - level.abs();
    let ours = answers(&Array1::from_iter(steps.iter().map(|k| level + k * ulp)));
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
      assert!(
        (a - b).abs() <= bound * (1.0 + b.abs()),
        "level {level}, answer {i}: {a} against the integers' {b}"
      );
    }
  }
}

/// A level is no part of a mean-shift cost: one shift is one changepoint at `1e8` and `1e10` too.
#[test]
fn pelt_finds_the_same_changepoints_at_every_level() {
  let mut series = normals(30, 400, 0.5);
  series[200..].iter_mut().for_each(|v| *v += 2.0);
  let at = |level: f64| {
    let lifted = Array1::from_iter(series.iter().map(|v| v + level));
    pelt(lifted.view(), 10.0, 5).unwrap()
  };
  let plain = at(0.0);
  assert_eq!(plain.changepoints, vec![200]);
  for level in [1e6, 1e8, 1e10] {
    let lifted = at(level);
    assert_eq!(lifted.changepoints, plain.changepoints, "level {level:e}");
    // The lifted values round at `ulp(level)`, each of 400 by half a step of its deviation.
    let rounding = 400.0 * 4.0 * level * f64::EPSILON;
    assert!(
      (lifted.cost - plain.cost).abs() <= rounding,
      "level {level:e}: {}",
      lifted.cost
    );
  }
}
