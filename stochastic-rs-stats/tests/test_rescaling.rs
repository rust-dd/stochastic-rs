//! The scale-free tests run at unit scale: a power of two moves no bit of an answer, whatever the
//! size of the data, and an answer in the data's units moves by that power of two alone.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::acf::LjungBoxConfig;
use stochastic_rs_stats::acf::acf;
use stochastic_rs_stats::acf::ljung_box;
use stochastic_rs_stats::hurst::Dfa;
use stochastic_rs_stats::hurst::HurstDiagnostic;
use stochastic_rs_stats::kde::silverman_bandwidth;
use stochastic_rs_stats::normality::anderson_darling::AndersonDarlingConfig;
use stochastic_rs_stats::normality::anderson_darling::anderson_darling_normal_test;
use stochastic_rs_stats::normality::jarque_bera::JarqueBeraConfig;
use stochastic_rs_stats::normality::jarque_bera::jarque_bera_test;
use stochastic_rs_stats::normality::shapiro_francia::ShapiroFranciaConfig;
use stochastic_rs_stats::normality::shapiro_francia::shapiro_francia_test;
use stochastic_rs_stats::realized::bns_jump_test;
use stochastic_rs_stats::realized::lee_mykland_test;
use stochastic_rs_stats::stationarity::lo_mackinlay::LoMacKinlayConfig;
use stochastic_rs_stats::stationarity::lo_mackinlay::lo_mackinlay_test;
use stochastic_rs_stats::traits::HurstEstimator;

fn normals(seed: u64, n: usize, sd: f64) -> Vec<f64> {
  let mut z = vec![0.0; n];
  SimdNormal::<f64>::new(0.0, sd)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  z
}

/// A sample at a decimal size and its twin `2^k` times it, an exact product: at the ends of the
/// doubles the twin is of ordinary size, so the pair pins the end's answer to the ordinary one.
fn twins(sample: &[f64], reach: i32) -> Vec<(Array1<f64>, Array1<f64>, f64)> {
  let ends = [(1e-200, 664_i32), (1e-170, 565), (1e300, -996)];
  let near = [(1.0, -1_i32), (1.0, 1), (1.0, -400), (1.0, 400), (3.0, 7)];
  let cases = near.into_iter().chain(ends);
  cases
    .filter(|(_, k)| k.abs() <= reach)
    .map(|(size, k)| {
      let power = 2f64.powi(k);
      let at = Array1::from_iter(sample.iter().map(|v| v * size));
      let twin = at.mapv(|v| v * power);
      (at, twin, power)
    })
    .collect()
}

fn assert_no_bit_moves(sample: &[f64], what: &str, answer: impl Fn(&Array1<f64>) -> String) {
  for (at, twin, power) in twins(sample, 1000) {
    assert_eq!(
      answer(&at),
      answer(&twin),
      "{what} at {:e}, × {power:e}",
      at[0]
    );
  }
}

#[test]
fn the_normality_tests_answer_one_way_at_every_power_of_two() {
  let sample = normals(11, 400, 1.0);
  assert_no_bit_moves(&sample, "Anderson–Darling", |x| {
    format!(
      "{:?}",
      anderson_darling_normal_test(x.view(), AndersonDarlingConfig::default())
    )
  });
  assert_no_bit_moves(&sample, "Jarque–Bera", |x| {
    format!(
      "{:?}",
      jarque_bera_test(x.view(), JarqueBeraConfig::default())
    )
  });
  assert_no_bit_moves(&sample, "Shapiro–Francia", |x| {
    let cfg = ShapiroFranciaConfig {
      bootstrap_samples: 99,
      ..ShapiroFranciaConfig::default()
    };
    format!("{:?}", shapiro_francia_test(x.view(), cfg))
  });
}

#[test]
fn the_autocorrelations_answer_one_way_at_every_power_of_two() {
  let mut sample = normals(12, 600, 1.0);
  for t in 1..sample.len() {
    sample[t] += 0.4 * sample[t - 1];
  }
  assert_no_bit_moves(&sample, "acf", |x| format!("{:?}", acf(x.view(), 12)));
  assert_no_bit_moves(&sample, "Ljung–Box", |x| {
    format!("{:?}", ljung_box(x.view(), LjungBoxConfig::default()))
  });
  let single = Array1::from_iter(sample.iter().map(|v| *v as f32));
  for (size, k) in [(1.0_f32, 40), (1.0, -40), (1e-30, 100), (1e30, -100)] {
    let at = single.mapv(|v| v * size);
    let twin = at.mapv(|v| v * 2f32.powi(k));
    let answer = |x: &Array1<f32>| format!("{:?}", acf(x.view(), 12));
    assert_eq!(answer(&at), answer(&twin), "f32 acf at {size:e}, × 2^{k}");
  }
}

#[test]
fn lo_mackinlay_answers_one_way_at_every_power_of_two() {
  let returns = normals(13, 800, 0.01);
  for input_is_log_prices in [false, true] {
    assert_no_bit_moves(&returns, "Lo–MacKinlay", |x| {
      let cfg = LoMacKinlayConfig {
        input_is_log_prices,
        ..LoMacKinlayConfig::default()
      };
      format!("{:?}", lo_mackinlay_test(x.view(), cfg))
    });
  }
}

#[test]
fn lee_mykland_flags_the_same_jumps_at_every_power_of_two() {
  let mut returns = normals(14, 600, 0.01);
  returns[400] = 0.3;
  for (at, twin, power) in twins(&returns, 1000) {
    let (here, there) = (
      lee_mykland_test(at.view(), 30, 0.05).unwrap(),
      lee_mykland_test(twin.view(), 30, 0.05).unwrap(),
    );
    assert_eq!(here.jump_indices, vec![400], "at {:e}", at[0]);
    assert_eq!(here.jump_indices, there.jump_indices);
    let bits = |v: &Array1<f64>| v.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
    assert_eq!(
      bits(&here.statistics),
      bits(&there.statistics),
      "at {:e}",
      at[0]
    );
    let lifted = here.local_volatility.mapv(|v| v * power);
    assert_eq!(
      bits(&lifted),
      bits(&there.local_volatility),
      "at {:e}",
      at[0]
    );
  }
}

/// The tripower quarticity is the returns' fourth power, so the pairs stop where it leaves the
/// doubles; past that the test names the scale, not a degenerate series or a NaN.
#[test]
fn bns_answers_one_way_at_every_power_of_two_its_measures_fit_in() {
  let returns = normals(15, 600, 0.01);
  for (at, twin, power) in twins(&returns, 200) {
    let (here, there) = (
      bns_jump_test(at.view(), 0.05).unwrap(),
      bns_jump_test(twin.view(), 0.05).unwrap(),
    );
    let free = |t: &stochastic_rs_stats::realized::BnsJumpTest| {
      (
        t.statistic.to_bits(),
        t.p_value.map(f64::to_bits),
        t.rejects_null,
      )
    };
    assert_eq!(free(&here), free(&there), "at {:e}", at[0]);
    assert_eq!(here.rv * power * power, there.rv);
    assert_eq!(here.bv * power * power, there.bv);
    assert_eq!(here.tpq * power.powi(4), there.tpq);
  }
  for size in [1e-200, 1e300] {
    let far = Array1::from_iter(returns.iter().map(|v| v * size));
    assert_eq!(
      bns_jump_test(far.view(), 0.05).unwrap_err(),
      StatsError::Numerical {
        stage: "data scale"
      },
      "{size:e}"
    );
  }
}

#[test]
fn silverman_s_bandwidth_follows_a_power_of_two_exactly() {
  let sample = normals(16, 500, 1.0);
  for (at, twin, power) in twins(&sample, 1000) {
    let (here, there) = (
      silverman_bandwidth(at.view()).unwrap(),
      silverman_bandwidth(twin.view()).unwrap(),
    );
    assert_eq!(here * power, there, "at {:e}", at[0]);
  }
}

/// The exponent and its fit move no bit; the logged fluctuations move by `k ln 2`, to the rounding
/// of one logarithm and one difference each.
#[test]
fn dfa_answers_one_way_at_every_power_of_two() {
  let noise = normals(17, 2000, 1.0);
  for (at, twin, power) in twins(&noise, 1000) {
    let fit = |x: &Array1<f64>| Dfa::default().estimate(x.view()).unwrap();
    let (here, there) = (fit(&at), fit(&twin));
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
    assert_eq!(
      here.hurst.to_bits(),
      there.hurst.to_bits(),
      "at {:e}",
      at[0]
    );
    let ((free, intercept, logs), (free_there, intercept_there, logs_there)) =
      (parts(here.diagnostic), parts(there.diagnostic));
    assert_eq!(
      format!("{free:?}"),
      format!("{free_there:?}"),
      "at {:e}",
      at[0]
    );
    let shift = power.ln();
    let moved =
      std::iter::once((intercept, intercept_there)).chain(logs.into_iter().zip(logs_there));
    for (a, b) in moved {
      let bound = 4.0 * f64::EPSILON * a.abs().max(b.abs()).max(shift.abs());
      assert!((b - a - shift).abs() <= bound, "{a} + {shift} against {b}");
    }
  }
}
