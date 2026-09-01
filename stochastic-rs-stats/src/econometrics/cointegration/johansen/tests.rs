use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;

fn random_walk(seed: u64, n: usize, sigma: f64) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, sigma, &Deterministic::new(seed));
  let mut steps = vec![0.0_f64; n];
  dist.fill_slice(&mut steps);
  let mut out = Array1::<f64>::zeros(n);
  for i in 1..n {
    out[i] = out[i - 1] + steps[i];
  }
  out
}

fn three_walks() -> Array2<f64> {
  let mut s = Array2::<f64>::zeros((500, 3));
  let r1 = random_walk(31, 500, 1.0);
  let r2 = random_walk(41, 500, 1.0);
  let r3 = random_walk(43, 500, 1.0);
  for i in 0..500 {
    s[[i, 0]] = r1[i];
    s[[i, 1]] = r2[i];
    s[[i, 2]] = r3[i];
  }
  s
}

/// A cointegrated three-variable system (two series sharing one random
/// walk, a third independent) built from an integer LCG so the Rust and
/// Python inputs are bit-identical. Reference values below come from
/// `statsmodels` 0.14: `coint_johansen(y, det_order=0, k_ar_diff=lags-1)`
/// and `VECM(y, k_ar_diff=lags-1, coint_rank=r, deterministic="co").fit()`.
fn lcg_system() -> Array2<f64> {
  let mut x: u64 = 7;
  let mut next = || {
    x = (1_103_515_245 * x + 12_345) % 2_147_483_648;
    x as f64 / 2_147_483_648.0
  };
  let mut y = Array2::<f64>::zeros((150, 3));
  let mut w = 0.0_f64;
  let mut v = 0.0_f64;
  for t in 0..150 {
    let a = next();
    let b = next();
    let c = next();
    let d = next();
    w += a - 0.5;
    v += d - 0.5;
    y[[t, 0]] = w + 0.3 * (b - 0.5);
    y[[t, 1]] = 0.7 * w + 0.2 * (c - 0.5);
    y[[t, 2]] = v;
  }
  y
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
  assert!((got - want).abs() < tol, "{what}: {got} vs {want}");
}

fn assert_matrix_close(got: &Array2<f64>, want: &[[f64; 3]; 3], tol: f64, what: &str) {
  for i in 0..3 {
    for j in 0..3 {
      assert_close(got[[i, j]], want[i][j], tol, &format!("{what}[{i},{j}]"));
    }
  }
}

#[test]
fn johansen_returns_eigenvalues_in_unit_interval() {
  let res = johansen_test(three_walks().view(), 1);
  for &l in res.eigenvalues.iter() {
    assert!((0.0..1.0).contains(&l));
  }
  assert_eq!(res.trace_statistics.len(), 3);
  assert!(res.trace_statistics.iter().all(|v| v.is_finite()));
}

/// $\lambda_{\max}(r) = \lambda_{\mathrm{trace}}(r) - \lambda_{\mathrm{trace}}(r+1)$
/// by construction, with the last two statistics coinciding.
#[test]
fn max_eigenvalue_is_the_trace_increment() {
  let res = johansen_test(three_walks().view(), 2);
  for r in 0..2 {
    let want = res.trace_statistics[r] - res.trace_statistics[r + 1];
    assert_close(res.max_eig_statistics[r], want, 1e-9, "max-eig increment");
  }
  assert_close(
    res.max_eig_statistics[2],
    res.trace_statistics[2],
    1e-12,
    "last statistic",
  );
}

/// Critical values run by $K - r$: for three series the $r = 0$ entry is
/// the three-dimensional quantile and $r = 2$ the $\chi^2_1$ one.
#[test]
fn critical_values_are_indexed_by_k_minus_r() {
  let res = johansen_test(three_walks().view(), 1);
  for (got, want) in res
    .trace_critical_5pct
    .iter()
    .zip([29.7961, 15.4943, 3.8415])
  {
    assert_close(*got, want, 1e-12, "trace critical value");
  }
  for (got, want) in res
    .max_eig_critical_5pct
    .iter()
    .zip([21.1314, 14.2639, 3.8415])
  {
    assert_close(*got, want, 1e-12, "max-eig critical value");
  }
  let wide = critical_values(&TRACE_CRITICAL_5PCT, 13);
  assert!(wide[0].is_nan());
  assert_close(wide[1], 334.9795, 1e-12, "twelve-dimensional entry");
}

#[test]
fn rank_tests_match_statsmodels_on_the_lcg_system() {
  let res = johansen_test(lcg_system().view(), 2);
  assert_eq!(res.nobs, 148);
  let eig = [
    0.388_165_205_199_946_7,
    0.014_177_980_525_231_053,
    0.008_930_823_223_618_703,
  ];
  let trace = [
    76.152_418_230_169_1,
    3.441_057_775_037_630_3,
    1.327_699_425_890_949_1,
  ];
  let max_eig = [
    72.711_360_455_131_47,
    2.113_358_349_146_680_7,
    1.327_699_425_890_949_1,
  ];
  for i in 0..3 {
    assert_close(res.eigenvalues[i], eig[i], 1e-9, "eigenvalue");
    assert_close(res.trace_statistics[i], trace[i], 1e-7, "trace statistic");
    assert_close(
      res.max_eig_statistics[i],
      max_eig[i],
      1e-7,
      "max-eig statistic",
    );
  }
  assert_eq!(res.rank_trace, 1);
  assert_eq!(res.rank_max_eig, 1);
  // First eigenvector in statsmodels' normalisation V' S11 V = I; the
  // sign of an eigenvector is arbitrary, its magnitudes are not.
  let evec = [
    12.899_828_452_440_86,
    -18.377_631_528_142_77,
    -0.037_398_344_894_584_53,
  ];
  let flip = if res.eigenvectors[[0, 0]] < 0.0 {
    -1.0
  } else {
    1.0
  };
  for i in 0..3 {
    assert_close(
      flip * res.eigenvectors[[i, 0]],
      evec[i],
      1e-6,
      "eigenvector",
    );
  }
}

#[test]
fn vecm_rank_one_matches_statsmodels_on_the_lcg_system() {
  let fit = vecm_fit(lcg_system().view(), 2, 1);
  assert_eq!((fit.rank, fit.lags, fit.nobs), (1, 2, 148));
  assert_eq!(fit.beta.dim(), (3, 1));
  assert_eq!(fit.gamma.len(), 1);
  let pi = [
    [
      -0.687_929_153_844_434_2,
      0.980_052_452_126_061_8,
      0.001_994_399_526_580_617,
    ],
    [
      0.363_063_789_056_520_1,
      -0.517_235_757_133_627_7,
      -0.001_052_570_958_748_221_3,
    ],
    [
      -0.214_522_618_517_483_3,
      0.305_617_834_539_553_3,
      0.000_621_930_043_844_099_3,
    ],
  ];
  assert_matrix_close(&fit.pi, &pi, 1e-8, "pi");
  let gamma = [
    [
      0.110_264_323_192_130_55,
      -0.079_215_665_221_366_53,
      0.036_852_715_997_102_09,
    ],
    [
      0.064_452_952_395_844_42,
      -0.030_064_977_402_946_988,
      0.040_787_257_057_419_78,
    ],
    [
      -0.058_011_507_322_851_14,
      0.055_298_103_922_527_195,
      -0.071_416_477_214_476_12,
    ],
  ];
  assert_matrix_close(&fit.gamma[0], &gamma, 1e-8, "gamma");
  let sigma = [
    [
      0.084_436_524_322_726_44,
      0.056_075_391_373_337_05,
      -0.003_644_494_093_003_901_3,
    ],
    [
      0.056_075_391_373_337_05,
      0.044_033_036_580_359_515,
      -0.002_118_462_552_164_226,
    ],
    [
      -0.003_644_494_093_003_901_3,
      -0.002_118_462_552_164_226,
      0.079_263_116_237_804_09,
    ],
  ];
  assert_matrix_close(&fit.sigma, &sigma, 1e-9, "sigma");
  let intercept = [
    0.007_247_397_679_619_97,
    0.007_711_346_337_208_233,
    -0.009_772_037_569_560_767,
  ];
  let beta = [1.0, -1.424_641_544_335_083_3, -0.002_899_135_056_909_688];
  for i in 0..3 {
    assert_close(fit.intercept[i], intercept[i], 1e-9, "intercept");
    assert_close(
      fit.beta[[i, 0]] / fit.beta[[0, 0]],
      beta[i],
      1e-8,
      "beta ratio",
    );
  }
  let first = [
    -0.397_880_610_647_111_4,
    -0.254_309_996_486_158_1,
    0.261_492_120_671_434,
  ];
  let last = [
    0.337_740_039_355_506_8,
    0.159_290_369_876_536_44,
    0.065_467_504_542_652_04,
  ];
  for j in 0..3 {
    assert_close(fit.residuals[[0, j]], first[j], 1e-9, "first residual");
    assert_close(fit.residuals[[147, j]], last[j], 1e-9, "last residual");
  }
}

#[test]
fn vecm_without_lagged_differences_matches_statsmodels_at_rank_two() {
  let fit = vecm_fit(lcg_system().view(), 1, 2);
  assert_eq!((fit.rank, fit.lags, fit.nobs), (2, 1, 149));
  assert!(fit.gamma.is_empty());
  let pi = [
    [
      -0.624_823_388_289_293_8,
      0.857_127_312_737_491_2,
      0.012_449_689_222_049_054,
    ],
    [
      0.386_217_295_452_605_5,
      -0.573_749_946_618_184,
      0.006_939_658_350_708_305,
    ],
    [
      -0.251_908_017_862_471_5,
      0.385_116_353_087_588_5,
      -0.008_153_861_997_812_137,
    ],
  ];
  assert_matrix_close(&fit.pi, &pi, 1e-8, "pi");
  let intercept = [
    0.015_284_225_852_862_796,
    0.014_455_198_943_958_624,
    -0.010_135_064_323_076_49,
  ];
  for i in 0..3 {
    assert_close(fit.intercept[i], intercept[i], 1e-9, "intercept");
  }
  assert_close(
    fit.sigma[[0, 0]],
    0.084_897_262_345_741_67,
    1e-9,
    "sigma[0,0]",
  );
  assert_close(
    fit.sigma[[2, 2]],
    0.080_349_973_193_716_81,
    1e-9,
    "sigma[2,2]",
  );
}

/// At full rank the VECM is the unrestricted regression, so the residuals
/// satisfy the OLS normal equations against every regressor.
#[test]
fn full_rank_residuals_are_orthogonal_to_the_regressors() {
  let y = lcg_system();
  let fit = vecm_fit(y.view(), 2, 3);
  let c = concentrate(y.view(), 2);
  let against_levels = fit.residuals.t().dot(&c.z1);
  let against_lags = fit.residuals.t().dot(&c.z2);
  assert!(against_levels.iter().all(|v| v.abs() < 1e-8));
  assert!(against_lags.iter().all(|v| v.abs() < 1e-8));
}

#[test]
fn rank_zero_drops_the_error_correction_term() {
  let fit = vecm_fit(lcg_system().view(), 2, 0);
  assert_eq!(fit.beta.dim(), (3, 0));
  assert!(fit.pi.iter().all(|v| *v == 0.0));
}

#[test]
#[should_panic(expected = "rank must lie in 0..=3")]
fn rejects_a_rank_above_the_dimension() {
  let _ = vecm_fit(lcg_system().view(), 1, 4);
}
