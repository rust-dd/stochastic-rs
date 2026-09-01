use ndarray::Array1;

use super::*;

/// GARCH(1,1) path ($\omega = 0.05$, $\alpha = 0.10$, $\beta = 0.85$,
/// $\mu = 0.02$) driven by sum-of-twelve-uniform innovations from an integer
/// LCG, bit-identical to the Python generator behind the `arch` 7.x
/// reference numbers below (`arch_model(..., rescale=False).fit(backcast=
/// mean((r - mean(r))**2), tol=1e-12)`).
fn lcg_garch_returns() -> Array1<f64> {
  let mut x: u64 = 11;
  let mut next = || {
    x = (1_103_515_245 * x + 12_345) % 2_147_483_648;
    x as f64 / 2_147_483_648.0
  };
  let (omega, alpha, beta, mu) = (0.05_f64, 0.10_f64, 0.85_f64, 0.02_f64);
  let mut sigma2 = omega / (1.0 - alpha - beta);
  let mut r = Array1::<f64>::zeros(1_000);
  for t in 0..1_000 {
    let mut z = 0.0;
    for _ in 0..12 {
      z += next();
    }
    z -= 6.0;
    let eps = sigma2.sqrt() * z;
    r[t] = mu + eps;
    sigma2 = omega + alpha * eps * eps + beta * sigma2;
  }
  r
}

const BACKCAST: f64 = 0.811_752_254_967_882_4;

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
  assert!((got - want).abs() < tol, "{what}: {got} vs {want}");
}

fn assert_relative(got: f64, want: f64, tol: f64, what: &str) {
  assert!(
    ((got - want) / want).abs() < tol,
    "{what}: {got} vs {want} (relative {})",
    (got - want) / want
  );
}

/// The likelihood evaluated at `arch`'s optimum must reproduce its
/// log-likelihood and conditional variances, isolating the recursion from
/// the optimiser.
fn check_likelihood_at(spec: GarchSpec, natural: &[f64], loglik: f64, sigma2: [f64; 3]) {
  let r = lcg_garch_returns();
  let returns: Vec<f64> = r.to_vec();
  let (ll, _) = recursion::log_likelihood_terms(&spec, natural, &returns, BACKCAST);
  assert_close(ll, loglik, 1e-8, "log-likelihood at arch's optimum");
  let mut resid = vec![0.0; 1_000];
  let mut var = vec![0.0; 1_000];
  recursion::variance_path(&spec, natural, &returns, BACKCAST, &mut resid, &mut var);
  assert_close(var[0], sigma2[0], 1e-12, "sigma2[0]");
  assert_close(var[1], sigma2[1], 1e-12, "sigma2[1]");
  assert_close(var[999], sigma2[2], 1e-12, "sigma2[999]");
}

/// The fit must land on `arch`'s optimum: never a worse likelihood,
/// parameters within `5e-4`, both standard-error flavours within 2%.
fn check_fit(
  spec: GarchSpec,
  params: &[f64],
  loglik: f64,
  std_err: &[f64],
  robust: &[f64],
) -> GarchFit {
  let r = lcg_garch_returns();
  let fit = garch_fit(r.view(), spec);
  assert_close(fit.backcast, BACKCAST, 1e-12, "backcast");
  assert!(fit.converged, "simplex did not converge");
  assert!(
    fit.log_likelihood >= loglik - 1e-7,
    "worse likelihood than arch: {} vs {loglik}",
    fit.log_likelihood
  );
  assert_close(fit.log_likelihood, loglik, 1e-4, "log-likelihood");
  assert_eq!(fit.params.len(), params.len());
  for (i, name) in spec.param_names().iter().enumerate() {
    assert_close(fit.params[i], params[i], 5e-4, name);
    assert_relative(fit.std_errors[i], std_err[i], 0.02, &format!("se({name})"));
    assert_relative(
      fit.robust_std_errors[i],
      robust[i],
      0.02,
      &format!("robust se({name})"),
    );
  }
  let k = params.len() as f64;
  assert_close(fit.aic, 2.0 * k - 2.0 * fit.log_likelihood, 1e-9, "aic");
  assert_close(
    fit.bic,
    k * 1_000_f64.ln() - 2.0 * fit.log_likelihood,
    1e-9,
    "bic",
  );
  assert_eq!(fit.nobs, 1_000);
  assert_eq!(fit.conditional_variance.len(), 1_000);
  for t in 0..1_000 {
    let z = fit.residuals[t] / fit.conditional_variance[t].sqrt();
    assert_close(
      fit.standardized_residuals[t],
      z,
      1e-12,
      "standardised residual",
    );
  }
  fit
}

#[test]
fn dataset_is_bit_identical_to_the_python_generator() {
  let r = lcg_garch_returns();
  assert_eq!(r[0], -0.335_564_168_654_382_04);
  assert_eq!(r[1], -0.442_694_160_818_927_5);
  assert_eq!(r[2], -0.239_489_913_987_010_44);
  assert_eq!(r[999], 0.030_532_903_954_076_145);
  assert!((r.sum() + 35.712_031_983_457_2).abs() < 1e-9);
}

#[test]
fn garch11_likelihood_matches_arch_at_its_optimum() {
  check_likelihood_at(
    GarchSpec::garch(1, 1),
    &[
      -0.025_956_123_332_940_735,
      0.056_589_700_129_218_36,
      0.085_808_751_984_951_25,
      0.844_532_046_749_548_8,
    ],
    -1_290.659_101_911_746_8,
    [
      0.811_795_941_390_569_5,
      0.750_402_769_755_277,
      0.559_695_238_399_947_4,
    ],
  );
}

#[test]
fn garch11_fit_matches_arch() {
  let fit = check_fit(
    GarchSpec::garch(1, 1),
    &[
      -0.025_956_123_332_940_735,
      0.056_589_700_129_218_36,
      0.085_808_751_984_951_25,
      0.844_532_046_749_548_8,
    ],
    -1_290.659_101_911_746_8,
    &[
      0.026_684_756_678_186_01,
      0.025_366_879_617_840_038,
      0.025_597_245_226_740_614,
      0.048_923_891_677_150_49,
    ],
    &[
      0.026_845_687_183_653_787,
      0.026_620_225_820_051_042,
      0.027_250_152_765_373_64,
      0.052_377_362_920_558_466,
    ],
  );
  assert_close(
    fit.persistence,
    fit.alpha[0] + fit.beta[0],
    1e-15,
    "persistence",
  );
  assert!(fit.gamma.is_empty());
  assert_eq!(
    fit.spec.param_names(),
    ["mu", "omega", "alpha[1]", "beta[1]"]
  );
}

#[test]
fn gjr11_likelihood_matches_arch_at_its_optimum() {
  check_likelihood_at(
    GarchSpec::gjr(1, 1),
    &[
      -0.028_082_355_002_584_367,
      0.055_765_688_908_631_04,
      0.078_765_800_099_416_1,
      0.012_818_913_432_787_772,
      0.846_256_091_558_872_9,
    ],
    -1_290.600_488_947_766_7,
    [
      0.811_857_186_299_555_2,
      0.751_463_661_050_930_4,
      0.565_588_314_915_980_9,
    ],
  );
}

#[test]
fn gjr11_fit_matches_arch() {
  let fit = check_fit(
    GarchSpec::gjr(1, 1),
    &[
      -0.028_082_355_002_584_367,
      0.055_765_688_908_631_04,
      0.078_765_800_099_416_1,
      0.012_818_913_432_787_772,
      0.846_256_091_558_872_9,
    ],
    -1_290.600_488_947_766_7,
    &[
      0.027_383_930_424_649_08,
      0.025_352_486_354_492_22,
      0.031_904_809_181_526_626,
      0.037_368_183_692_527_16,
      0.049_219_996_586_011_78,
    ],
    &[
      0.027_314_718_629_390_96,
      0.026_818_435_096_062_326,
      0.035_099_982_845_393_035,
      0.036_991_141_889_006_51,
      0.053_173_189_208_051_75,
    ],
  );
  assert_close(
    fit.persistence,
    fit.alpha[0] + 0.5 * fit.gamma[0] + fit.beta[0],
    1e-15,
    "persistence",
  );
}

#[test]
fn egarch11_likelihood_matches_arch_at_its_optimum() {
  check_likelihood_at(
    GarchSpec::egarch(1, 1),
    &[
      -0.033_442_240_567_495_09,
      -0.015_109_110_620_694_07,
      0.163_917_069_263_891_07,
      -0.019_912_145_881_267_81,
      0.940_092_471_011_468_6,
    ],
    -1_291.359_015_313_646,
    [
      0.809_632_465_107_058_6,
      0.753_748_816_327_556_6,
      0.526_308_795_962_575_7,
    ],
  );
}

#[test]
fn egarch11_fit_matches_arch() {
  let fit = check_fit(
    GarchSpec::egarch(1, 1),
    &[
      -0.033_442_240_567_495_09,
      -0.015_109_110_620_694_07,
      0.163_917_069_263_891_07,
      -0.019_912_145_881_267_81,
      0.940_092_471_011_468_6,
    ],
    -1_291.359_015_313_646,
    &[
      0.027_355_243_307_150_72,
      0.009_315_775_333_983_197,
      0.046_857_119_896_931_386,
      0.023_466_292_231_184_044,
      0.029_750_787_663_689_35,
    ],
    &[
      0.027_297_973_036_666_867,
      0.010_541_471_614_004_351,
      0.053_824_272_167_873_04,
      0.022_854_334_696_665_805,
      0.035_782_138_540_230_95,
    ],
  );
  assert_close(fit.persistence, fit.beta[0], 1e-15, "persistence");
}

#[test]
fn zero_mean_garch11_fit_matches_arch() {
  let fit = check_fit(
    GarchSpec::garch(1, 1).with_mean(MeanSpec::Zero),
    &[
      0.056_560_358_155_496_364,
      0.086_088_704_385_016_45,
      0.844_345_320_329_681,
    ],
    -1_291.132_223_737_396,
    &[
      0.025_560_962_402_249_008,
      0.025_848_950_564_641_33,
      0.049_498_200_301_157_654,
    ],
    &[
      0.026_989_814_880_712_944,
      0.027_555_152_427_677_285,
      0.053_322_085_329_482_59,
    ],
  );
  assert_eq!(fit.mu, 0.0);
  assert_eq!(fit.spec.param_names(), ["omega", "alpha[1]", "beta[1]"]);
}

#[test]
fn garch21_fit_matches_arch() {
  let fit = check_fit(
    GarchSpec::garch(2, 1),
    &[
      -0.026_523_636_999_036_686,
      0.071_384_684_507_657_78,
      0.066_028_341_917_295_95,
      0.035_463_237_148_953_934,
      0.810_752_349_502_573_9,
    ],
    -1_290.385_997_417_674_8,
    &[
      0.026_630_097_423_094_638,
      0.038_473_758_346_902_52,
      0.035_805_908_924_293_396,
      0.048_674_774_595_477_21,
      0.077_533_973_372_525_18,
    ],
    &[
      0.026_707_218_723_036_514,
      0.048_978_413_555_944_97,
      0.035_794_570_261_030_605,
      0.060_279_832_342_877_676,
      0.102_854_773_473_739_34,
    ],
  );
  assert_eq!(fit.alpha.len(), 2);
  assert_eq!(
    fit.spec.param_names(),
    ["mu", "omega", "alpha[1]", "alpha[2]", "beta[1]"]
  );
}

#[test]
fn transform_round_trips_every_kind() {
  let cases: [(GarchSpec, Vec<f64>); 4] = [
    (GarchSpec::garch(2, 1), vec![0.1, 0.05, 0.04, 0.06, 0.85]),
    (
      GarchSpec::gjr(1, 2),
      vec![-0.2, 0.03, 0.05, 0.08, 0.4, 0.45],
    ),
    (
      GarchSpec::egarch(1, 2).with_mean(MeanSpec::Zero),
      vec![-0.1, 0.2, -0.05, 0.5, 0.4],
    ),
    (
      GarchSpec::garch(1, 0).with_mean(MeanSpec::Zero),
      vec![0.5, 0.3],
    ),
  ];
  for (spec, natural) in cases {
    assert_eq!(natural.len(), spec.n_params());
    let theta = transform::to_unconstrained(&spec, &natural);
    assert_eq!(theta.len(), natural.len());
    let back = transform::to_natural(&spec, &theta);
    for (b, n) in back.iter().zip(&natural) {
      assert_close(*b, *n, 1e-12, "round trip");
    }
  }
}

#[test]
fn param_names_follow_the_layout() {
  assert_eq!(
    GarchSpec::gjr(2, 1).param_names(),
    [
      "mu", "omega", "alpha[1]", "alpha[2]", "gamma[1]", "gamma[2]", "beta[1]"
    ]
  );
  assert_eq!(GarchSpec::gjr(2, 1).n_params(), 7);
  assert_eq!(GarchSpec::egarch(1, 0).n_params(), 4);
}

#[test]
#[should_panic(expected = "p must be at least 1")]
fn rejects_zero_arch_order() {
  let _ = garch_fit(lcg_garch_returns().view(), GarchSpec::garch(0, 1));
}

#[test]
#[should_panic(expected = "need at least 20 observations")]
fn rejects_a_short_series() {
  let r = Array1::from(vec![0.01_f64; 10]);
  let _ = garch_fit(r.view(), GarchSpec::garch(1, 1));
}
