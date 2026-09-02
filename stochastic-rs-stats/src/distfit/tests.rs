use ndarray::Array1;

use super::*;

/// Integer LCG uniforms, bit-identical to the Python generator behind the
/// reference optima below.
fn lcg(seed: u64, n: usize) -> Vec<f64> {
  let mut x = seed;
  (0..n)
    .map(|_| {
      x = (1_103_515_245 * x + 12_345) % 2_147_483_648;
      x as f64 / 2_147_483_648.0
    })
    .collect()
}

/// Sum-of-twelve-uniform normal proxies pushed through a skewed cubic,
/// `x = 0.05 + 0.6 z - 0.1 z² + c z³`: `c = 0.2` (heavy tails) for the
/// Johnson SU and VG fits, `c = 0.03` for the skew-t fit, where the heavy
/// version drives η to its boundary.
fn skewed_sample(cubic: f64) -> Array1<f64> {
  let u = lcg(17, 12 * 1_500);
  (0..1_500)
    .map(|t| {
      let z = u[12 * t..12 * t + 12].iter().sum::<f64>() - 6.0;
      0.05 + 0.6 * z - 0.1 * z * z + cubic * z * z * z
    })
    .collect()
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
  assert!((got - want).abs() < tol, "{what}: {got} vs {want}");
}

#[test]
fn samples_are_bit_identical_to_the_python_generator() {
  let x = skewed_sample(0.2);
  assert_close(x[0], -1.480_411_019_663_611_8, 1e-13, "x[0]");
  assert_close(x[1_499], 0.222_922_189_986_181_77, 1e-13, "x[1499]");
  let x2 = skewed_sample(0.03);
  assert_close(x2[0], -1.040_191_321_436_558_8, 1e-13, "x2[0]");
  assert_close(x2[1_499], 0.218_595_920_201_029_36, 1e-13, "x2[1499]");
}

/// Reference: a tightly converged `scipy.optimize` Nelder–Mead on the same
/// log-likelihood (`scipy.stats.johnsonsu.fit` lands within 3e-5 of it).
#[test]
fn johnson_su_fit_matches_the_scipy_optimum() {
  let fit = johnson_su_fit(skewed_sample(0.2).view());
  assert!(fit.converged);
  assert_eq!(fit.nobs, 1_500);
  let loglik = -2_207.705_168_550_967_3;
  assert!(
    fit.log_likelihood >= loglik - 1e-6,
    "{}",
    fit.log_likelihood
  );
  assert_close(fit.log_likelihood, loglik, 1e-5, "log-likelihood");
  assert_close(fit.gamma, 0.260_943_669_141_750_43, 5e-4, "gamma");
  assert_close(fit.delta, 0.876_365_866_049_264_5, 5e-4, "delta");
  assert_close(fit.xi, 0.221_269_271_922_239_47, 5e-4, "xi");
  assert_close(fit.lambda, 0.566_675_921_447_501_7, 5e-4, "lambda");
  assert!(fit.std_errors.iter().all(|s| s.is_finite() && *s > 0.0));
  assert_close(fit.aic, 8.0 - 2.0 * fit.log_likelihood, 1e-9, "aic");
}

/// Reference: `scipy.optimize` Nelder–Mead on `arch`'s `SkewStudent`
/// log-likelihood with location and scale.
#[test]
fn skew_t_fit_matches_the_arch_optimum() {
  let fit = skew_t_fit(skewed_sample(0.03).view());
  assert!(fit.converged);
  let loglik = -1_538.591_372_939_148_2;
  assert!(
    fit.log_likelihood >= loglik - 1e-6,
    "{}",
    fit.log_likelihood
  );
  assert_close(fit.log_likelihood, loglik, 1e-5, "log-likelihood");
  assert_close(fit.mu, -0.069_018_291_705_388_34, 5e-4, "mu");
  assert_close(fit.sigma, 0.736_023_498_453_124_9, 5e-4, "sigma");
  assert_close(fit.eta, 6.697_173_403_367_799, 5e-3, "eta");
  assert_close(fit.lambda, -0.437_618_601_032_002_3, 5e-4, "lambda");
  assert!(fit.eta > 2.0 && fit.lambda.abs() < 1.0);
  assert!(fit.std_errors.iter().all(|s| s.is_finite() && *s > 0.0));
  assert_close(
    fit.bic,
    4.0 * 1_500_f64.ln() - 2.0 * fit.log_likelihood,
    1e-9,
    "bic",
  );
}

/// Reference: `scipy.optimize` Nelder–Mead on the Bessel-form VG
/// log-density evaluated with `scipy.special.kve`.
/// The likelihood has a cusp in μ at every data point; both optimisers
/// land on the same one, and the profile pass here ends a few 1e-5 above
/// the reference likelihood.
#[test]
fn variance_gamma_fit_matches_the_scipy_optimum() {
  let fit = variance_gamma_fit(skewed_sample(0.2).view());
  assert!(fit.converged);
  let loglik = -2_215.084_505_676_344_6;
  assert!(
    fit.log_likelihood >= loglik - 1e-6,
    "{}",
    fit.log_likelihood
  );
  assert_close(fit.log_likelihood, loglik, 5e-5, "log-likelihood");
  assert_close(fit.sigma, 1.209_304_845_131_462, 5e-4, "sigma");
  assert_close(fit.nu, 1.545_582_185_003_798, 1e-3, "nu");
  assert_close(fit.theta, -0.248_374_498_213_412_4, 5e-4, "theta");
  assert_close(fit.mu, 0.157_844_590_842_724_4, 5e-4, "mu");
  assert!(fit.std_errors.iter().all(|s| s.is_finite() && *s > 0.0));
}

#[test]
#[should_panic(expected = "need at least 10 observations")]
fn rejects_a_short_series() {
  let _ = johnson_su_fit(Array1::from(vec![1.0_f64, 2.0, 3.0]).view());
}

#[test]
#[should_panic(expected = "observations must not be constant")]
fn rejects_a_constant_series() {
  let _ = skew_t_fit(Array1::from(vec![1.0_f64; 20]).view());
}
