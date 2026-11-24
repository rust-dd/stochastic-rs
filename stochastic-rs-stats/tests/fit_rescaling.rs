//! The distribution, extreme-value and GARCH fits run at unit scale and answer in the data's units:
//! a power-of-two scale moves no bit, and any other the estimate by the search's tolerance only.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::distfit::fit_johnson_su;
use stochastic_rs_stats::distfit::fit_skew_t;
use stochastic_rs_stats::distfit::fit_variance_gamma;
use stochastic_rs_stats::evt::fit_gev;
use stochastic_rs_stats::evt::fit_gpd;
use stochastic_rs_stats::evt::fit_gpd_pwm;
use stochastic_rs_stats::garch::GarchSpec;
use stochastic_rs_stats::garch::fit_garch;

fn normals(seed: u64, n: usize) -> Vec<f64> {
  let mut z = vec![0.0; n];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  z
}

fn skewed() -> Array1<f64> {
  let (z, s) = (normals(6, 2000), normals(7, 2000));
  z.iter()
    .zip(&s)
    .map(|(z, s)| 0.01 * (z + 0.3 * (z * z - 1.0)) * (0.4 * s).exp())
    .collect()
}

fn maxima() -> Array1<f64> {
  normals(8, 10_000)
    .chunks(50)
    .map(|b| b.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    .collect()
}

fn excesses() -> Array1<f64> {
  let (z, e) = (normals(9, 20_000), normals(10, 20_000));
  z.iter()
    .zip(&e)
    .map(|(z, e)| z.abs() / (0.5 + e.abs()))
    .filter(|v| *v > 2.0)
    .map(|v| v - 2.0)
    .collect()
}

fn garch_path() -> Array1<f64> {
  let (omega, alpha, beta) = (1e-6_f64, 0.08, 0.9);
  let mut var = omega / (1.0 - alpha - beta);
  normals(11, 1500)
    .into_iter()
    .map(|z| {
      let r = var.sqrt() * z;
      var = omega + alpha * r * r + beta * var;
      r
    })
    .collect()
}

/// A fit's parameters and their standard errors.
type Fit = Box<dyn Fn(&Array1<f64>) -> Result<(Vec<f64>, Vec<f64>), StatsError>>;

/// Each fit with its data and the power of the data's unit each parameter carries.
fn fits() -> Vec<(&'static str, Array1<f64>, Fit, Vec<i32>)> {
  vec![
    (
      "johnson_su",
      skewed(),
      Box::new(|x| {
        fit_johnson_su(x.view()).map(|f| {
          (
            vec![f.gamma, f.delta, f.xi, f.lambda],
            f.std_errors.to_vec(),
          )
        })
      }),
      vec![0, 0, 1, 1],
    ),
    (
      "skew_t",
      skewed(),
      Box::new(|x| {
        fit_skew_t(x.view()).map(|f| (vec![f.mu, f.sigma, f.eta, f.lambda], f.std_errors.to_vec()))
      }),
      vec![1, 1, 0, 0],
    ),
    (
      "variance_gamma",
      skewed(),
      Box::new(|x| {
        fit_variance_gamma(x.view())
          .map(|f| (vec![f.sigma, f.nu, f.theta, f.mu], f.std_errors.to_vec()))
      }),
      vec![1, 0, 1, 1],
    ),
    (
      "gev",
      maxima(),
      Box::new(|x| fit_gev(x.view()).map(|f| (vec![f.mu, f.sigma, f.xi], f.std_errors.to_vec()))),
      vec![1, 1, 0],
    ),
    (
      "gpd",
      excesses(),
      Box::new(|x| fit_gpd(x.view()).map(|f| (vec![f.sigma, f.xi], f.std_errors.to_vec()))),
      vec![1, 0],
    ),
    (
      "gpd_pwm",
      excesses(),
      // A closed form with no standard error: a thousandth of each estimate stands in for one.
      Box::new(|x| {
        fit_gpd_pwm(x.view())
          .map(|f| (vec![f.sigma, f.xi], vec![1e-3 * f.sigma, 1e-3 * f.xi.abs()]))
      }),
      vec![1, 0],
    ),
    (
      "garch",
      garch_path(),
      Box::new(|x| {
        fit_garch(x.view(), GarchSpec::garch(1, 1))
          .map(|f| (f.params.to_vec(), f.std_errors.to_vec()))
      }),
      vec![1, 2, 0, 0],
    ),
  ]
}

/// `v` in data scaled by `c`, one factor at a time.
fn scaled(v: f64, c: f64, power: i32) -> f64 {
  (0..power).fold(v, |v, _| v * c)
}

#[test]
fn a_power_of_two_scale_moves_no_bit() {
  for (name, x, fit, powers) in fits() {
    let (base, _) = fit(&x).unwrap();
    for c in [2f64.powi(-400), 2f64.powi(400)] {
      let (got, _) = fit(&x.mapv(|v| v * c)).unwrap();
      for (i, &power) in powers.iter().enumerate() {
        assert_eq!(
          got[i].to_bits(),
          scaled(base[i], c, power).to_bits(),
          "{name} at {c:e}: parameter {i} {} against {}",
          got[i],
          scaled(base[i], c, power)
        );
      }
    }
  }
}

/// At `1e±150` and `1e±300` a parameter reads the unit fit's within `1e-3` of its standard error:
/// both fits pass the `1e-7` check, `√(2e-7)` errors from the maximum each. GARCH's variance at
/// `1e±300` has no double.
#[test]
fn a_scale_far_from_one_fits_the_same_law() {
  for (name, x, fit, powers) in fits() {
    let (base, errors) = fit(&x).unwrap();
    for c in [1e-300, 1e-150, 1e150, 1e300] {
      let got = fit(&x.mapv(|v| v * c));
      if name == "garch" && (c == 1e300 || c == 1e-300) {
        assert_eq!(
          got.unwrap_err(),
          StatsError::Numerical {
            stage: "data scale"
          },
          "{name} at {c:e}"
        );
        continue;
      }
      let (got, _) = got.unwrap_or_else(|e| panic!("{name} at {c:e}: {e:?}"));
      for (i, &power) in powers.iter().enumerate() {
        let read = scaled(got[i], 1.0 / c, power);
        assert!(
          (read - base[i]).abs() <= 1e-3 * errors[i],
          "{name} at {c:e}: parameter {i} reads {read} against {}",
          base[i]
        );
      }
    }
  }
}

/// Sixty values near `1.5e308`, whose sum overflows, answer a finite fit or `Numerical`.
#[test]
fn a_sample_whose_sum_overflows_is_fitted_or_numerical() {
  let x = Array1::from_iter((0..60).map(|j| 1.5e308 * (0.9 + 0.1 * ((j * 37) % 61) as f64 / 61.0)));
  for (name, _, fit, _) in fits() {
    match fit(&x) {
      Ok((params, _)) => assert!(params.iter().all(|p| p.is_finite()), "{name}: {params:?}"),
      Err(StatsError::Numerical { .. }) => {}
      Err(e) => panic!("{name}: {e:?}"),
    }
  }
}

/// `2^498` is exact, so the fit at `1e-150` and its twin at `1e-150 · 2^498` share every bit.
#[test]
fn a_gev_fit_at_1e_minus_150_has_an_exact_twin() {
  let (small, twin) = (maxima().mapv(|v| v * 1e-150), 2f64.powi(498));
  let fit = fit_gev(small.view()).unwrap();
  let moved = fit_gev(small.mapv(|v| v * twin).view()).unwrap();
  assert_eq!(fit.message, "converged");
  let pairs = [
    (fit.mu * twin, moved.mu),
    (fit.sigma * twin, moved.sigma),
    (fit.xi, moved.xi),
  ];
  for (a, b) in pairs {
    assert_eq!(a.to_bits(), b.to_bits(), "{a} against {b}");
  }
  assert_eq!(
    (fit.converged, fit.iterations),
    (moved.converged, moved.iterations)
  );
}

/// Uniforms from pairs of normals: `exp(-(z₁² + z₂²)/2)`, a `χ²₂` survival.
fn uniforms(seed: u64, n: usize) -> Vec<f64> {
  normals(seed, 2 * n)
    .chunks(2)
    .map(|z| (-0.5 * (z[0] * z[0] + z[1] * z[1])).exp())
    .collect()
}

/// Each fit on a sample of its own law converges, every estimate within four standard errors.
#[test]
fn a_fit_of_its_own_law_converges_within_four_standard_errors() {
  let within = |name: &str, message: &str, got: &[f64], truth: &[f64], errors: &[f64]| {
    assert_eq!(message, "converged", "{name}");
    for i in 0..truth.len() {
      let off = (got[i] - truth[i]).abs() / errors[i];
      assert!(off < 4.0, "{name}: parameter {i} is {off} errors off");
    }
  };
  let johnson = normals(21, 4_000)
    .into_iter()
    .map(|z| 0.1 + 0.5 * ((z + 0.4) / 1.6).sinh())
    .collect::<Array1<f64>>();
  let f = fit_johnson_su(johnson.view()).unwrap();
  let got = [f.gamma, f.delta, f.xi, f.lambda];
  let errors = f.std_errors.to_vec();
  within(
    "johnson_su",
    &f.message,
    &got,
    &[-0.4, 1.6, 0.1, 0.5],
    &errors,
  );
  let maxima = uniforms(22, 2_000)
    .into_iter()
    .map(|u| 1.0 + 0.5 * ((-u.ln()).powf(-0.1) - 1.0) / 0.1)
    .collect::<Array1<f64>>();
  let f = fit_gev(maxima.view()).unwrap();
  let errors = f.std_errors.to_vec();
  within(
    "gev",
    &f.message,
    &[f.mu, f.sigma, f.xi],
    &[1.0, 0.5, 0.1],
    &errors,
  );
  let excesses = uniforms(23, 2_000)
    .into_iter()
    .map(|u| 2.0 * (u.powf(-0.25) - 1.0) / 0.25)
    .collect::<Array1<f64>>();
  let f = fit_gpd(excesses.view()).unwrap();
  let errors = f.std_errors.to_vec();
  within("gpd", &f.message, &[f.sigma, f.xi], &[2.0, 0.25], &errors);
  let f = fit_garch(garch_path().view(), GarchSpec::garch(1, 1)).unwrap();
  let (got, errors) = (f.params.to_vec(), f.std_errors.to_vec());
  within("garch", &f.message, &got, &[0.0, 1e-6, 0.08, 0.9], &errors);
}
