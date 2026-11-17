use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::ou::Ou;

use super::*;
use crate::mle::fit_mle;
use crate::qmle::fit_ou_mle;
use crate::qmle::test_support::cir;
use crate::qmle::test_support::simulate_ou;

fn ou(theta: f64, mu: f64, sigma: f64) -> Ou<f64, Deterministic> {
  Ou::new(theta, mu, sigma, 2, None, None, Deterministic::new(0))
}

/// The mean and variance of `sde::dcKessler` as its source writes them (`R/dcApprox.R:37-39`).
fn dc_kessler(model: &dyn DiffusionModel, x0: f64, dt: f64) -> (f64, f64) {
  let (mu, mu1, mu2) = (
    model.drift(x0, 0.0),
    model.drift_x(x0, 0.0),
    model.drift_xx(x0, 0.0),
  );
  let (sg, sg1, sg2) = (
    model.diffusion(x0, 0.0),
    model.diffusion_x(x0, 0.0),
    model.diffusion_xx(x0, 0.0),
  );
  let ex = x0 + mu * dt + (mu * mu1 + 0.5 * (sg * sg * mu2)) * (dt * dt) / 2.0;
  let vx = x0 * x0
    + (2.0 * mu * x0 + sg * sg) * dt
    + (2.0 * mu * (mu1 * x0 + mu + sg * sg1)
      + sg * sg * (mu2 * x0 + 2.0 * mu1 + sg1 * sg1 + sg * sg2))
      * (dt * dt)
      / 2.0
    - ex * ex;
  (ex, vx)
}

/// Kessler's moments are `dcKessler`'s, and for OU and CIR the closed forms its expansion has:
/// the variance `σ²Δ(1 − θΔ) + θb²Δ³(1 − θΔ/4)` for OU, with `b` the drift.
#[test]
fn kesslers_moments_are_those_of_dc_kessler() {
  let near = |got: f64, want: f64, tol: f64| (got / want - 1.0).abs() < tol;
  for dt in [1.0_f64 / 252.0, 0.01, 1.0 / 12.0] {
    for x0 in [0.02, 0.4, 1.3] {
      let (th, m, s) = (2.0, 1.0, 0.3);
      let drift = th * (m - x0);
      let tail = th * drift * drift * dt.powi(3) * (1.0 - th * dt / 4.0);
      let mean = x0 + drift * dt * (1.0 - th * dt / 2.0);
      for (model, var) in [
        (
          &ou(th, m, s) as &dyn DiffusionModel,
          s * s * dt * (1.0 - th * dt) + tail,
        ),
        (
          &cir(th, m, s),
          s * s * dt * (x0 * (1.0 - 1.5 * th * dt) + th * m * dt / 2.0) + tail,
        ),
      ] {
        let got = DensityApprox::moments_kessler(model, x0, 0.0, dt);
        let source = dc_kessler(model, x0, dt);
        assert!(
          near(got.0, source.0, 1e-13) && near(got.1, source.1, 1e-8),
          "{x0}, {dt}: {got:?} for dcKessler's {source:?}"
        );
        assert!(
          near(got.0, mean, 1e-13) && near(got.1, var, 1e-11),
          "{x0}, {dt}: {got:?} for ({mean}, {var})"
        );
      }
    }
  }
  // mpmath, 50 digits, on `dcKessler`'s lines.
  let got = DensityApprox::moments_kessler(&cir(3.0, 0.3, 0.2), 0.4, 0.0, 1.0 / 250.0);
  assert!(
    near(got.0, 0.398_807_2, 1e-11) && near(got.1, 6.315_322_816e-5, 1e-9),
    "{got:?}"
  );
}

/// Where the expansion's variance is not positive, as past `θΔ = 1` at the level of an OU, there
/// is no density, as in `dcKessler` (`Vx[Vx<0] <- NA`): not that of its absolute value.
#[test]
fn kesslers_density_is_nan_where_its_variance_is_not_positive() {
  let model = ou(300.0, 0.5, 0.3);
  let (_, var) = DensityApprox::moments_kessler(&model, 0.5, 0.0, 0.01);
  assert!((var + 0.0018).abs() < 1e-12, "{var}");
  assert!(
    DensityApprox::Kessler
      .density(&model, 0.5, 0.5, 0.0, 0.01)
      .is_nan()
  );
  assert!(
    DensityApprox::Kessler
      .ln_density(&model, 0.5, 0.5, 0.0, 0.01)
      .is_nan()
  );
  // Off the level the drift's term makes it positive again: 0.0657, mpmath's density beside it.
  let off = DensityApprox::Kessler.density(&model, 0.6, 0.5, 0.0, 0.01);
  assert!((off - 0.967_289_705_640_249).abs() < 1e-7, "{off}");
}

/// Kessler's `σ̂` is the exact MLE's to second order in `θΔ`: daily at a speed of 8 within
/// `(θΔ)²/3 = 3.4e-4`, where a variance without the `2b′σ²` term was off by `θΔ/2 = 1.6 %`.
#[test]
fn a_kessler_fit_of_an_ou_path_has_the_exact_fits_sigma_to_second_order() {
  let dt = 1.0 / 252.0;
  for seed in [1u64, 2, 3] {
    let path = simulate_ou(8.0, 1.0, 0.3, 1.0, dt, 1000, seed);
    let closed = fit_ou_mle(path.view(), dt).unwrap();
    let mut model = ou(8.0, 1.0, 0.3);
    let fit = fit_mle(&mut model, path.view(), dt, DensityApprox::Kessler, None).unwrap();
    assert!(fit.converged, "seed {seed}");
    assert!(
      (fit.params[2] / closed.sigma - 1.0).abs() < 2e-3,
      "seed {seed}: σ̂ = {} for {}",
      fit.params[2],
      closed.sigma
    );
  }
}

/// Elerian's density is the Milstein step's law (Elerian 1998, eq. 2.5) and integrates to one;
/// in `u`, with `x = lo + u²` from the support's end, its root singularity there is gone.
#[test]
fn elerians_density_integrates_to_one() {
  let (model, dt) = (cir(2.0, 0.04, 0.3), 1.0 / 252.0);
  for x0 in [1e-6, 1e-3, 0.05, 0.5] {
    let (sig, sig_x) = (model.diffusion(x0, 0.0), model.diffusion_x(x0, 0.0));
    let a = 0.5 * sig * sig_x * dt;
    let lo = x0 + model.drift(x0, 0.0) * dt - a - 0.5 * sig / sig_x;
    let sd = (sig * sig * dt + 2.0 * a * a).sqrt();
    let (top, n) = ((x0 + 40.0 * sd - lo).sqrt(), 400_000);
    let h = top / n as f64;
    let mass = (0..n)
      .map(|i| {
        let u = h * (i as f64 + 0.5);
        DensityApprox::Elerian.density(&model, x0, lo + u * u, 0.0, dt) * 2.0 * u * h
      })
      .sum::<f64>();
    assert!((mass - 1.0).abs() < 1e-6, "{x0}: {mass}");
  }
}

/// The Milstein step's log-density against mpmath's, there summed over the two roots of the step
/// in its normal draw, which uses neither Elerian's square nor his non-central χ².
#[test]
fn elerians_log_density_is_the_milstein_steps() {
  let (model, dt) = (cir(2.0, 0.04, 0.3), 1.0 / 252.0);
  for (x0, xt, reference) in [
    (1e-9, 0.05, -273.476_060_320_268),
    (1e-5, 0.05, -266.327_254_327_862),
    (1e-3, 0.01, -19.977_873_609_906_2),
    (0.05, 0.051, 4.498_262_554_308_18),
    (0.05, 0.5, -1_306.374_166_590_9),
    (0.5, 0.5, 3.353_575_108_573_65),
  ] {
    let ln = DensityApprox::Elerian.ln_density(&model, x0, xt, 0.0, dt);
    assert!(
      (ln - reference).abs() < 1e-9 * (1.0 + reference.abs()),
      "{x0} to {xt}: {ln} for {reference}"
    );
  }
  let peak = DensityApprox::Elerian.density(&model, 0.05, 0.05, 0.0, dt);
  assert!((peak.ln() - 4.545_136_215_442_17).abs() < 1e-9, "{peak}");
}

/// From a start near zero Kessler reads both derivatives of `σ√x`, each without bound there:
/// with the closed forms its law is `dcKessler`'s, here against mpmath's.
#[test]
fn kesslers_cir_density_holds_from_a_start_near_zero() {
  let (model, dt) = (cir(2.0, 0.04, 0.3), 1.0 / 252.0);
  let near_zero = DensityApprox::Kessler.ln_density(&model, 1e-9, 0.0003, 0.0, dt);
  assert!(
    (near_zero - 7.414_623_635_640_39).abs() < 1e-6,
    "{near_zero}"
  );
  let ordinary = DensityApprox::Kessler.ln_density(&model, 0.05, 0.051, 0.0, dt);
  assert!((ordinary - 4.519_108_977_014_08).abs() < 1e-6, "{ordinary}");
}

/// The mean and variance of `sde::dcOzaki` as its source writes them (`R/dcApprox.R:46-49`).
fn dc_ozaki(model: &dyn DiffusionModel, x0: f64, dt: f64) -> (f64, f64) {
  let (d, lx) = (model.drift(x0, 0.0), model.drift_x(x0, 0.0));
  let s = model.diffusion(x0, 0.0);
  let kx = (1.0 + d * ((lx * dt).exp() - 1.0) / (x0 * lx)).ln() / dt;
  let ex = x0 + d / lx * ((lx * dt).exp() - 1.0);
  let vx = s * s * ((2.0 * kx * dt).exp() - 1.0) / (2.0 * kx);
  (ex, vx)
}

/// Ozaki's moments are `dcOzaki`'s on both sides of the level, where `K` changes sign: a state
/// the step shrinks has `K < 0` and its own variance, not Euler's, which is off by `θΔ` there.
#[test]
fn ozakis_moments_are_those_of_dc_ozaki() {
  let near = |got: f64, want: f64, tol: f64| (got / want - 1.0).abs() < tol;
  for dt in [1.0_f64 / 252.0, 0.01, 1.0 / 12.0] {
    for x0 in [0.02, 0.4, 1.3, 2.5] {
      for model in [
        &ou(2.0, 1.0, 0.3) as &dyn DiffusionModel,
        &cir(2.0, 1.0, 0.3),
      ] {
        let got = DensityApprox::moments_ozaki(model, x0, 0.0, dt);
        let source = dc_ozaki(model, x0, dt);
        assert!(
          near(got.0, source.0, 1e-13) && near(got.1, source.1, 1e-9),
          "{x0}, {dt}: {got:?} for dcOzaki's {source:?}"
        );
      }
    }
  }
  // mpmath, 50 digits, on `dcOzaki`'s lines: the law, the step, the mean, the variance, the log.
  let (daily, monthly) = (1.0 / 252.0, 1.0 / 12.0);
  for (model, x0, xt, dt, mean, var, ln) in [
    (
      &ou(2.0, 1.0, 0.3) as &dyn DiffusionModel,
      (1.3, 1.28, daily),
      (1.297_628_470_896_995_9, 3.564_915_362_477_539e-4),
      2.614_798_433_160_978,
    ),
    (
      &ou(8.0, 0.03, 0.3),
      (-0.1, -0.05, monthly),
      (-0.036_744_225_474_236_96, 3.239_845_701_323_553e-3),
      1.920_058_363_539_173_6,
    ),
    (
      &cir(2.0, 0.04, 0.3),
      (0.05, 0.051, daily),
      (0.049_920_949_029_899_864, 1.782_891_780_717_299e-5),
      4.515_752_438_279_622,
    ),
    (
      &cir(2.0, 0.04, 0.3),
      (1e-9, 0.0003, daily),
      (3.162_048_724_954_485_6e-4, 1.409_850_923_720_677_6e-3),
      2.363_197_027_685_320_6,
    ),
  ]
  .map(|(model, (x0, xt, dt), (mean, var), ln)| (model, x0, xt, dt, mean, var, ln))
  {
    let got = DensityApprox::moments_ozaki(model, x0, 0.0, dt);
    assert!(
      near(got.0, mean, 1e-10) && near(got.1, var, 1e-8),
      "{x0}: {got:?} for ({mean}, {var})"
    );
    let got = DensityApprox::Ozaki.ln_density(model, x0, xt, 0.0, dt);
    assert!((got - ln).abs() < 1e-7, "{x0} to {xt}: {got} for {ln}");
  }
}

/// Where the step carries the state across zero `K`'s logarithm has no value and there is no
/// density, as in `dcOzaki`: not one built on `K = μ_x` or on Euler's variance.
#[test]
fn ozakis_density_is_nan_where_the_step_crosses_zero() {
  let (model, dt) = (ou(8.0, 0.03, 0.3), 1.0 / 12.0);
  let (mean, var) = DensityApprox::moments_ozaki(&model, -0.01, 0.0, dt);
  assert!(mean > 0.0 && var.is_nan(), "{mean}, {var}");
  let ozaki = DensityApprox::Ozaki;
  assert!(ozaki.density(&model, -0.01, 0.0, 0.0, dt).is_nan());
  assert!(ozaki.ln_density(&model, -0.01, 0.0, 0.0, dt).is_nan());
}

/// On the set `x₀μ_x = 0` Ozaki (1992, eq. 2.12) defines the step by its mean and Euler's
/// variance; on the level, where `K = 0`, the variance's limit is Euler's as well.
#[test]
fn ozakis_law_at_a_zero_state_or_slope_is_his_eq_2_12() {
  let near = |got: (f64, f64), want: (f64, f64)| {
    (got.0 - want.0).abs() < 1e-12 && (got.1 / want.1 - 1.0).abs() < 1e-14
  };
  let (model, dt) = (ou(2.0, 1.0, 0.3), 0.01);
  let euler = 0.3 * 0.3 * dt;
  let from_zero = DensityApprox::moments_ozaki(&model, 0.0, 0.0, dt);
  assert!(
    near(from_zero, (-(-0.02_f64).exp_m1(), euler)),
    "{from_zero:?}"
  );
  let on_the_level = DensityApprox::moments_ozaki(&model, 1.0, 0.0, dt);
  assert!(near(on_the_level, (1.0, euler)), "{on_the_level:?}");
  let flat = Ckls::new(0.1, 0.0, 0.2, 0.5, 2, None, None, Deterministic::new(0));
  let no_slope = DensityApprox::moments_ozaki(&flat, 0.5, 0.0, dt);
  assert!(near(no_slope, (0.501, 0.02 * dt)), "{no_slope:?}");
}

/// An OU path about a level of `0.03` crosses zero, and at its law a monthly step carries the
/// states just below zero across it: Ozaki's likelihood has no value there, and the fit is `Err`.
#[test]
fn an_ozaki_fit_is_an_err_where_the_laws_steps_cross_zero() {
  let dt = 1.0 / 12.0;
  let path = simulate_ou(8.0, 0.03, 0.3, 0.03, dt, 1000, 1001);
  let mut model = ou(8.0, 0.03, 0.3);
  let fit = fit_mle(&mut model, path.view(), dt, DensityApprox::Ozaki, None);
  let stage = "log-likelihood";
  assert!(matches!(fit, Err(crate::StatsError::Numerical { stage: s }) if s == stage));
}

/// On a path that stays above zero Ozaki's fit converges, with the exact fit's `σ̂` to first order
/// in `θΔ`.
#[test]
fn an_ozaki_fit_of_a_path_above_zero_converges_beside_the_exact_fit() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, 1001);
  assert!(path.iter().all(|&x| x > 0.0));
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let mut model = ou(2.0, 1.0, 0.3);
  let fit = fit_mle(&mut model, path.view(), dt, DensityApprox::Ozaki, None).unwrap();
  assert!(fit.converged);
  assert!(
    (fit.params[2] / closed.sigma - 1.0).abs() < 0.02,
    "σ̂ = {} for {}",
    fit.params[2],
    closed.sigma
  );
}
