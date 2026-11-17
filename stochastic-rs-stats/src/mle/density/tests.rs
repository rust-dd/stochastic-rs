use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;

use super::*;

fn cir(theta: f64, mu: f64, sigma: f64) -> Cir<f64, Deterministic> {
  let floor = VarianceBoundary::Floor;
  Cir::new(
    theta,
    mu,
    sigma,
    2,
    None,
    None,
    floor,
    Deterministic::new(0),
  )
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
