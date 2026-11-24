use gauss_quad::GaussLegendre;

use super::mittag_leffler;
use super::ratios;
use crate::pricing::fourier::FourierModelExt;
use crate::pricing::fourier::RoughHeston;
use crate::pricing::fourier::RoughHestonCf;
use crate::pricing::fourier::RoughHestonParams;

/// `c₂ − θT = ¼ν²θ∫₀ᵀG² − ρνθ∫₀ᵀG` at `V₀ = θ`, `G(x) = x^αE_{α,α+1}(−κx^α)`: the variance of the
/// log-price in closed form, which reads the chf's `ρν` and `ν²` terms without a simulation.
fn excess_variance(p: &RoughHestonParams, t: f64) -> f64 {
  let alpha = p.hurst + 0.5;
  let g = |x: f64| {
    let y = x.powf(alpha);
    y * mittag_leffler(alpha, alpha + 1.0, -p.kappa * y)
  };
  // `x = tu³` smooths the `x^α` start for the Gauss–Legendre rule.
  let rule = GaussLegendre::new(64.try_into().expect("64 > 0"));
  let integral =
    |f: &dyn Fn(f64) -> f64| rule.integrate(0.0, 1.0, |u| f(t * u * u * u) * 3.0 * t * u * u);
  let (first, second) = (integral(&g), integral(&|x| g(x) * g(x)));
  0.25 * p.nu * p.nu * p.theta * second - p.rho * p.nu * p.theta * first
}

/// The Adams error falls at the scheme's order (above 1.2 a halving), under 2e-4 of the excess at
/// 400 steps; the (4,4) approximant sits 7.0e-5 away at worst, and is held to twice that.
#[test]
fn the_second_cumulant_is_the_closed_form_s() {
  for p in [
    RoughHestonParams::new(0.04, 1.0, 0.04, 0.4, -0.65, 0.1),
    RoughHestonParams::new(0.02, 0.3, 0.02, 0.3, -0.7, 0.05),
  ] {
    let t = 1.0;
    let want = excess_variance(&p, t);
    let gap = |cf: RoughHestonCf| {
      let c2 = RoughHeston::new(p, cf).cumulants(t).c2;
      (c2 - p.theta * t - want).abs() / want.abs()
    };
    let errors = [100, 200, 400].map(|steps| gap(RoughHestonCf::Adams { steps }));
    assert!(errors[2] < 2e-4, "{p:?}: {errors:?}");
    for ratio in ratios(&errors) {
      assert!(ratio > 2.0_f64.powf(1.2), "{p:?}: {errors:?}");
    }
    let pade = gap(RoughHestonCf::default());
    assert!(pade < 1.5e-4, "{p:?}: {pade}");
  }
}
