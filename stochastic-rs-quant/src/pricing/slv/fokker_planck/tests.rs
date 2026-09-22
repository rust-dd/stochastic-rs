use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::traits::Expr;
use stochastic_rs_distributions::traits::Fn2D;
use stochastic_rs_distributions::traits::Grid2D;

use super::*;
use crate::pricing::heston::HestonPricer;
use crate::traits::ModelPricer;

const S0: f64 = 100.0;
const R: f64 = 0.03;
const Q: f64 = 0.01;

fn params(eta: f64) -> HestonSlvParams {
  HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.4,
    rho: -0.6,
    v0: 0.05,
    eta,
  }
}

fn method() -> FokkerPlanckMethod {
  FokkerPlanckMethod::default()
    .with_nodes(121, 60)
    .with_steps_per_year(100)
}

fn unit_leverage() -> Fn2D<f64> {
  Expr::lit(1.0).into()
}

/// Under a unit leverage the model is Heston, whose calls have a closed
/// form: the scheme's marginal has to reprice them. On this mesh — the
/// reference's halved in each direction, its time step doubled — the worst
/// error over the ladder is 0.008 on a spot of 100, and 0.003 at the default
/// settings; the bound is twice the former.
#[test]
fn a_unit_leverage_density_reprices_the_closed_form_heston_calls() {
  let p = params(1.0);
  let density =
    heston_slv_density(&p, S0, R, Q, &unit_leverage(), &[0.25, 1.0], &method()).unwrap();
  let exact = HestonPricer::new(p.v0, p.rho, p.kappa, p.theta, p.sigma, Some(0.0));
  for (index, tau) in [(0, 0.25), (1, 1.0)] {
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
      let fv = density.call_price(index, k, R, tau);
      let closed = exact.price_call(S0, k, R, Q, tau);
      assert!(
        (fv - closed).abs() < 0.02,
        "tau = {tau}, K = {k}: finite volume {fv}, closed form {closed}"
      );
    }
  }
}

/// Every boundary flux is zero, so the mass never moves; the log-spot
/// marginal's mean is the forward and the variance's mean the CIR mean, the
/// two first moments the exact density carries.
#[test]
fn the_density_keeps_its_mass_and_its_first_moments() {
  let p = params(0.7);
  let maturities = [0.1, 0.5, 1.0];
  let density = heston_slv_density(&p, S0, R, Q, &unit_leverage(), &maturities, &method()).unwrap();
  let xi = p.sigma_mixed();
  for (index, &tau) in maturities.iter().enumerate() {
    let mass = density.mass[index];
    assert!((mass - 1.0).abs() < 1e-8, "mass at tau = {tau} is {mass}");
    let forward = S0 * ((R - Q) * tau).exp();
    let mean = density.spot_mean(index);
    assert!(
      (mean / forward - 1.0).abs() < 2e-3,
      "spot mean {mean} vs forward {forward} at tau = {tau}"
    );
    let cir = p.theta + (p.v0 - p.theta) * (-p.kappa * tau).exp();
    let v_mean = density.variance_means[index];
    assert!(
      (v_mean - cir).abs() < 2e-3 * cir.max(xi * xi),
      "variance mean {v_mean} vs CIR mean {cir} at tau = {tau}"
    );
    assert!(density.marginals[index].iter().all(|d| d.is_finite()));
  }
}

/// A leverage that varies with the spot enters the scheme through the
/// advection, the diffusion and the mixed flux: the call prices move away
/// from the unit-leverage ones in the direction a lower volatility implies.
#[test]
fn a_smaller_leverage_lowers_the_option_value() {
  let p = params(1.0);
  let unit = heston_slv_density(&p, S0, R, Q, &unit_leverage(), &[0.5], &method()).unwrap();
  let damped: Fn2D<f64> = Expr::lit(0.8).into();
  let low = heston_slv_density(&p, S0, R, Q, &damped, &[0.5], &method()).unwrap();
  let (c_unit, c_low) = (
    unit.call_price(0, 100.0, R, 0.5),
    low.call_price(0, 100.0, R, 0.5),
  );
  assert!(
    c_low < c_unit,
    "L = 0.8 prices {c_low}, L = 1 prices {c_unit}"
  );
  assert!((low.mass[0] - 1.0).abs() < 1e-8);
}

fn smiling_local_vol() -> Grid2D<f64> {
  let ts = Array1::from_vec(vec![0.1, 0.5, 1.0]);
  let ks = Array1::linspace(50.0, 200.0, 31);
  let values = Array2::from_shape_fn((ts.len(), ks.len()), |(j, i)| {
    0.18 + 0.12 * (ks[i] / S0 - 1.0).powi(2) + 0.03 * ts[j]
  });
  Grid2D::new(ts, ks, values)
}

/// The calibrated surface is laid on the log-spot mesh with the spot a node,
/// starts at `t = 0` with the first computed row as the reference
/// prescribes, stays finite and positive, and the calibration's own
/// marginal reprices the local-volatility model's forward.
#[test]
fn the_calibration_lays_a_finite_surface_on_the_mesh_and_keeps_the_forward() {
  let p = params(0.6);
  let run =
    calibrate_leverage_fokker_planck(&p, S0, R, Q, &smiling_local_vol(), &[0.25, 0.75], &method())
      .unwrap();
  let lev = &run.leverage;
  assert_eq!(lev.times()[0], 0.0);
  assert_eq!(lev.horizon(), 0.75);
  assert!(
    lev.spots().iter().any(|s| (s - S0).abs() < 1e-9),
    "the spot is a node"
  );
  assert_eq!(
    lev.values().row(0),
    lev.values().row(1),
    "row 0 is the first computed row"
  );
  assert!(lev.values().iter().all(|l| l.is_finite() && *l > 0.0));
  let atm = lev.interpolate(S0, 0.5);
  assert!((0.5..2.0).contains(&atm), "ATM leverage {atm}");
  for (index, tau) in [(0, 0.25), (1, 0.75)] {
    let forward = S0 * ((R - Q) * tau).exp();
    let mean = run.density.spot_mean(index);
    assert!(
      (mean / forward - 1.0).abs() < 2e-3,
      "spot mean {mean} vs forward {forward}"
    );
    assert!((run.density.mass[index] - 1.0).abs() < 1e-8);
  }
}

/// Under `eta = 0` at the long-run variance the variance stays put and the
/// calibration condition is the closed form `L = sigma_LV / sqrt(v0)`: the
/// trapezoid on `|P|` returns it up to the central scheme's leakage into the
/// neighbouring variance cells.
#[test]
fn eta_zero_at_the_long_run_variance_is_close_to_the_closed_form() {
  let p = HestonSlvParams {
    v0: 0.04,
    theta: 0.04,
    ..params(0.0)
  };
  let grid = smiling_local_vol();
  let run = calibrate_leverage_fokker_planck(&p, S0, R, Q, &grid, &[0.5], &method()).unwrap();
  let lev = &run.leverage;
  let mut worst = 0.0_f64;
  for (j, &t) in lev.times().iter().enumerate().skip(1) {
    for (i, &s) in lev.spots().iter().enumerate() {
      if !(70.0..=140.0).contains(&s) {
        continue;
      }
      let expected = grid.eval(t, s) / 0.2;
      worst = worst.max((lev.values()[[j, i]] / expected - 1.0).abs());
    }
  }
  assert!(
    worst < 0.03,
    "worst relative deviation from the closed form is {worst}"
  );
}

#[test]
fn bad_settings_are_errors() {
  let p = params(1.0);
  let grid = smiling_local_vol();
  let run = |m: FokkerPlanckMethod| {
    calibrate_leverage_fokker_planck(&p, S0, R, Q, &grid, &[0.5], &m).map(|_| ())
  };
  assert!(run(method().with_nodes(2, 60)).is_err());
  assert!(run(method().with_inner_iterations(0)).is_err());
  assert!(run(method().with_theta(0.0)).is_err());
  assert!(run(method().with_v_max(0.01)).is_err());
  assert!(run(method().with_x_half_width(-1.0)).is_err());
  assert!(run(method().with_x_stretch(0.0)).is_err());
  assert!(calibrate_leverage_fokker_planck(&p, S0, R, Q, &grid, &[], &method()).is_err());
}
