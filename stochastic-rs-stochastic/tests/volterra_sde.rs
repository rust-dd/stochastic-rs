//! Tests for [`VolterraSde`]/[`VolterraLift`] against [`reference_path`], the permanent $O(n^2)$
//! cross-implementation oracle for the Markov lift (`reference.rs` says why it is kept).
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::rough::kernel::RlKernel;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volterra::ExponentialKernel;
use stochastic_rs_stochastic::volterra::VolterraKernel;
use stochastic_rs_stochastic::volterra::VolterraLift;
use stochastic_rs_stochastic::volterra::VolterraSde;
use stochastic_rs_stochastic::volterra::reference::reference_path;

fn mean_reverting_drift(_t: f64, x: f64) -> f64 {
  0.6 * (1.0 - x)
}

fn const_diffusion(_t: f64, _x: f64) -> f64 {
  0.3
}

fn zero_drift(_t: f64, _x: f64) -> f64 {
  0.0
}

fn state_dependent_diffusion(_t: f64, x: f64) -> f64 {
  0.2 + 0.1 * x
}

fn one_diffusion(_t: f64, _x: f64) -> f64 {
  1.0
}

/// The lift against the direct $O(n^2)$ convolution, with a fitted kernel and a state-dependent
/// drift: `7.04e-4` measured, inside $4\times$ the fit's $5\text{e-}3$, the margin for the drift.
#[test]
fn lift_agrees_with_the_reference_convolution() {
  const REL_TOL: f64 = 4.0 * 5e-3;

  let hurst = 0.45_f64;
  let degree = 150_usize;
  let n = 100_usize;
  let t_max = 1.0_f64;
  let dt = t_max / (n as f64 - 1.0);
  let x0 = 0.2_f64;
  let kernel = RlKernel::<f64>::new(hurst, degree);

  let gn = Gn::<f64, Deterministic>::new(n - 1, Some(t_max), Deterministic::new(2026));
  let dw = gn.sample();
  let dw_slice = dw.as_slice().expect("dw contiguous");

  let lift = VolterraLift::new(kernel.clone(), dt);
  let lift_path = lift.simulate(x0, mean_reverting_drift, const_diffusion, dw_slice);

  let ref_path = reference_path(
    |tau: f64| VolterraKernel::evaluate(&kernel, tau),
    mean_reverting_drift,
    const_diffusion,
    x0,
    dt,
    dw_slice,
  );

  for i in 0..n {
    let tol = REL_TOL * (1.0 + ref_path[i].abs());
    let diff = (lift_path[i] - ref_path[i]).abs();
    assert!(
      diff <= tol,
      "i={i}: lift={} ref={} |diff|={diff:e} exceeds tol={tol:e}",
      lift_path[i],
      ref_path[i]
    );
  }
}

/// An exponential kernel is exact at $N'=1$, and zero drift leaves only the diffusion, where lift
/// and reference apply the same $K$: reassociation noise, `6.80e-16` measured against `1e-9`.
#[test]
fn exponential_kernel_lift_matches_reference_to_machine_precision() {
  const REL_TOL: f64 = 1e-9;

  let beta = 0.7_f64;
  let c = 1.3_f64;
  let x0 = 0.2_f64;
  let t_max = 1.0_f64;
  let kernel = ExponentialKernel::new(beta, c);

  for n in [16_usize, 32, 64, 128, 256] {
    let dt = t_max / (n as f64 - 1.0);
    let gn = Gn::<f64, Deterministic>::new(n - 1, Some(t_max), Deterministic::new(7));
    let dw = gn.sample();
    let dw_slice = dw.as_slice().expect("dw contiguous");

    let lift = VolterraLift::new(kernel.clone(), dt);
    let lift_path = lift.simulate(x0, zero_drift, state_dependent_diffusion, dw_slice);

    let ref_path = reference_path(
      |tau: f64| VolterraKernel::evaluate(&kernel, tau),
      zero_drift,
      state_dependent_diffusion,
      x0,
      dt,
      dw_slice,
    );

    for i in 0..n {
      let tol = REL_TOL * (1.0 + ref_path[i].abs());
      let diff = (lift_path[i] - ref_path[i]).abs();
      assert!(
        diff <= tol,
        "n={n} i={i}: lift={} ref={} |diff|={diff:e} exceeds tol={tol:e}",
        lift_path[i],
        ref_path[i]
      );
    }
  }
}

/// The counterfactual: the same setup with [`mean_reverting_drift`] misses `1e-9` by a gap of
/// `2.55e-3`, the drift's exact integral against the reference's left-Riemann sum.
#[test]
fn nonzero_drift_breaks_the_machine_precision_tolerance() {
  const TIGHT_TOL: f64 = 1e-9;

  let beta = 0.7_f64;
  let c = 1.3_f64;
  let x0 = 0.2_f64;
  let t_max = 1.0_f64;
  let n = 16_usize;
  let dt = t_max / (n as f64 - 1.0);
  let kernel = ExponentialKernel::new(beta, c);

  let gn = Gn::<f64, Deterministic>::new(n - 1, Some(t_max), Deterministic::new(7));
  let dw = gn.sample();
  let dw_slice = dw.as_slice().expect("dw contiguous");

  let lift = VolterraLift::new(kernel.clone(), dt);
  let lift_path = lift.simulate(
    x0,
    mean_reverting_drift,
    state_dependent_diffusion,
    dw_slice,
  );

  let ref_path = reference_path(
    |tau: f64| VolterraKernel::evaluate(&kernel, tau),
    mean_reverting_drift,
    state_dependent_diffusion,
    x0,
    dt,
    dw_slice,
  );

  let mut max_rel = 0.0_f64;
  for i in 0..n {
    let rel = (lift_path[i] - ref_path[i]).abs() / (1.0 + ref_path[i].abs());
    if rel > max_rel {
      max_rel = rel;
    }
  }
  assert!(
    max_rel > TIGHT_TOL,
    "expected a non-zero drift to break the {TIGHT_TOL:e} tolerance (max_rel={max_rel:e}) — \
     if this now passes, the drift-integration gap this test guards against may have vanished \
     for a reason worth understanding before treating it as good news"
  );
}

/// Zero drift and unit diffusion give $\mathrm{Var}(X_t) = \int_0^t K(u)^2\,du$, here
/// $c^2(1-e^{-2\beta t})/(2\beta)$. The 4% bound: $5\sqrt{2/N}$ at $N=50{,}000$ and a 0.35% bias.
#[test]
fn gaussian_case_matches_closed_form_variance() {
  let beta = 0.7_f64;
  let c = 1.0_f64;
  let n = 200_usize;
  let t_max = 1.0_f64;
  let num_paths = 50_000_usize;
  const REL_TOL: f64 = 0.04;

  let sde = VolterraSde::new(
    ExponentialKernel::new(beta, c),
    zero_drift as fn(f64, f64) -> f64,
    one_diffusion as fn(f64, f64) -> f64,
    n,
    Some(0.0),
    Some(t_max),
    Deterministic::new(2026),
  );

  let paths = sde.sample_par(num_paths);
  let terminal = paths.iter().map(|p| p[n - 1]).collect::<Vec<_>>();
  let mean = terminal.iter().sum::<f64>() / num_paths as f64;
  let sample_var =
    terminal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (num_paths as f64 - 1.0);

  let closed_form = c * c * (1.0 - (-2.0 * beta * t_max).exp()) / (2.0 * beta);
  let rel = (sample_var - closed_form).abs() / closed_form;
  assert!(
    rel < REL_TOL,
    "sample_var={sample_var} closed_form={closed_form} rel={rel} exceeds REL_TOL={REL_TOL}"
  );
}
