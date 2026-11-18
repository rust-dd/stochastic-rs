//! The `volterra/` slice of the reproducibility guard, `../reproducibility_all_processes.rs`.

use stochastic_rs_stochastic::rough::kernel::RlKernel;
use stochastic_rs_stochastic::volterra::ExponentialKernel;
use stochastic_rs_stochastic::volterra::GaussianPolynomialVolatility;
use stochastic_rs_stochastic::volterra::Volterra;
use stochastic_rs_stochastic::volterra::VolterraKernelSpec;
use stochastic_rs_stochastic::volterra::VolterraSde;
use stochastic_rs_stochastic::volterra::VolterraSquareRoot;

use crate::common::N;
use crate::common::guard;

fn mean_reverting_drift(_t: f64, x: f64) -> f64 {
  0.3 * (0.5 - x)
}

fn const_diffusion(_t: f64, _x: f64) -> f64 {
  0.2
}

guard!(volterra_sde, "VolterraSde", |s| VolterraSde::new(
  ExponentialKernel::new(0.7, 1.0),
  mean_reverting_drift as fn(f64, f64) -> f64,
  const_diffusion as fn(f64, f64) -> f64,
  N,
  Some(0.1),
  Some(1.0),
  s
));

// `VolterraSquareRoot` is guarded where the Feller condition fails (2*kappa*theta = 0.02 against
// nu^2 = 0.25), so the truncation branch runs here too.
guard!(volterra_square_root, "VolterraSquareRoot", |s| {
  VolterraSquareRoot::new(
    RlKernel::<f64>::new(0.1, 24),
    0.5,
    0.02,
    0.5,
    N,
    0.02,
    Some(1.0),
    s,
  )
});

// The quintic parameterisation of arXiv:2212.10917 at its Figure-1 calibration, over an
// exponential kernel, which is an exact Ornstein-Uhlenbeck driver.
guard!(
  gaussian_polynomial_volatility,
  "GaussianPolynomialVolatility",
  |s| GaussianPolynomialVolatility::quintic(
    ExponentialKernel::new(1.5, 1.0),
    0.5907,
    1.0,
    0.2893,
    0.0549,
    N,
    Some(1.0),
    s
  )
);

guard!(volterra, "Volterra", |s| Volterra::new(
  VolterraKernelSpec::FractionalBM { h: 0.7 },
  N,
  Some(1.0),
  s
));
