//! h-functions, quantiles, cdfs, densities, Kendall's τ, Archimedean generators and nested Archimedean copulas at the
//! ends of every admitted parameter range: finite and in range, and within `1e-12` of mpmath values wherever those are normal.

mod archimedean;
mod conditional;
mod densities;
mod kendall_tau;

use ndarray::Array1;
use ndarray::Axis;
use ndarray::stack;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::traits::BivariateExt;

/// The edges, the extreme normal and subnormal values, and the extreme uniform draws `2⁻⁵³` and `1 − 2⁻⁵³`.
const GRID: [f64; 9] = [
  0.0,
  5e-324,
  1e-300,
  1e-15,
  f64::EPSILON / 2.0,
  0.5,
  1.0 - 1e-15,
  1.0 - f64::EPSILON / 2.0,
  1.0,
];

fn with_theta<C: BivariateExt>(mut copula: C, theta: f64) -> C {
  copula.set_theta(theta);
  copula
}

fn build(family: &str, p: &[f64]) -> Box<dyn BivariateExt> {
  match family {
    "galambos" => Box::new(with_theta(Galambos::default(), p[0])),
    "clayton" => Box::new(Clayton::new(p[0])),
    "plackett" => Box::new(Plackett::new(p[0])),
    "amh" => Box::new(Amh::new(p[0])),
    "frank" => Box::new(Frank::new(p[0])),
    "gumbel" => Box::new(Gumbel::new(p[0])),
    "joe" => Box::new(Joe::new(p[0])),
    "husler_reiss" => Box::new(HuslerReiss::new(p[0])),
    "bb1" => Box::new(Bb1::new(p[0], p[1])),
    "bb7" => Box::new(Bb7::new(p[0], p[1])),
    "t" => Box::new(TCopula::new(p[0], p[1])),
    "gaussian" => Box::new(GaussianCopula::new(p[0])),
    "fgm" => Box::new(Fgm::new(p[0])),
    "marshall_olkin" => Box::new(MarshallOlkin::new(p[0], p[1])),
    "independence" => Box::new(Independence::new()),
    other => panic!("no family {other}"),
  }
}

/// `kind` over the whole grid in one call, `a` the row and `b` the column: `h(a | b)`, `h⁻¹(a | b)`, `C(a, b)`, or
/// `ln c(a, b)` on the open grid, without its edges.
fn on_grid(copula: &dyn BivariateExt, kind: &str) -> Array1<f64> {
  let grid: &[f64] = if kind == "log_pdf" {
    &GRID[1..8]
  } else {
    &GRID
  };
  let a = Array1::from_iter(grid.iter().flat_map(|&a| grid.iter().map(move |_| a)));
  let b = Array1::from_iter(grid.iter().flat_map(|_| grid.iter().copied()));
  match kind {
    "h" => copula.partial_derivative(stack![Axis(1), a, b].view()),
    "q" => copula.percent_point(a.view(), b.view()),
    "cdf" => copula.cdf(stack![Axis(1), a, b].view()),
    _ => copula.log_pdf(stack![Axis(1), a, b].view()),
  }
  .unwrap_or_else(|e| panic!("{e}"))
}

/// A normal truth to `1e-12` relative; one below `f64::MIN_POSITIVE` to within it, so 0 passes.
fn agrees(got: f64, want: f64) -> bool {
  if want >= f64::MIN_POSITIVE {
    (got - want).abs() <= 1e-12 * want
  } else {
    (got - want).abs() <= f64::MIN_POSITIVE
  }
}

/// An infinite truth exactly, a finite one as `agrees` holds it.
fn matches(got: f64, want: f64) -> bool {
  if want.is_infinite() {
    got == want
  } else {
    agrees(got, want)
  }
}
