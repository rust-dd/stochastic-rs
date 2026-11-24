use ndarray::Array1;
use ndarray::ArrayView1;

use super::super::periodic_map;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;
use crate::pricing::fourier::RoughHestonCf;
use crate::pricing::fourier::RoughHestonParams;

pub(super) type Bounds = [(f64, f64); 6];

pub(super) const NAMES: [&str; 6] = ["v0", "kappa", "theta", "nu", "rho", "hurst"];

/// The model's own box of `(V₀, κ, θ, ν, ρ, H)`: `H` inside `(0, ½)`, `κ` above 0.
const MODEL: Bounds = [
  (0.005, 0.25),
  (0.01, 10.0),
  (0.001, 1.0),
  (0.01, 2.0),
  (-0.9999, 0.9999),
  (0.005, 0.495),
];

/// `(coordinate, lowest, highest)` where `cf` is validated on less than the model's box: the
/// approximant to `ν = 0.4`, and in `ρ` while it is neither NaN nor 1e-6 under a call's intrinsic.
fn validity(cf: &RoughHestonCf) -> &'static [(usize, f64, f64)] {
  match cf {
    RoughHestonCf::Pade { order: 3 } => &[(3, 0.01, 0.4), (4, -0.99, 0.85)],
    RoughHestonCf::Pade { .. } => &[(3, 0.01, 0.4), (4, -0.85, 0.75)],
    RoughHestonCf::Adams { .. } | RoughHestonCf::Lifted { .. } => &[],
  }
}

/// The fit's box under `cf`: the model's, cut at the selector's validity bounds.
pub(super) fn bounds(cf: &RoughHestonCf) -> Bounds {
  let mut bounds = MODEL;
  for &(j, least, most) in validity(cf) {
    bounds[j] = (bounds[j].0.max(least), bounds[j].1.min(most));
  }
  bounds
}

/// A raw point folded into `bounds`, coordinate by coordinate.
pub(super) fn folded(x: ArrayView1<f64>, bounds: &Bounds) -> RoughHestonParams {
  let places = x.iter().zip(bounds);
  let values = places.map(|(&x, &(lo, hi))| periodic_map(x, lo, hi));
  RoughHestonParams::from(values.collect::<Array1<f64>>())
}

/// Each raw coordinate's room before its folded value meets a bound.
pub(super) fn room_in(bounds: Bounds) -> impl Fn(ArrayView1<f64>) -> Room + 'static {
  move |x| {
    let point = Array1::from(folded(x, &bounds));
    room(&point, &bounds, periodic_slopes(x, &bounds).view())
  }
}

/// The first parameter a fit left on a validity bound of `cf` inside the model's box: the
/// selector's limit, where a bound of the model's own box is a constrained optimum.
pub(super) fn held(params: RoughHestonParams, cf: &RoughHestonCf) -> Option<&'static str> {
  let values = Array1::from(params);
  let on_bound = |&&(j, least, most): &&(usize, f64, f64)| {
    let (lo, hi) = MODEL[j];
    let near = |bound: f64| (values[j] - bound).abs() <= 1e-9 * (hi - lo);
    (least > lo && near(least)) || (most < hi && near(most))
  };
  validity(cf).iter().find(on_bound).map(|&(j, ..)| NAMES[j])
}

impl From<RoughHestonParams> for Array1<f64> {
  fn from(p: RoughHestonParams) -> Self {
    Array1::from_vec(vec![p.v0, p.kappa, p.theta, p.nu, p.rho, p.hurst])
  }
}

impl From<Array1<f64>> for RoughHestonParams {
  fn from(v: Array1<f64>) -> Self {
    RoughHestonParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      nu: v[3],
      rho: v[4],
      hurst: v[5],
    }
  }
}
