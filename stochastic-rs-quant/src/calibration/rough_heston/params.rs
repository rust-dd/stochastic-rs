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

/// Where `cf` is validated to less than the model's box: the approximant to `ν = 0.4`.
fn validity(cf: &RoughHestonCf) -> &'static [(usize, f64)] {
  match cf {
    RoughHestonCf::Pade { .. } => &[(3, 0.4)],
    _ => &[],
  }
}

/// The fit's box under `cf`: the model's, cut at the selector's validity bounds.
pub(super) fn bounds(cf: &RoughHestonCf) -> Bounds {
  let mut bounds = MODEL;
  for &(j, most) in validity(cf) {
    bounds[j].1 = most;
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

/// The first parameter a fit left on a validity bound of `cf`: the selector's limit, where a
/// bound of the model's own box is a constrained optimum.
pub(super) fn held(params: RoughHestonParams, cf: &RoughHestonCf) -> Option<&'static str> {
  let values = Array1::from(params);
  let on_bound = |&&(j, most): &&(usize, f64)| most - values[j] <= 1e-9 * (most - MODEL[j].0);
  validity(cf).iter().find(on_bound).map(|&(j, _)| NAMES[j])
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
