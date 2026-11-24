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

/// The box of `(V₀, κ, θ, ν, ρ, H)` under `cf`: `H` inside `(0, ½)`, `κ` above 0, and `ν` to 0.4
/// under the approximant, the range it is validated on, to 2 under the two solves.
pub(super) fn bounds(cf: &RoughHestonCf) -> Bounds {
  let most = match cf {
    RoughHestonCf::Pade { .. } => 0.4,
    _ => 2.0,
  };
  [
    (0.005, 0.25),
    (0.01, 10.0),
    (0.001, 1.0),
    (0.01, most),
    (-0.9999, 0.9999),
    (0.005, 0.495),
  ]
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

/// The first parameter a fit left on a bound of its box.
pub(super) fn held(params: RoughHestonParams, bounds: &Bounds) -> Option<&'static str> {
  let values = Array1::from(params);
  let on_bound =
    |(&value, &(lo, hi)): (&f64, &(f64, f64))| (value - lo).min(hi - value) <= 1e-9 * (hi - lo);
  let place = values.iter().zip(bounds).position(on_bound);
  place.map(|j| NAMES[j])
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
