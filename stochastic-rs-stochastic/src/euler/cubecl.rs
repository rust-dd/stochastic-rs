//! CubeCL device path of the Euler engine: one thread per path, the whole
//! recursion in the kernel, normals from a counter hash of
//! `(path, step, seed)` pushed through Box–Muller — the same generator as
//! the fGN device kernels. `f32` on the device (the portable GPU float),
//! widened on the way back.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use ndarray::Array2;
use ndarray::Array3;
use parking_lot::Mutex;

use super::EulerCoefficients;
use super::EulerKernel;
use super::EulerSpec;
use super::families::cube;
use super::families::cube_report;
use crate::device::DeviceError;
use crate::device::DeviceInfo;

type DeviceResult<T> = std::result::Result<T, DeviceError>;

const WG_SIZE: u32 = 256;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn euler_paths_kernel(
  out: &mut Array<f32>,
  params: &Array<f32>,
  incs: &Array<f32>,
  curve: &Array<f32>,
  family: u32,
  components: u32,
  noises: u32,
  x00: f32,
  x01: f32,
  x02: f32,
  x03: f32,
  dt: f32,
  sqrt_dt: f32,
  seed: u32,
  steps: u32,
  paths: u32,
  first_path: u32,
  increments: u32,
  has_curve: u32,
  jump_lambda: f32,
  has_jumps: u32,
) {
  let path = ABSOLUTE_POS as u32;
  if path < paths {
    let base = (path * steps) as usize;
    let plane = (paths * steps) as usize;
    let mut ct = 0.0f32;
    if has_curve != 0u32 {
      ct = curve[0];
    }
    let mut nj = 0.0f32;
    let mut u = 0.0f32;
    let mut u2 = 0.0f32;
    let mut s0 = x00;
    let mut s1 = x01;
    let mut s2 = x02;
    let mut s3 = x03;
    out[base] = report(family, 0u32, s0, s1, s2, s3, params, ct, nj, u, u2);
    if components > 1u32 {
      out[plane + base] = report(family, 1u32, s0, s1, s2, s3, params, ct, nj, u, u2);
    }
    if components > 2u32 {
      out[2usize * plane + base] = report(family, 2u32, s0, s1, s2, s3, params, ct, nj, u, u2);
    }
    if components > 3u32 {
      out[3usize * plane + base] = report(family, 3u32, s0, s1, s2, s3, params, ct, nj, u, u2);
    }
    for i in 1..steps {
      let g = (first_path + path) * steps + i;
      let mut d0 = normal(g, 0u32, seed) * sqrt_dt;
      let mut d1 = 0.0f32;
      let mut d2 = 0.0f32;
      let mut d3 = 0.0f32;
      if noises > 1u32 {
        d1 = normal(g, 1u32, seed) * sqrt_dt;
      }
      if noises > 2u32 {
        d2 = normal(g, 2u32, seed) * sqrt_dt;
      }
      if noises > 3u32 {
        d3 = normal(g, 3u32, seed) * sqrt_dt;
      }
      if increments != 0u32 {
        d0 = incs[(path * (steps - 1) + (i - 1)) as usize];
      }
      if has_curve != 0u32 {
        ct = curve[i as usize];
      }
      if has_jumps != 0u32 {
        nj = poisson(g, seed, jump_lambda * dt);
      }
      u = uniform(g ^ 2135587861u32, seed);
      u2 = uniform(g ^ 3266489917u32, seed);
      let n0 = step(
        family, 0u32, s0, s1, s2, s3, params, dt, ct, nj, u, u2, d0, d1, d2, d3,
      );
      let mut n1 = s1;
      let mut n2 = s2;
      let mut n3 = s3;
      if components > 1u32 {
        n1 = step(
          family, 1u32, s0, s1, s2, s3, params, dt, ct, nj, u, u2, d0, d1, d2, d3,
        );
      }
      if components > 2u32 {
        n2 = step(
          family, 2u32, s0, s1, s2, s3, params, dt, ct, nj, u, u2, d0, d1, d2, d3,
        );
      }
      if components > 3u32 {
        n3 = step(
          family, 3u32, s0, s1, s2, s3, params, dt, ct, nj, u, u2, d0, d1, d2, d3,
        );
      }
      s0 = n0;
      s1 = n1;
      s2 = n2;
      s3 = n3;
      out[base + i as usize] = report(family, 0u32, s0, s1, s2, s3, params, ct, nj, u, u2);
      if components > 1u32 {
        out[plane + base + i as usize] =
          report(family, 1u32, s0, s1, s2, s3, params, ct, nj, u, u2);
      }
      if components > 2u32 {
        out[2usize * plane + base + i as usize] =
          report(family, 2u32, s0, s1, s2, s3, params, ct, nj, u, u2);
      }
      if components > 3u32 {
        out[3usize * plane + base + i as usize] =
          report(family, 3u32, s0, s1, s2, s3, params, ct, nj, u, u2);
      }
    }
  }
}

/// One uniform in `[0, 1)` from a counter, by the same Murmur3-style
/// finalizer the normals use.
#[cube]
fn uniform(g: u32, seed: u32) -> f32 {
  let mut h = g ^ (seed * 2654435761u32);
  h ^= h >> 16;
  h *= 2246822519u32;
  h ^= h >> 13;
  h *= 3266489917u32;
  h ^= h >> 16;
  f32::cast_from(h) * 2.3283064e-10f32
}

/// A Poisson draw with mean `mean`, by Knuth's product of uniforms: the
/// uniforms come from a hash stream of the step's counter that no noise
/// component uses, so a family that takes jumps draws the same Gaussian
/// shocks it would without them. The loop is bounded, which for a mean small
/// enough to model a jump process it never reaches.
#[cube]
fn poisson(g: u32, seed: u32, mean: f32) -> f32 {
  let ell = Exp::exp(0.0f32 - mean);
  let mut prod = 1.0f32;
  let mut cnt = 0.0f32;
  let mut running: u32 = 1u32;
  for j in 0..64u32 {
    if running != 0u32 {
      let mut h = (g ^ (2166136261u32 + j * 16777619u32)) ^ (seed * 374761393u32);
      h ^= h >> 16;
      h *= 2246822519u32;
      h ^= h >> 13;
      h *= 3266489917u32;
      h ^= h >> 16;
      prod *= f32::cast_from(h) * 2.3283064e-10f32;
      if prod <= ell {
        running = 0u32;
      } else {
        cnt += 1.0f32;
      }
    }
  }
  cnt
}

/// One standard normal for noise component `k` of counter `g`, from two
/// decorrelated uniforms via integer hashing (Murmur3-style finalizer).
/// Component `0` hashes the counter itself and every further one xors in a
/// constant of its own, so a single-noise family draws exactly the stream it
/// drew before the engine learned about systems. The salt is xored rather
/// than multiplied in because WGSL constant-folds the multiplication and
/// rejects it as an overflow.
#[cube]
fn normal(g: u32, k: u32, seed: u32) -> f32 {
  let mut gk = g;
  if k == 1u32 {
    gk = g ^ 2654435769u32;
  }
  if k == 2u32 {
    gk = g ^ 2246822519u32;
  }
  if k == 3u32 {
    gk = g ^ 3266489917u32;
  }
  let mut a = (gk * 2u32) ^ (seed * 2654435761u32);
  a ^= a >> 16;
  a *= 2246822519u32;
  a ^= a >> 13;
  a *= 3266489917u32;
  a ^= a >> 16;
  let mut b = (gk * 2u32 + 1u32) ^ (seed * 668265263u32);
  b ^= b >> 16;
  b *= 2246822519u32;
  b ^= b >> 13;
  b *= 3266489917u32;
  b ^= b >> 16;
  let inv = 2.3283064e-10f32;
  let u1 = f32::cast_from(a) * inv * 0.999998f32 + 1.0e-6f32;
  let u2 = f32::cast_from(b) * inv;
  Sqrt::sqrt(-2.0f32 * Log::ln(u1)) * Cos::cos(core::f32::consts::TAU * u2)
}

/// Dispatches to the family's generated step. The formulas live in the
/// declarations in [`super::families`]; what stands here is the parameter
/// order each family reads from the buffer, which the compiler checks by
/// arity.
#[cube]
#[allow(clippy::too_many_arguments)]
fn step(
  family: u32,
  component: u32,
  x0: f32,
  x1: f32,
  x2: f32,
  x3: f32,
  params: &Array<f32>,
  dt: f32,
  ct: f32,
  nj: f32,
  u: f32,
  u2: f32,
  dz0: f32,
  dz1: f32,
  dz2: f32,
  dz3: f32,
) -> f32 {
  let mut stepped = x0;
  if family == 0u32 {
    stepped = cube::GeometricBrownian(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 1u32 {
    stepped = cube::OrnsteinUhlenbeck(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 2u32 {
    stepped = cube::SquareRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 6u32 {
    stepped = cube::Jacobi(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 9u32 {
    stepped = cube::Logistic(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 12u32 {
    stepped = cube::RadialOrnsteinUhlenbeck(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 17u32 {
    stepped = cube::AitSahalia(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 18u32 {
    stepped = cube::Gompertz(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 19u32 {
    stepped = cube::Kimura(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 20u32 {
    stepped = cube::Quadratic(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 21u32 {
    stepped = cube::Pearson(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 22u32 {
    stepped = cube::Verhulst(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 23u32 {
    stepped = cube::VerhulstClamped(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 24u32 {
    stepped = cube::FellerLogistic(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 25u32 {
    stepped = cube::FellerLogisticReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 26u32 {
    stepped = cube::SquaredBesselState(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 27u32 {
    stepped = cube::SquaredBesselStateReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 28u32 {
    stepped = cube::BesselFromSquared(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 29u32 {
    stepped = cube::BesselFromSquaredReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 30u32 {
    stepped = cube::HyperbolicDiffusion(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 31u32 {
    stepped = cube::NonLinear(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 32u32 {
    stepped = cube::Displaced(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 33u32 {
    stepped = cube::TanhOrnsteinUhlenbeck(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 34u32 {
    stepped = cube::BoundedCorrelation(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 35u32 {
    stepped = cube::Heston(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 36u32 {
    stepped = cube::HestonReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 37u32 {
    stepped = cube::Sabr(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 38u32 {
    stepped = cube::Bergomi(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 39u32 {
    stepped = cube::TwoScaleOrnsteinUhlenbeck(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 40u32 {
    stepped = cube::LogHeston(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 41u32 {
    stepped = cube::LogHestonReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 42u32 {
    stepped = cube::DoubleHeston(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 43u32 {
    stepped = cube::DoubleHestonReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 44u32 {
    stepped = cube::StochasticCorrelationHeston(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 45u32 {
    stepped = cube::HullWhite(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 46u32 {
    stepped = cube::CurveDrift(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 47u32 {
    stepped = cube::LogMeanReverting(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 48u32 {
    stepped = cube::ShiftedSquareRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 49u32 {
    stepped = cube::ShiftedSquareRootMirrored(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 50u32 {
    stepped = cube::TimeVaryingGeometricBrownian(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 51u32 {
    stepped = cube::CorrelatedBrownian(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 52u32 {
    stepped = cube::BrownianBridge(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 53u32 {
    stepped = cube::TwoFactorHullWhite(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 54u32 {
    stepped = cube::TwoFactorSquareRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 55u32 {
    stepped = cube::DuffieKan(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 56u32 {
    stepped = cube::TwoAssetHeston(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 57u32 {
    stepped = cube::TwoAssetHestonReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 58u32 {
    stepped = cube::MertonJumpLog(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 59u32 {
    stepped = cube::BatesJump(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 60u32 {
    stepped = cube::BatesJumpReflected(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 61u32 {
    stepped = cube::AndersenQe(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 62u32 {
    stepped = cube::CountingProcess(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 63u32 {
    stepped = cube::InverseGaussianSubordinator(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 64u32 {
    stepped = cube::NormalInverseGaussian(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 65u32 {
    stepped = cube::StableSubordinator(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 16u32 {
    stepped = cube::FellerRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 15u32 {
    stepped = cube::ModifiedSquareRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 14u32 {
    stepped = cube::Hyperbolic(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 13u32 {
    stepped = cube::LinearSde(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 11u32 {
    stepped = cube::LogGeometric(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 10u32 {
    stepped = cube::ThreeHalf(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 8u32 {
    stepped = cube::Ckls(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 7u32 {
    stepped = cube::ConstantElasticity(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 5u32 {
    stepped = cube::MirroredSquareRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 4u32 {
    stepped = cube::ReflectedSquareRoot(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  if family == 3u32 {
    stepped = cube::Additive(
      component, x0, x1, x2, x3, params, dt, ct, nj, u, u2, dz0, dz1, dz2, dz3,
    );
  }
  stepped
}

/// Dispatches to the family's generated report.
#[cube]
fn report(
  family: u32,
  component: u32,
  x0: f32,
  x1: f32,
  x2: f32,
  x3: f32,
  params: &Array<f32>,
  ct: f32,
  nj: f32,
  u: f32,
  u2: f32,
) -> f32 {
  let mut reported = x0;
  if family == 0u32 {
    reported = cube_report::GeometricBrownian(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 1u32 {
    reported = cube_report::OrnsteinUhlenbeck(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 2u32 {
    reported = cube_report::SquareRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 6u32 {
    reported = cube_report::Jacobi(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 9u32 {
    reported = cube_report::Logistic(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 12u32 {
    reported =
      cube_report::RadialOrnsteinUhlenbeck(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 17u32 {
    reported = cube_report::AitSahalia(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 18u32 {
    reported = cube_report::Gompertz(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 19u32 {
    reported = cube_report::Kimura(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 20u32 {
    reported = cube_report::Quadratic(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 21u32 {
    reported = cube_report::Pearson(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 22u32 {
    reported = cube_report::Verhulst(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 23u32 {
    reported = cube_report::VerhulstClamped(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 24u32 {
    reported = cube_report::FellerLogistic(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 25u32 {
    reported =
      cube_report::FellerLogisticReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 26u32 {
    reported = cube_report::SquaredBesselState(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 27u32 {
    reported =
      cube_report::SquaredBesselStateReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 28u32 {
    reported = cube_report::BesselFromSquared(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 29u32 {
    reported =
      cube_report::BesselFromSquaredReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 30u32 {
    reported = cube_report::HyperbolicDiffusion(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 31u32 {
    reported = cube_report::NonLinear(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 32u32 {
    reported = cube_report::Displaced(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 33u32 {
    reported = cube_report::TanhOrnsteinUhlenbeck(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 34u32 {
    reported = cube_report::BoundedCorrelation(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 35u32 {
    reported = cube_report::Heston(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 36u32 {
    reported = cube_report::HestonReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 37u32 {
    reported = cube_report::Sabr(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 38u32 {
    reported = cube_report::Bergomi(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 39u32 {
    reported =
      cube_report::TwoScaleOrnsteinUhlenbeck(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 40u32 {
    reported = cube_report::LogHeston(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 41u32 {
    reported = cube_report::LogHestonReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 42u32 {
    reported = cube_report::DoubleHeston(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 43u32 {
    reported = cube_report::DoubleHestonReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 44u32 {
    reported =
      cube_report::StochasticCorrelationHeston(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 45u32 {
    reported = cube_report::HullWhite(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 46u32 {
    reported = cube_report::CurveDrift(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 47u32 {
    reported = cube_report::LogMeanReverting(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 48u32 {
    reported = cube_report::ShiftedSquareRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 49u32 {
    reported =
      cube_report::ShiftedSquareRootMirrored(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 50u32 {
    reported =
      cube_report::TimeVaryingGeometricBrownian(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 51u32 {
    reported = cube_report::CorrelatedBrownian(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 52u32 {
    reported = cube_report::BrownianBridge(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 53u32 {
    reported = cube_report::TwoFactorHullWhite(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 54u32 {
    reported = cube_report::TwoFactorSquareRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 55u32 {
    reported = cube_report::DuffieKan(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 56u32 {
    reported = cube_report::TwoAssetHeston(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 57u32 {
    reported =
      cube_report::TwoAssetHestonReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 58u32 {
    reported = cube_report::MertonJumpLog(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 59u32 {
    reported = cube_report::BatesJump(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 60u32 {
    reported = cube_report::BatesJumpReflected(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 61u32 {
    reported = cube_report::AndersenQe(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 62u32 {
    reported = cube_report::CountingProcess(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 63u32 {
    reported =
      cube_report::InverseGaussianSubordinator(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 64u32 {
    reported = cube_report::NormalInverseGaussian(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 65u32 {
    reported = cube_report::StableSubordinator(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 16u32 {
    reported = cube_report::FellerRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 15u32 {
    reported = cube_report::ModifiedSquareRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 14u32 {
    reported = cube_report::Hyperbolic(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 13u32 {
    reported = cube_report::LinearSde(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 11u32 {
    reported = cube_report::LogGeometric(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 10u32 {
    reported = cube_report::ThreeHalf(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 8u32 {
    reported = cube_report::Ckls(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 7u32 {
    reported = cube_report::ConstantElasticity(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 5u32 {
    reported = cube_report::MirroredSquareRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 4u32 {
    reported = cube_report::ReflectedSquareRoot(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  if family == 3u32 {
    reported = cube_report::Additive(component, x0, x1, x2, x3, params, ct, nj, u, u2);
  }
  reported
}

/// A CubeCL runtime this crate can open, with its own cached compute client.
/// One implementor per `cubecl-*` feature, so a build with both reaches both
/// devices; the kernels themselves are runtime-agnostic.
pub trait CubeclRuntime: Copy + Default + Send + Sync + 'static {
  /// The CubeCL runtime this opens.
  type Rt: cubecl::Runtime;

  /// What [`DeviceInfo::backend`] reports.
  const BACKEND: &'static str;

  /// The runtime's device at `ordinal`.
  fn device(ordinal: usize) -> <Self::Rt as cubecl::Runtime>::Device;

  /// The cached client for `ordinal`, opened on first use.
  fn client(ordinal: usize) -> DeviceResult<ComputeClient<Self::Rt>>;
}

/// The client cache of one runtime: CubeCL clients are cheap to clone and
/// expensive to open, and switching ordinal re-opens.
pub(crate) struct Context<Rt: cubecl::Runtime> {
  ordinal: usize,
  client: ComputeClient<Rt>,
}

// SAFETY: the client is only ever handed out as a clone under the mutex, and
// CubeCL's own client is internally synchronised.
unsafe impl<Rt: cubecl::Runtime> Send for Context<Rt> {}

/// The cached client for `ordinal`, re-opening when the ordinal changes.
/// CubeCL panics rather than erroring when no device exists, so the opening
/// is caught and reported as a [`DeviceError`].
pub(crate) fn open<Rt: cubecl::Runtime>(
  slot: &Mutex<Option<Context<Rt>>>,
  ordinal: usize,
  device: fn(usize) -> Rt::Device,
) -> DeviceResult<ComputeClient<Rt>> {
  let mut guard = slot.lock();
  if !guard.as_ref().is_some_and(|c| c.ordinal == ordinal) {
    *guard = None;
    // The device is built inside the closure: a `Runtime::Device` reference
    // is not `RefUnwindSafe`, a `usize` and a fn pointer are.
    match std::panic::catch_unwind(|| Rt::client(&device(ordinal))) {
      Ok(client) => *guard = Some(Context { ordinal, client }),
      Err(payload) => return Err(DeviceError::Unavailable(crate::device::panic_text(payload))),
    }
  }
  Ok(guard.as_ref().expect("initialised").client.clone())
}

/// The client cache of CubeCL's CUDA runtime. The tag it belongs to is
/// [`crate::device::CudaRuntime`], so a handle names the runtime in its type.
#[cfg(feature = "cubecl-cuda")]
mod cuda_rt {
  use super::*;

  static CONTEXT: Mutex<Option<Context<cubecl_cuda::CudaRuntime>>> = Mutex::new(None);

  impl CubeclRuntime for crate::device::CudaRuntime {
    type Rt = cubecl_cuda::CudaRuntime;

    const BACKEND: &'static str = "CubeclCuda";

    fn device(ordinal: usize) -> cubecl_cuda::CudaDevice {
      cubecl_cuda::CudaDevice { index: ordinal }
    }

    fn client(ordinal: usize) -> DeviceResult<ComputeClient<Self::Rt>> {
      open(&CONTEXT, ordinal, Self::device)
    }
  }
}

/// The client cache of CubeCL's wgpu runtime — Metal on macOS, Vulkan on
/// Linux, WebGPU on the web. `ordinal` `0` is the default adapter, `n > 0` the
/// n-th discrete GPU.
#[cfg(feature = "cubecl-wgpu")]
mod wgpu_rt {
  use super::*;

  static CONTEXT: Mutex<Option<Context<cubecl_wgpu::WgpuRuntime>>> = Mutex::new(None);

  impl CubeclRuntime for crate::device::WgpuRuntime {
    type Rt = cubecl_wgpu::WgpuRuntime;

    const BACKEND: &'static str = "CubeclWgpu";

    fn device(ordinal: usize) -> cubecl_wgpu::WgpuDevice {
      if ordinal == 0 {
        cubecl_wgpu::WgpuDevice::default()
      } else {
        cubecl_wgpu::WgpuDevice::DiscreteGpu(ordinal)
      }
    }

    fn client(ordinal: usize) -> DeviceResult<ComputeClient<Self::Rt>> {
      open(&CONTEXT, ordinal, Self::device)
    }
  }
}

/// The runtime's device at `ordinal`, or why it cannot be used.
pub fn probe<C: CubeclRuntime>(ordinal: usize) -> DeviceResult<DeviceInfo> {
  let cl = C::client(ordinal)?;
  Ok(DeviceInfo::new(
    C::BACKEND,
    <C::Rt as cubecl::Runtime>::name(&cl).to_string(),
    &["f32"],
    Some(ordinal),
  ))
}

/// Splits a 1-D cube count into a 2-D grid so no dimension exceeds WebGPU's
/// 65535 per-dimension limit.
fn count_2d(cubes: u32) -> CubeCount {
  if cubes <= 65535 {
    CubeCount::Static(cubes.max(1), 1, 1)
  } else {
    let mut x = cubes;
    let mut y = 1u32;
    while x > 32768 {
      x = x.div_ceil(2);
      y *= 2;
    }
    CubeCount::Static(x, y, 1)
  }
}

#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
impl<R: CubeclRuntime> EulerKernel<f32> for crate::device::Cubecl<R> {
  fn euler_kernel<P: EulerCoefficients<f32>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> DeviceResult<Array2<f32>> {
    let planes = device_paths::<R>(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      process.fgn_spec(),
      process.curve().as_deref().unwrap_or(&[]),
      process.jump_intensity(),
    )?;
    Ok(planes.index_axis_move(ndarray::Axis(0), 0))
  }

  /// A system's launch: the same kernel, its state slots filled from the
  /// process's own initial state and every component's plane returned.
  fn euler_system_kernel<const D: usize, P: super::EulerSystem<f32, D>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> DeviceResult<Array3<f32>> {
    let spec = process.euler_spec();
    super::check_arity(&spec, D);
    let slots = process.initial_state();
    device_paths::<R>(
      self.ordinal,
      spec,
      slots,
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      None,
      process.curve().as_deref().unwrap_or(&[]),
      process.jump_intensity(),
    )
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget
  }
}

/// The kernel launch for an explicit specification.
#[allow(clippy::too_many_arguments)]
fn device_paths<C: CubeclRuntime>(
  ordinal: usize,
  spec: EulerSpec<f32>,
  x0: [f32; 4],
  n: usize,
  dt: f32,
  first: usize,
  m: usize,
  seed: u64,
  fgn: Option<crate::euler::FgnSpec<'_, f32>>,
  curve: &[f32],
  jump_lambda: Option<f32>,
) -> DeviceResult<Array3<f32>> {
  {
    let (family, params) = spec.encode();
    let arity = super::families::Family::from_code(family).expect("a declared family");
    let (components, noises) = (arity.components(), arity.noises());
    if n == 0 || m == 0 {
      return Ok(Array3::<f32>::zeros((components, m, n)));
    }
    let params32: Vec<f32> = params.to_vec();
    let dt = dt as f64;
    let total = components * m * n;
    let cl = &C::client(ordinal)?;
    let data: Vec<f32> = {
      let params_h = cl.create_from_slice(f32::as_bytes(&params32));
      let out_h = cl.empty(total * 4);
      // Every declared buffer is bound; an unused increment slot gets one
      // float. A fractional process has its increments produced on this same
      // runtime and read from the handle they were written to.
      let (incs_h, incs_len) = match fgn.as_ref() {
        Some(spec) => {
          let (handle, out_size) = crate::noise::fgn::cubecl::backend::sample_cubecl_f32_handle::<C>(
            spec.sqrt_eigenvalues,
            spec.n,
            m,
            spec.offset,
            spec.hurst,
            spec.t,
            first,
            seed as u32,
            ordinal,
          )?;
          (handle, m * out_size)
        }
        None => (cl.empty(4), 1),
      };
      // Every declared buffer is bound; an unused curve slot gets one float.
      let (curve_h, curve_len) = if curve.is_empty() {
        (cl.empty(4), 1)
      } else {
        (cl.create_from_slice(f32::as_bytes(curve)), curve.len())
      };
      unsafe {
        euler_paths_kernel::launch::<C::Rt>(
          cl,
          count_2d((m as u32).div_ceil(WG_SIZE)),
          CubeDim::new_1d(WG_SIZE),
          ArrayArg::from_raw_parts::<f32>(&out_h, total, 1),
          ArrayArg::from_raw_parts::<f32>(&params_h, crate::euler::PARAM_SLOTS, 1),
          ArrayArg::from_raw_parts::<f32>(&incs_h, incs_len, 1),
          ArrayArg::from_raw_parts::<f32>(&curve_h, curve_len, 1),
          ScalarArg::new(family),
          ScalarArg::new(components as u32),
          ScalarArg::new(noises as u32),
          ScalarArg::new(x0[0]),
          ScalarArg::new(x0[1]),
          ScalarArg::new(x0[2]),
          ScalarArg::new(x0[3]),
          ScalarArg::new(dt as f32),
          ScalarArg::new(dt.sqrt() as f32),
          ScalarArg::new((seed ^ (seed >> 32)) as u32),
          ScalarArg::new(n as u32),
          ScalarArg::new(m as u32),
          ScalarArg::new(first as u32),
          ScalarArg::new(u32::from(fgn.is_some())),
          ScalarArg::new(u32::from(!curve.is_empty())),
          ScalarArg::new(jump_lambda.unwrap_or(0.0)),
          ScalarArg::new(u32::from(jump_lambda.is_some())),
        )
        .map_err(|e| DeviceError::Launch(format!("euler_paths launch: {e:?}")))?;
      }
      let bytes = cl.read_one(out_h.clone());
      f32::from_bytes(&bytes).to_vec()
    };
    Ok(
      Array3::from_shape_vec((components, m, n), data)
        .expect("the kernel returns components * m * n values"),
    )
  }
}
