//! CGMYSV sample path generation (Algorithm 1).
//!
//! Generates discrete sample paths of the CGMYSV process by combining
//! exact CIR simulation (non-central $\chi^2$) with the truncated Rosiński
//! series representation of the time-changed standard CGMY process.
//!
//! Reference: Kim, Y. S. (2021), arXiv:2101.11001, Section 3.

use ndarray::{Array1, Array2};

use crate::distributions::exp::SimdExp;
use crate::distributions::gamma::SimdGamma;
use crate::distributions::normal::SimdNormal;
use crate::distributions::poisson::SimdPoisson;
use crate::distributions::uniform::SimdUniform;

use super::model::CgmysvParams;

/// CGMYSV sample path generator implementing Algorithm 1 from Kim (2021).
#[derive(Debug, Clone)]
pub struct CgmysvPathGen {
  pub params: CgmysvParams,
  /// Number of time steps ($M$).
  pub n_steps: usize,
  /// Truncation terms for the series representation ($J$).
  pub n_jumps: usize,
  /// Time horizon $T$ in years.
  pub t: f64,
}

impl CgmysvPathGen {
  /// Generate `n_paths` independent CGMYSV sample paths in parallel.
  ///
  /// Returns `(l_paths, v_paths)` each of shape `(n_paths, n_steps + 1)`.
  pub fn generate(&self, n_paths: usize) -> (Array2<f64>, Array2<f64>) {
    use rayon::prelude::*;

    let results: Vec<(Array1<f64>, Array1<f64>)> = (0..n_paths)
      .into_par_iter()
      .map(|_| self.sample_single())
      .collect();

    let m = self.n_steps;
    let mut l_paths = Array2::zeros((n_paths, m + 1));
    let mut v_paths = Array2::zeros((n_paths, m + 1));
    for (i, (l, v)) in results.into_iter().enumerate() {
      l_paths.row_mut(i).assign(&l);
      v_paths.row_mut(i).assign(&v);
    }

    (l_paths, v_paths)
  }

  /// Generate a single sample path of $(L, v)$.
  fn sample_single(&self) -> (Array1<f64>, Array1<f64>) {
    let p = &self.params;
    let m = self.n_steps;
    let j = self.n_jumps;
    let t = self.t;
    let dt = t / m as f64;

    let c_cir = 2.0 * p.kappa / ((1.0 - (-p.kappa * dt).exp()) * p.zeta * p.zeta);
    let df = 4.0 * p.kappa * p.eta / (p.zeta * p.zeta);
    let big_c = p.norm_const();

    // Internal SIMD-accelerated distributions (thread-local RNG)
    let uniform = SimdUniform::<f64>::new(0.0, 1.0);
    let exp1 = SimdExp::<f64>::new(1.0);
    let normal = SimdNormal::<f64>::new(0.0, 1.0);

    // Step 1: CIR variance path via non-central χ²
    let mut v = Array1::<f64>::zeros(m + 1);
    v[0] = p.v0;
    for i in 1..=m {
      let ncp = (2.0 * c_cir * v[i - 1] * (-p.kappa * dt).exp()).max(0.0);
      v[i] = (sample_noncentral_chi_sq(&normal, df, ncp) / (2.0 * c_cir)).max(0.0);
    }

    // Step 2: Generate jump components in bulk
    let mut gamma_arrivals = Array1::<f64>::zeros(j);
    let mut gamma_sum = 0.0_f64;
    for jj in 0..j {
      gamma_sum += exp1.sample_fast();
      gamma_arrivals[jj] = gamma_sum;
    }

    let mut u_vals = Array1::<f64>::zeros(j);
    uniform.fill_slice_fast(u_vals.as_slice_mut().unwrap());

    let mut e_vals = Array1::<f64>::zeros(j);
    for jj in 0..j {
      e_vals[jj] = exp1.sample_fast();
    }

    let v_signs: Array1<f64> = Array1::from_shape_fn(j, |_| {
      if uniform.sample_fast() <= 0.5 {
        p.lambda_plus
      } else {
        -p.lambda_minus
      }
    });

    let mut tau_raw = Array1::<f64>::zeros(j);
    uniform.fill_slice_fast(tau_raw.as_slice_mut().unwrap());
    let tau_vals = tau_raw * t;

    // c(τ_j) = C · v_{k-1} where (k-1)Δt < τ_j ≤ kΔt
    let c_tau: Array1<f64> = Array1::from_shape_fn(j, |jj| {
      let k = ((tau_vals[jj] / dt).ceil() as usize).max(1).min(m);
      big_c * v[k - 1]
    });

    // Precompute drift asymmetry ratio
    let asym = (p.lambda_plus.powf(p.alpha - 1.0) - p.lambda_minus.powf(p.alpha - 1.0))
      / ((1.0 - p.alpha)
        * (p.lambda_plus.powf(p.alpha - 2.0) + p.lambda_minus.powf(p.alpha - 2.0)));

    // Step 3: Build CGMYSV path
    let mut y = Array1::<f64>::zeros(m + 1);
    let mut l = Array1::<f64>::zeros(m + 1);
    let inv_alpha = -1.0 / p.alpha;

    for step in 1..=m {
      let t_lo = (step - 1) as f64 * dt;
      let t_hi = step as f64 * dt;
      let b_m = -v[step - 1] * asym;

      let mut jump_sum = 0.0;
      for jj in 0..j {
        if tau_vals[jj] > t_lo && tau_vals[jj] <= t_hi && c_tau[jj] > 1e-15 {
          let term1 =
            (p.alpha * gamma_arrivals[jj] / (2.0 * c_tau[jj] * t)).powf(inv_alpha);
          let term2 = e_vals[jj] * u_vals[jj].powf(1.0 / p.alpha) / v_signs[jj].abs();
          jump_sum += term1.min(term2) * v_signs[jj].signum();
        }
      }

      y[step] = y[step - 1] + jump_sum + b_m * dt;
      l[step] = y[step] + p.rho * v[step];
    }

    (l, v)
  }
}

/// Sample from a non-central $\chi^2$ distribution using internal distributions.
///
/// Uses the decomposition $\chi^2_\nu(\lambda) = (Z + \sqrt\lambda)^2 + \chi^2_{\nu-1}$
/// for $\nu \ge 1$, and Poisson mixture for $\nu < 1$.
fn sample_noncentral_chi_sq(normal: &SimdNormal<f64>, df: f64, ncp: f64) -> f64 {
  if ncp < 1e-10 {
    if df < 1e-10 {
      return 0.0;
    }
    return SimdGamma::<f64>::new(df / 2.0, 2.0).sample_fast();
  }

  if df >= 1.0 {
    let z = normal.sample_fast();
    let sq = (z + ncp.sqrt()).powi(2);
    let rem = df - 1.0;
    if rem > 1e-10 {
      sq + SimdGamma::<f64>::new(rem / 2.0, 2.0).sample_fast()
    } else {
      sq
    }
  } else {
    // 0 < df < 1: Poisson mixture
    let poi = SimdPoisson::<u32>::new(ncp / 2.0);
    let n = poi.sample_fast() as f64;
    let adj = df + 2.0 * n;
    if adj < 1e-10 {
      return 0.0;
    }
    SimdGamma::<f64>::new(adj / 2.0, 2.0).sample_fast()
  }
}
