//! [`fit_l1`] keeps a kernel's nodes and refits the weights of $K(t)\approx\sum_l w_l e^{-x_l t}$
//! in $L^1$ by iteratively reweighted least squares: the quadrature fit for paths, this for prices.
//!
//! Reference: Bayer & Breneis (2023), arXiv:2309.07023, the weak error follows the $L^1$ error

use ndarray::Array1;
use ndarray::Array2;

use super::kernel::SumOfExponentials;
use super::kernel::VolterraKernel;
use crate::traits::FloatExt;

/// Number of IRLS iterations: by five the weight change is below the fit error itself.
const IRLS_ITERATIONS: usize = 5;

/// Ridge parameter, relative to the mean diagonal of the normal matrix: neighbouring nodes give
/// near-collinear columns, and a bare Cholesky can fail on the ill-conditioned normal equations.
const RIDGE: f64 = 1e-12;

/// Floor on `|r_i|` in the IRLS observation weight, so an exactly-zero
/// residual cannot produce an infinite weight.
const RESIDUAL_FLOOR: f64 = 1e-300;

/// Refit `kernel`'s weights in $L^1$ on `grid` log-spaced points of `[t_min, t_max]`, unchanged
/// if the normal equations fail; panics unless `0 < t_min < t_max` and `grid >= degree`.
#[must_use]
pub fn fit_l1<T: FloatExt, K: VolterraKernel<T>>(
  kernel: &K,
  t_min: T,
  t_max: T,
  grid: usize,
) -> SumOfExponentials<T> {
  let lo = t_min.to_f64().unwrap_or(f64::NAN);
  let hi = t_max.to_f64().unwrap_or(f64::NAN);
  assert!(lo > 0.0, "t_min must be strictly positive (got {lo})");
  assert!(hi > lo, "t_max must exceed t_min (got {lo}..{hi})");
  let n_prime = kernel.degree();
  assert!(
    grid >= n_prime,
    "grid ({grid}) must be at least the kernel degree ({n_prime}) or the fit is underdetermined"
  );

  let nodes = kernel
    .nodes()
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();

  let (t, quad) = log_grid(lo, hi, grid);
  let target = t
    .iter()
    .map(|&ti| {
      kernel
        .evaluate(T::from_f64_fast(ti))
        .to_f64()
        .unwrap_or(f64::NAN)
    })
    .collect::<Vec<f64>>();

  let mut design = Array2::<f64>::zeros((grid, n_prime));
  for (i, &ti) in t.iter().enumerate() {
    for (l, &xl) in nodes.iter().enumerate() {
      design[[i, l]] = (-xl * ti).exp();
    }
  }

  // Start from the incoming weights, so the first solve refines a good fit
  // rather than searching from nothing.
  let mut weights = kernel
    .weights()
    .iter()
    .map(|w| w.to_f64().unwrap_or(f64::NAN))
    .collect::<Vec<f64>>();

  for _ in 0..IRLS_ITERATIONS {
    let obs = (0..grid)
      .map(|i| {
        let fitted: f64 = (0..n_prime).map(|l| design[[i, l]] * weights[l]).sum();
        let residual = (target[i] - fitted).abs().max(RESIDUAL_FLOOR);
        quad[i] / residual
      })
      .collect::<Vec<f64>>();

    match weighted_least_squares(&design, &target, &obs, n_prime) {
      Some(next) => weights = next,
      // A failed factorisation adds nothing; the previous iteration's weights are still a valid
      // fit, so stop.
      None => break,
    }
  }

  let refit = Array1::from_iter(weights.into_iter().map(T::from_f64_fast));
  SumOfExponentials::new(kernel.nodes().clone(), refit)
}

/// Total $L^1$ error $\int_{t_{\min}}^{t_{\max}} |K(t) - \sum_l w_l e^{-x_l t}|\,dt$ of `approx`
/// against `truth`'s exact kernel on [`fit_l1`]'s grid; panics unless `0 < t_min < t_max`.
#[must_use]
pub fn l1_error<T: FloatExt, K: VolterraKernel<T>, A: VolterraKernel<T>>(
  truth: &K,
  approx: &A,
  t_min: T,
  t_max: T,
  grid: usize,
) -> T {
  let lo = t_min.to_f64().unwrap_or(f64::NAN);
  let hi = t_max.to_f64().unwrap_or(f64::NAN);
  assert!(lo > 0.0, "t_min must be strictly positive (got {lo})");
  assert!(hi > lo, "t_max must exceed t_min (got {lo}..{hi})");

  let (t, quad) = log_grid(lo, hi, grid);
  let nodes = approx.nodes();
  let weights = approx.weights();

  let total: f64 = t
    .iter()
    .zip(quad.iter())
    .map(|(&ti, &dti)| {
      let ti_t = T::from_f64_fast(ti);
      let exact = truth.evaluate(ti_t).to_f64().unwrap_or(f64::NAN);
      let fitted: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(x, w)| {
          let xl = x.to_f64().unwrap_or(f64::NAN);
          let wl = w.to_f64().unwrap_or(f64::NAN);
          wl * (-xl * ti).exp()
        })
        .sum();
      dti * (exact - fitted).abs()
    })
    .sum();

  T::from_f64_fast(total)
}

/// Log-spaced points on `[lo, hi]` with their trapezoidal weights, so the fit and the error
/// metric integrate against one rule.
fn log_grid(lo: f64, hi: f64, grid: usize) -> (Vec<f64>, Vec<f64>) {
  let ln_lo = lo.ln();
  let step = (hi.ln() - ln_lo) / (grid - 1) as f64;
  let t = (0..grid)
    .map(|i| (ln_lo + step * i as f64).exp())
    .collect::<Vec<f64>>();

  let mut quad = vec![0.0; grid];
  for i in 0..grid {
    let left = if i == 0 {
      t[0]
    } else {
      0.5 * (t[i - 1] + t[i])
    };
    let right = if i + 1 == grid {
      t[grid - 1]
    } else {
      0.5 * (t[i] + t[i + 1])
    };
    quad[i] = right - left;
  }
  (t, quad)
}

/// Solve `(AᵀUA + λI) w = AᵀU b` by Cholesky, `U = diag(obs)`; `None` when the ridge-regularised
/// normal matrix still does not factorise.
fn weighted_least_squares(
  design: &Array2<f64>,
  target: &[f64],
  obs: &[f64],
  n_prime: usize,
) -> Option<Vec<f64>> {
  let rows = target.len();
  let mut normal = Array2::<f64>::zeros((n_prime, n_prime));
  let mut rhs = vec![0.0; n_prime];

  for i in 0..rows {
    let ui = obs[i];
    for a in 0..n_prime {
      let ia = design[[i, a]];
      rhs[a] += ui * ia * target[i];
      for b in a..n_prime {
        normal[[a, b]] += ui * ia * design[[i, b]];
      }
    }
  }
  for a in 0..n_prime {
    for b in 0..a {
      normal[[a, b]] = normal[[b, a]];
    }
  }

  let mean_diag: f64 = (0..n_prime).map(|a| normal[[a, a]]).sum::<f64>() / n_prime as f64;
  let lambda = RIDGE * mean_diag;
  for a in 0..n_prime {
    normal[[a, a]] += lambda;
  }

  cholesky_solve(&mut normal, &mut rhs, n_prime).then_some(rhs)
}

/// In-place Cholesky factorisation and solve. Returns `false` if the matrix is
/// not positive definite to working precision.
fn cholesky_solve(a: &mut Array2<f64>, b: &mut [f64], n: usize) -> bool {
  for j in 0..n {
    let mut diag = a[[j, j]];
    for k in 0..j {
      diag -= a[[j, k]] * a[[j, k]];
    }
    // `is_finite` is tested first so a NaN pivot is rejected explicitly rather
    // than relying on how a comparison against NaN happens to fall out.
    if !diag.is_finite() || diag <= 0.0 {
      return false;
    }
    let d = diag.sqrt();
    a[[j, j]] = d;
    for i in (j + 1)..n {
      let mut acc = a[[i, j]];
      for k in 0..j {
        acc -= a[[i, k]] * a[[j, k]];
      }
      a[[i, j]] = acc / d;
    }
  }

  for i in 0..n {
    let mut acc = b[i];
    for k in 0..i {
      acc -= a[[i, k]] * b[k];
    }
    b[i] = acc / a[[i, i]];
  }
  for i in (0..n).rev() {
    let mut acc = b[i];
    for k in (i + 1)..n {
      acc -= a[[k, i]] * b[k];
    }
    b[i] = acc / a[[i, i]];
  }
  true
}

#[cfg(test)]
mod tests;
