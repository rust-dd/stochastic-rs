//! # $L^1$-optimal exponential-sum fitting
//!
//! The Markov lift replaces a kernel by $K(t)\approx\sum_l w_l e^{-x_l t}$, so
//! how the fit is scored decides what the resulting simulation is good at.
//!
//! Quadrature-derived fits — including [`RlKernel`](crate::rough::kernel::RlKernel)'s
//! generalised Gauss–Laguerre rule — implicitly target a squared/pointwise
//! criterion. That is the right objective for **strong** (pathwise) error.
//! Bayer & Breneis (2023), *Weak Markovian approximations of rough Heston*
//! (arXiv:2309.07023), show that the **weak** error — the one that governs
//! option prices — is instead bounded by the $L^1$ error of the kernel
//! approximation. Fitting in $L^1$ therefore buys accuracy exactly where a
//! pricing application spends it.
//!
//! [`fit_l1`] keeps the incoming kernel's nodes, which quadrature already
//! places well, and refits only the weights. That is a linear $L^1$
//! regression, solved here by iteratively reweighted least squares: the
//! $L^1$ objective $\sum_i |r_i|$ is minimised as a limit of weighted
//! least-squares problems with observation weights $1/|r_i|$.
//!
//! ## What this is and is not
//!
//! Measured at degree 40 on $[10^{-4}, 1]$, the refit lowers the $L^1$ error
//! by **2.3× to 3.6×** across $H \in \{0.05, 0.1, 0.3, 0.45\}$, and — on this
//! interval — its worst pointwise relative error improves too, to between
//! 0.70 and 0.86 of the quadrature fit's. It is nevertheless free to be
//! *locally* worse at a particular $t$: minimising an integral is not the same
//! as bounding a maximum, and at $t = 10^{-2}$ the quadrature fit is the
//! tighter of the two. Both properties are pinned by tests.
//!
//! This does not move the nodes, so it cannot rescue a fit whose nodes are
//! badly placed to begin with. Near the singularity neither fit is accurate:
//! at $t = 10^{-4}$ with $H = 0.05$ both are ~80% off in relative terms,
//! because no degree-40 exponential sum tracks $t^{H-1/2}$ that close to the
//! origin. That is a property of the representation, not of the objective, and
//! is exactly why the lift carries a separate boundary term
//! ([`VolterraKernel::integral_from_zero`]) for the first step rather than
//! leaning on the sum there.
//!
//! Use the quadrature fit for pathwise work and this one for pricing.

use ndarray::Array1;
use ndarray::Array2;

use super::kernel::SumOfExponentials;
use super::kernel::VolterraKernel;
use crate::traits::FloatExt;

/// Number of IRLS iterations. Five is enough for the weight vector to settle
/// on the fits this crate produces; the residual change past that is below
/// the fit error itself.
const IRLS_ITERATIONS: usize = 5;

/// Ridge parameter, relative to the mean diagonal of the normal matrix. The
/// exponential basis is strongly non-orthogonal — neighbouring nodes give
/// near-collinear columns — so the normal equations are ill-conditioned and a
/// bare Cholesky can fail outright. This is small enough to leave the fit
/// alone and large enough to keep the factorisation well defined.
const RIDGE: f64 = 1e-12;

/// Floor on `|r_i|` in the IRLS observation weight, so an exactly-zero
/// residual cannot produce an infinite weight.
const RESIDUAL_FLOOR: f64 = 1e-300;

/// Refit `kernel`'s weights to minimise the $L^1$ error of the exponential-sum
/// approximation on `[t_min, t_max]`, keeping its nodes.
///
/// `grid` is the number of sample points; they are log-spaced, because for a
/// weakly singular kernel both the kernel mass and the approximation error
/// concentrate near zero, and a uniform grid would barely resolve the region
/// that dominates the integral.
///
/// Returns the incoming weights unchanged if the normal equations cannot be
/// factorised on the first iteration. That is reachable only through
/// [`SumOfExponentials`] with nodes badly scaled to `[t_min, t_max]` — nodes
/// so fast that every basis function underflows to zero across the whole grid
/// leave a zero normal matrix, and the ridge, which scales with its mean
/// diagonal, is then zero too. No kernel this crate constructs reaches that
/// state at any sane degree. There is deliberately no error return; use
/// [`l1_error`] to confirm a fit actually improved rather than assuming it.
///
/// # Panics
/// - if `t_min` is not strictly positive, or `t_max <= t_min` (the fractional
///   kernel is singular at the origin, so the interval must exclude it)
/// - if `grid` is smaller than the kernel's degree, which would leave the
///   weight fit underdetermined
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
      // A failed factorisation means this iteration had nothing to add; the
      // weights from the previous one are still a valid fit, so stop rather
      // than return something unchecked.
      None => break,
    }
  }

  let refit = Array1::from_iter(weights.into_iter().map(T::from_f64_fast));
  SumOfExponentials::new(kernel.nodes().clone(), refit)
}

/// Total $L^1$ error $\int_{t_{\min}}^{t_{\max}} |K(t) - \sum_l w_l e^{-x_l t}|\,dt$
/// of `approx` measured against `truth`'s exact kernel, on the same log-spaced
/// grid [`fit_l1`] fits on.
///
/// Exposed because a fit's whole claim is that this number is smaller, and a
/// caller should be able to check that rather than take it on faith.
///
/// # Panics
/// Under the same conditions as [`fit_l1`].
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

/// Log-spaced sample points on `[lo, hi]` with their trapezoidal quadrature
/// weights, returned together so the fit and the error metric integrate
/// against exactly the same rule.
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

/// Solve `(AᵀUA + λI) w = AᵀU b` by Cholesky, where `U = diag(obs)`.
///
/// Returns `None` if the ridge-regularised normal matrix still fails to
/// factorise, which the caller treats as "this iteration contributed nothing"
/// rather than as a hard error.
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
#[path = "fit_tests.rs"]
mod tests;
