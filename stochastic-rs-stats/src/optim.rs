//! Small dependency-free optimisers shared by the estimator modules.
//!
//! The square-root / mean-reverting calibrators (`gmm_cir`, `qmle`) minimise
//! a smooth 3-parameter objective in log-space and the GARCH fitter a
//! `p + q + 2`-parameter one; a compact Nelder-Mead simplex is enough for
//! both, so the fixed-size entry point is a thin wrapper over the
//! any-dimension one.

/// Nelder-Mead simplex minimiser for a 3-parameter objective.
///
/// Returns `(argmin, iterations, converged)`. The caller is responsible
/// for any parameter reparameterisation (e.g. carrying log-parameters so
/// the search stays in the positive orthant).
///
/// `max_iter` caps the simplex iterations: a smooth analytic objective
/// (GMM / QMLE) converges in tens of iterations to the `1e-10` tolerance,
/// but a Monte-Carlo objective with small resampling-induced kinks (the
/// particle-filter likelihood) never hits that tolerance and should be
/// capped at a few hundred iterations to bound runtime.
pub(crate) fn nelder_mead<F: Fn(&[f64; 3]) -> f64>(
  start: [f64; 3],
  max_iter: usize,
  f: F,
) -> ([f64; 3], usize, bool) {
  let (p, iters, converged) = nelder_mead_vec(&start, max_iter, |p| f(&[p[0], p[1], p[2]]));
  ([p[0], p[1], p[2]], iters, converged)
}

/// Nelder-Mead simplex minimiser for an objective of any dimension, under
/// the same reflection / expansion / contraction / shrink coefficients and
/// `1e-10` spread tolerance as [`nelder_mead`]; the initial simplex offsets
/// each coordinate by `0.1`.
pub(crate) fn nelder_mead_vec<F: Fn(&[f64]) -> f64>(
  start: &[f64],
  max_iter: usize,
  f: F,
) -> (Vec<f64>, usize, bool) {
  const ALPHA: f64 = 1.0;
  const GAMMA: f64 = 2.0;
  const RHO: f64 = 0.5;
  const SHRINK: f64 = 0.5;
  const TOL: f64 = 1e-10;

  let n = start.len();
  let mut simplex: Vec<Vec<f64>> = vec![start.to_vec(); n + 1];
  for i in 0..n {
    simplex[i + 1][i] += 0.1;
  }
  let mut fvals: Vec<f64> = simplex.iter().map(|p| f(p)).collect();

  let mut iters = 0;
  while iters < max_iter {
    iters += 1;
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
    let best = order[0];
    let worst = order[n];
    let second_worst = order[n - 1];

    if (fvals[worst] - fvals[best]).abs() < TOL {
      return (simplex[best].clone(), iters, true);
    }

    let mut centroid = vec![0.0; n];
    for &o in &order[..n] {
      for d in 0..n {
        centroid[d] += simplex[o][d] / n as f64;
      }
    }

    let reflect = combine(&centroid, &simplex[worst], ALPHA, true);
    let f_reflect = f(&reflect);

    if f_reflect < fvals[best] {
      let expand = combine(&centroid, &simplex[worst], ALPHA * GAMMA, true);
      let f_expand = f(&expand);
      if f_expand < f_reflect {
        simplex[worst] = expand;
        fvals[worst] = f_expand;
      } else {
        simplex[worst] = reflect;
        fvals[worst] = f_reflect;
      }
    } else if f_reflect < fvals[second_worst] {
      simplex[worst] = reflect;
      fvals[worst] = f_reflect;
    } else {
      let contract = combine(&centroid, &simplex[worst], RHO, false);
      let f_contract = f(&contract);
      if f_contract < fvals[worst] {
        simplex[worst] = contract;
        fvals[worst] = f_contract;
      } else {
        let anchor = simplex[best].clone();
        for &o in &order[1..] {
          for d in 0..n {
            simplex[o][d] = anchor[d] + SHRINK * (simplex[o][d] - anchor[d]);
          }
          fvals[o] = f(&simplex[o]);
        }
      }
    }
  }
  let mut best = 0;
  for i in 1..=n {
    if fvals[i] < fvals[best] {
      best = i;
    }
  }
  (simplex[best].clone(), iters, false)
}

/// Reflection (`reflect = true`) or contraction (`reflect = false`) of the
/// worst vertex through the centroid with the given coefficient.
fn combine(centroid: &[f64], worst: &[f64], coef: f64, reflect: bool) -> Vec<f64> {
  centroid
    .iter()
    .zip(worst)
    .map(|(c, w)| {
      if reflect {
        c + coef * (c - w)
      } else {
        c + coef * (w - c)
      }
    })
    .collect()
}
