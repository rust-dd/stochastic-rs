//! Small dependency-free optimisers shared by the estimator modules.
//!
//! The square-root / mean-reverting calibrators (`gmm_cir`, `qmle`) all
//! minimise a smooth 3-parameter objective in log-space; a compact
//! fixed-size Nelder-Mead simplex is enough and avoids pulling `argmin`
//! (and its `openblas` transitive linkage) into the default stats build.

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
  const ALPHA: f64 = 1.0;
  const GAMMA: f64 = 2.0;
  const RHO: f64 = 0.5;
  const SHRINK: f64 = 0.5;
  const TOL: f64 = 1e-10;

  let mut simplex = [start, start, start, start];
  for i in 0..3 {
    simplex[i + 1][i] += 0.1;
  }
  let mut fvals = [
    f(&simplex[0]),
    f(&simplex[1]),
    f(&simplex[2]),
    f(&simplex[3]),
  ];

  let mut iters = 0;
  while iters < max_iter {
    iters += 1;
    let mut order = [0, 1, 2, 3];
    order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
    let best = order[0];
    let worst = order[3];
    let second_worst = order[2];

    if (fvals[worst] - fvals[best]).abs() < TOL {
      return (simplex[best], iters, true);
    }

    let mut centroid = [0.0; 3];
    for &o in &order[..3] {
      for d in 0..3 {
        centroid[d] += simplex[o][d] / 3.0;
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
        for &o in &order[1..] {
          for d in 0..3 {
            simplex[o][d] = simplex[best][d] + SHRINK * (simplex[o][d] - simplex[best][d]);
          }
          fvals[o] = f(&simplex[o]);
        }
      }
    }
  }
  let mut best = 0;
  for i in 1..4 {
    if fvals[i] < fvals[best] {
      best = i;
    }
  }
  (simplex[best], iters, false)
}

/// Reflection (`reflect = true`) or contraction (`reflect = false`) of the
/// worst vertex through the centroid with the given coefficient.
fn combine(centroid: &[f64; 3], worst: &[f64; 3], coef: f64, reflect: bool) -> [f64; 3] {
  let mut p = [0.0; 3];
  for d in 0..3 {
    p[d] = if reflect {
      centroid[d] + coef * (centroid[d] - worst[d])
    } else {
      centroid[d] + coef * (worst[d] - centroid[d])
    };
  }
  p
}
