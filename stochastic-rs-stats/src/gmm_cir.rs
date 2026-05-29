//! # GMM estimator for the CIR / Heston-variance process
//!
//! $$
//! dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t
//! $$
//!
//! The variance leg of the Heston model is a Cox-Ingersoll-Ross (CIR)
//! square-root diffusion. This estimator recovers $(\kappa, \theta,
//! \sigma)$ from a discretely-observed variance / short-rate series by
//! Hansen's (1982) Generalized Method of Moments, using the **exact**
//! conditional moments of the CIR transition law (Cox-Ingersoll-Ross
//! 1985), so the estimator carries no Euler-discretisation bias at any
//! sampling interval $\Delta$.
//!
//! With $a = e^{-\kappa\Delta}$ the exact one-step moments are
//!
//! $$
//! \begin{aligned}
//! m(X_t) &= \mathbb{E}[X_{t+\Delta}\mid X_t] = \theta + (X_t - \theta)\,a,\\
//! v(X_t) &= \operatorname{Var}[X_{t+\Delta}\mid X_t]
//!        = X_t\,\frac{\sigma^2}{\kappa}(a - a^2)
//!        + \theta\,\frac{\sigma^2}{2\kappa}(1 - a)^2.
//! \end{aligned}
//! $$
//!
//! Defining the residuals $e_t = X_{t+\Delta} - m(X_t)$ and
//! $u_t = e_t^2 - v(X_t)$, the population moment conditions instrumented by
//! $\{1, X_t\}$ are $\mathbb{E}[g_t] = 0$ with
//! $g_t = (e_t,\ e_t X_t,\ u_t,\ u_t X_t)^\top$ — four conditions for three
//! parameters, so the system is over-identified by one, and Hansen's
//! $J$-statistic $T\,\bar g'\,\hat S^{-1}\,\bar g \sim \chi^2_1$ tests the
//! over-identifying restriction.
//!
//! Two-step GMM: stage 1 minimises $\bar g' \bar g$ (identity weighting);
//! stage 2 re-weights by the inverse moment-covariance $\hat S^{-1}$
//! evaluated at the stage-1 estimate (Hansen 1982, the efficient GMM
//! weighting matrix).
//!
//! References:
//! - Hansen, L.P. (1982), "Large sample properties of generalized method
//!   of moments estimators", *Econometrica* 50(4), 1029-1054.
//! - Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985), "A theory of the term
//!   structure of interest rates", *Econometrica* 53(2), 385-407
//!   (exact transition moments).
//! - Chan, K.C., Karolyi, G.A., Longstaff, F.A., Sanders, A.B. (1992),
//!   "An empirical comparison of alternative models of the short-term
//!   interest rate", *Journal of Finance* 47(3), 1209-1227 (GMM applied
//!   to short-rate diffusions).

use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Result of the two-step GMM fit of the CIR variance process.
#[derive(Debug, Clone)]
pub struct GmmCirResult {
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run level $\theta$.
  pub theta: f64,
  /// Volatility-of-variance $\sigma$.
  pub sigma: f64,
  /// Hansen over-identification $J$-statistic ($\sim \chi^2_1$).
  pub j_stat: f64,
  /// Asymptotic p-value of the $J$-statistic under $\chi^2_1$.
  pub j_pvalue: f64,
  /// Nelder-Mead iterations consumed (summed over both GMM stages).
  pub iterations: usize,
  /// Whether both optimisation stages converged.
  pub converged: bool,
}

/// Fit the CIR variance process by two-step GMM on the exact conditional
/// moments.
///
/// `series` is the discretely-observed variance (or short-rate) path and
/// `dt` the sampling interval $\Delta$. Requires at least 3 observations.
pub fn gmm_cir<T: FloatExt>(series: ArrayView1<T>, dt: f64) -> GmmCirResult {
  let n_obs = series.len();
  assert!(n_obs >= 3, "gmm_cir requires at least 3 observations");
  assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");

  let x: Vec<f64> = series.iter().map(|v| v.to_f64().unwrap()).collect();

  // Moment matrix builder: for params (kappa, theta, sigma) returns the
  // T-1 length-4 moment rows g_t and their column mean.
  let moments = |kappa: f64, theta: f64, sigma: f64| -> (Vec<[f64; 4]>, [f64; 4]) {
    let a = (-kappa * dt).exp();
    let s2 = sigma * sigma;
    let mut rows = Vec::with_capacity(n_obs - 1);
    let mut mean = [0.0_f64; 4];
    for t in 0..n_obs - 1 {
      let xt = x[t];
      let xnext = x[t + 1];
      let m = theta + (xt - theta) * a;
      let var = xt * (s2 / kappa) * (a - a * a) + theta * (s2 / (2.0 * kappa)) * (1.0 - a) * (1.0 - a);
      let e = xnext - m;
      let u = e * e - var;
      let row = [e, e * xt, u, u * xt];
      for k in 0..4 {
        mean[k] += row[k];
      }
      rows.push(row);
    }
    let inv_t = 1.0 / (n_obs - 1) as f64;
    for k in 0..4 {
      mean[k] *= inv_t;
    }
    (rows, mean)
  };

  // GMM objective with a fixed 4×4 weighting matrix `w`.
  let objective = |kappa: f64, theta: f64, sigma: f64, w: &[[f64; 4]; 4]| -> f64 {
    let (_, g) = moments(kappa, theta, sigma);
    let mut q = 0.0;
    for i in 0..4 {
      for j in 0..4 {
        q += g[i] * w[i][j] * g[j];
      }
    }
    q
  };

  // Initial guess from sample moments: θ̂ = mean(X); κ̂ from the lag-1
  // autocorrelation a = corr(X_{t+1}, X_t) ⟹ κ = -ln(a)/Δ; σ̂ from the
  // residual variance scale.
  let sample_mean = x.iter().sum::<f64>() / x.len() as f64;
  let theta0 = sample_mean.max(1e-8);
  let (mut num, mut den) = (0.0, 0.0);
  for t in 0..n_obs - 1 {
    num += (x[t] - sample_mean) * (x[t + 1] - sample_mean);
    den += (x[t] - sample_mean).powi(2);
  }
  let ac1 = if den > 0.0 { (num / den).clamp(1e-4, 0.999) } else { 0.5 };
  let kappa0 = (-ac1.ln() / dt).clamp(1e-3, 50.0);
  let mut resid_var = 0.0;
  for t in 0..n_obs - 1 {
    let a = (-kappa0 * dt).exp();
    let m = theta0 + (x[t] - theta0) * a;
    resid_var += (x[t + 1] - m).powi(2);
  }
  resid_var /= (n_obs - 1) as f64;
  // var(X_{t+1}|X_t) ≈ X̄ σ²/κ (a-a²) + ... ≈ scale; invert for σ².
  let a0 = (-kappa0 * dt).exp();
  let scale = sample_mean * (a0 - a0 * a0) / kappa0 + theta0 * (1.0 - a0).powi(2) / (2.0 * kappa0);
  let sigma0 = (resid_var / scale.max(1e-12)).sqrt().clamp(1e-3, 10.0);

  let identity = {
    let mut w = [[0.0; 4]; 4];
    for i in 0..4 {
      w[i][i] = 1.0;
    }
    w
  };

  // Stage 1: identity weighting.
  let (p1, it1, conv1) = nelder_mead(
    [kappa0.ln(), theta0.ln(), sigma0.ln()],
    |p| objective(p[0].exp(), p[1].exp(), p[2].exp(), &identity),
  );
  let (k1, th1, s1) = (p1[0].exp(), p1[1].exp(), p1[2].exp());

  // Stage 2: efficient weighting Ŝ⁻¹ evaluated at the stage-1 estimate.
  let (rows1, _) = moments(k1, th1, s1);
  let mut s_hat = [[0.0_f64; 4]; 4];
  for row in &rows1 {
    for i in 0..4 {
      for j in 0..4 {
        s_hat[i][j] += row[i] * row[j];
      }
    }
  }
  let inv_t = 1.0 / rows1.len() as f64;
  for i in 0..4 {
    for j in 0..4 {
      s_hat[i][j] *= inv_t;
    }
    // Ridge regularisation guards against a near-singular Ŝ on short series.
    s_hat[i][i] += 1e-10;
  }
  let w2 = invert4(&s_hat).unwrap_or(identity);

  let (p2, it2, conv2) = nelder_mead([p1[0], p1[1], p1[2]], |p| {
    objective(p[0].exp(), p[1].exp(), p[2].exp(), &w2)
  });
  let (kappa, theta, sigma) = (p2[0].exp(), p2[1].exp(), p2[2].exp());

  // Hansen J = T · ḡ' Ŝ⁻¹ ḡ ~ χ²(#moments − #params) = χ²(1).
  let j_stat = (rows1.len() as f64) * objective(kappa, theta, sigma, &w2);
  let j_pvalue = 1.0 - chi2_cdf_1dof(j_stat);

  GmmCirResult {
    kappa,
    theta,
    sigma,
    j_stat,
    j_pvalue,
    iterations: it1 + it2,
    converged: conv1 && conv2,
  }
}

/// Compact fixed-size Nelder-Mead simplex minimiser for the 3-parameter
/// GMM objective (parameters carried in log-space so they stay positive).
/// Returns `(argmin, iterations, converged)`.
fn nelder_mead<F: Fn(&[f64; 3]) -> f64>(start: [f64; 3], f: F) -> ([f64; 3], usize, bool) {
  const ALPHA: f64 = 1.0;
  const GAMMA: f64 = 2.0;
  const RHO: f64 = 0.5;
  const SIGMA: f64 = 0.5;
  const MAX_ITER: usize = 2000;
  const TOL: f64 = 1e-10;

  // Initial simplex: start plus a perturbation along each axis.
  let mut simplex = [start, start, start, start];
  for i in 0..3 {
    simplex[i + 1][i] += 0.1;
  }
  let mut fvals = [f(&simplex[0]), f(&simplex[1]), f(&simplex[2]), f(&simplex[3])];

  let mut iters = 0;
  while iters < MAX_ITER {
    iters += 1;
    // Order vertices by objective.
    let mut order = [0, 1, 2, 3];
    order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
    let best = order[0];
    let worst = order[3];
    let second_worst = order[2];

    if (fvals[worst] - fvals[best]).abs() < TOL {
      return (simplex[best], iters, true);
    }

    // Centroid of all but the worst vertex.
    let mut centroid = [0.0; 3];
    for &o in &order[..3] {
      for d in 0..3 {
        centroid[d] += simplex[o][d] / 3.0;
      }
    }

    let reflect = reflect_point(&centroid, &simplex[worst], ALPHA);
    let f_reflect = f(&reflect);

    if f_reflect < fvals[best] {
      let expand = reflect_point(&centroid, &simplex[worst], ALPHA * GAMMA);
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
      // Contraction.
      let contract = contract_point(&centroid, &simplex[worst], RHO);
      let f_contract = f(&contract);
      if f_contract < fvals[worst] {
        simplex[worst] = contract;
        fvals[worst] = f_contract;
      } else {
        // Shrink toward the best vertex.
        for &o in &order[1..] {
          for d in 0..3 {
            simplex[o][d] = simplex[best][d] + SIGMA * (simplex[o][d] - simplex[best][d]);
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

fn reflect_point(centroid: &[f64; 3], worst: &[f64; 3], coef: f64) -> [f64; 3] {
  let mut p = [0.0; 3];
  for d in 0..3 {
    p[d] = centroid[d] + coef * (centroid[d] - worst[d]);
  }
  p
}

fn contract_point(centroid: &[f64; 3], worst: &[f64; 3], coef: f64) -> [f64; 3] {
  let mut p = [0.0; 3];
  for d in 0..3 {
    p[d] = centroid[d] + coef * (worst[d] - centroid[d]);
  }
  p
}

/// 4×4 Gauss-Jordan matrix inverse. Returns `None` on a singular pivot.
fn invert4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
  let mut a = *m;
  let mut inv = {
    let mut e = [[0.0; 4]; 4];
    for i in 0..4 {
      e[i][i] = 1.0;
    }
    e
  };
  for col in 0..4 {
    // Partial pivot.
    let mut pivot = col;
    for r in (col + 1)..4 {
      if a[r][col].abs() > a[pivot][col].abs() {
        pivot = r;
      }
    }
    if a[pivot][col].abs() < 1e-300 {
      return None;
    }
    a.swap(col, pivot);
    inv.swap(col, pivot);
    let d = a[col][col];
    for k in 0..4 {
      a[col][k] /= d;
      inv[col][k] /= d;
    }
    for r in 0..4 {
      if r != col {
        let factor = a[r][col];
        for k in 0..4 {
          a[r][k] -= factor * a[col][k];
          inv[r][k] -= factor * inv[col][k];
        }
      }
    }
  }
  Some(inv)
}

/// CDF of the $\chi^2_1$ distribution at `x`: $P(\chi^2_1 \le x) =
/// \operatorname{erf}(\sqrt{x/2})$.
fn chi2_cdf_1dof(x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  stochastic_rs_distributions::special::erf((x / 2.0).sqrt())
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::Gamma;
  use rand_distr::Poisson;

  use super::*;

  /// Exact CIR transition: $X_{t+\Delta} = c\,\chi'^2_d(\lambda)$ with
  /// $c = \sigma^2(1-a)/(4\kappa)$, $d = 4\kappa\theta/\sigma^2$,
  /// $\lambda = X_t\,4\kappa a / (\sigma^2(1-a))$, $a = e^{-\kappa\Delta}$.
  /// Non-central χ² sampled as a Poisson mixture of central χ² (= Gamma).
  fn cir_exact_step(x_t: f64, kappa: f64, theta: f64, sigma: f64, dt: f64, rng: &mut StdRng) -> f64 {
    let a = (-kappa * dt).exp();
    let s2 = sigma * sigma;
    let c = s2 * (1.0 - a) / (4.0 * kappa);
    let d = 4.0 * kappa * theta / s2;
    let lambda = x_t * 4.0 * kappa * a / (s2 * (1.0 - a));
    // χ'²_d(λ) = χ²_{d + 2N}, N ~ Poisson(λ/2); central χ²_k = Gamma(k/2, 2).
    let n_pois = Poisson::new(lambda / 2.0).unwrap().sample(rng) as f64;
    let shape = d / 2.0 + n_pois;
    let chi2 = if shape > 0.0 {
      Gamma::new(shape, 2.0).unwrap().sample(rng)
    } else {
      0.0
    };
    c * chi2
  }

  fn simulate_cir_path(
    kappa: f64,
    theta: f64,
    sigma: f64,
    x0: f64,
    dt: f64,
    n: usize,
    seed: u64,
  ) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut path = Array1::<f64>::zeros(n);
    path[0] = x0;
    for t in 1..n {
      path[t] = cir_exact_step(path[t - 1], kappa, theta, sigma, dt, &mut rng);
    }
    path
  }

  /// Round-trip: simulate an exactly-distributed CIR path with known
  /// params, then GMM must recover them. θ (the long-run mean) is the
  /// best-identified; κ and σ are noisier on finite samples, so the bands
  /// reflect realistic GMM-CIR sampling variability.
  #[test]
  fn gmm_cir_recovers_params_from_exact_path() {
    let kappa_true = 2.0;
    let theta_true = 0.04;
    let sigma_true = 0.25;
    let dt = 1.0 / 252.0;
    let path = simulate_cir_path(kappa_true, theta_true, sigma_true, 0.04, dt, 8000, 7);

    let res = gmm_cir(path.view(), dt);
    assert!(res.converged, "GMM stages must converge");
    assert!(
      (res.theta - theta_true).abs() / theta_true < 0.12,
      "θ = {} vs true {theta_true} (>12% off)",
      res.theta
    );
    assert!(
      (res.kappa - kappa_true).abs() / kappa_true < 0.5,
      "κ = {} vs true {kappa_true} (>50% off)",
      res.kappa
    );
    assert!(
      (res.sigma - sigma_true).abs() / sigma_true < 0.3,
      "σ = {} vs true {sigma_true} (>30% off)",
      res.sigma
    );
  }

  /// The Hansen J p-value should be large (fail to reject the over-id
  /// restriction) when the model is correctly specified.
  #[test]
  fn gmm_cir_j_test_does_not_reject_correct_model() {
    let dt = 1.0 / 252.0;
    let path = simulate_cir_path(1.5, 0.05, 0.3, 0.05, dt, 6000, 11);
    let res = gmm_cir(path.view(), dt);
    assert!(
      res.j_pvalue > 0.01,
      "correctly-specified CIR should not be rejected at 1%, J p-value = {}",
      res.j_pvalue
    );
  }

  #[test]
  #[should_panic(expected = "at least 3 observations")]
  fn gmm_cir_panics_on_short_series() {
    let s = Array1::from(vec![0.04, 0.05]);
    let _ = gmm_cir(s.view(), 1.0 / 252.0);
  }

  #[test]
  #[should_panic(expected = "dt must be finite and positive")]
  fn gmm_cir_panics_on_bad_dt() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, 0.05]);
    let _ = gmm_cir(s.view(), 0.0);
  }
}
