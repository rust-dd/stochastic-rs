//! Two-step GMM of the CIR process on its exact transition moments: four conditions for three
//! parameters, stage 1 weighted by the identity and stage 2 by `Ŝ⁻¹`, and Hansen's `J ~ χ²₁`.
//!
//! $$
//! dX_t = \theta(\mu - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t,\qquad a = e^{-\theta\Delta},\qquad
//! g_t = \bigl(e_t,\ e_t X_t,\ u_t,\ u_t X_t\bigr)^\top
//! $$
//!
//! $$
//! e_t = X_{t+\Delta} - \mu - (X_t - \mu)\,a,\qquad J = T\,\bar g^\top \hat S^{-1} \bar g
//! $$
//!
//! $$
//! u_t = e_t^2 - X_t\,\frac{\sigma^2}{\theta}(a - a^2) - \mu\,\frac{\sigma^2}{2\theta}(1 - a)^2
//! $$
//!
//! Reference: Hansen, L. P. (1982), "Large sample properties of generalized method of moments estimators", Econometrica 50(4), 1029-1054.
//! Reference: Cox, J. C., Ingersoll, J. E. & Ross, S. A. (1985), "A theory of the term structure of interest rates", Econometrica 53(2), 385-407.
//! Reference: Chan, K. C., Karolyi, G. A., Longstaff, F. A. & Sanders, A. B. (1992), "An empirical comparison of alternative models of the short-term interest rate", Journal of Finance 47(3), 1209-1227.

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::StatsError;
use crate::optim::nelder_mead;
use crate::qmle::Carrier;
use crate::qmle::DiffusionKind;
use crate::qmle::ar1_start;
use crate::qmle::assert_dt;
use crate::qmle::diffusion_path;
use crate::traits::FloatExt;

/// The two-step GMM fit of the CIR process, in [`Cir`]'s names: `theta` the speed, `mu` the level.
#[derive(Debug, Clone)]
pub struct GmmCirFit {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  /// Hansen's over-identification statistic, `χ²₁` under the model.
  pub j_stat: f64,
  pub j_pvalue: f64,
  /// Simplex iterations over both stages.
  pub iterations: usize,
  /// Whether both stages converged.
  pub converged: bool,
}

impl GmmCirFit {
  /// The fitted [`Cir`].
  pub fn to_cir<S: SeedExt>(
    &self,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Cir<f64, S> {
    Cir::new(self.theta, self.mu, self.sigma, n, x0, t, boundary, seed)
  }
}

/// Two-step GMM of the CIR process on its exact moments, sampled every `dt`; `Err` below 5 points,
/// a negative or non-finite one, a constant path or a non-finite start; panics unless `0 < dt < ∞`.
pub fn fit_cir_gmm<T: FloatExt>(series: ArrayView1<T>, dt: f64) -> Result<GmmCirFit, StatsError> {
  assert_dt(dt);
  // `Ŝ` sums one outer product per transition, so its four moments need four transitions.
  let x = diffusion_path(series, DiffusionKind::Cir, 5)?;
  let nobs = x.len();

  // The `T − 1` moment rows `g_t` at `(θ, μ, σ)` and their column mean.
  let moments = |theta: f64, mu: f64, sigma: f64| -> (Vec<[f64; 4]>, [f64; 4]) {
    let a = (-theta * dt).exp();
    let s2 = sigma * sigma;
    let mut rows = Vec::with_capacity(nobs - 1);
    let mut mean = [0.0_f64; 4];
    for t in 0..nobs - 1 {
      let xt = x[t];
      let xnext = x[t + 1];
      let m = mu + (xt - mu) * a;
      let var = xt * (s2 / theta) * (a - a * a) + mu * (s2 / (2.0 * theta)) * (1.0 - a) * (1.0 - a);
      let e = xnext - m;
      let u = e * e - var;
      let row = [e, e * xt, u, u * xt];
      for k in 0..4 {
        mean[k] += row[k];
      }
      rows.push(row);
    }
    let inv_t = 1.0 / (nobs - 1) as f64;
    for entry in &mut mean {
      *entry *= inv_t;
    }
    (rows, mean)
  };

  let objective = |theta: f64, mu: f64, sigma: f64, w: &[[f64; 4]; 4]| -> f64 {
    let (_, g) = moments(theta, mu, sigma);
    let mut q = 0.0;
    for i in 0..4 {
      for j in 0..4 {
        q += g[i] * w[i][j] * g[j];
      }
    }
    q
  };

  let start = ar1_start(
    &x,
    dt,
    DiffusionKind::Cir,
    Carrier::new(&x, DiffusionKind::Cir)?,
  )?;

  let identity = {
    let mut w = [[0.0; 4]; 4];
    for (i, row) in w.iter_mut().enumerate() {
      row[i] = 1.0;
    }
    w
  };

  let (p1, it1, conv1) = nelder_mead(start, 2000, |p| {
    objective(p[0].exp(), p[1].exp(), p[2].exp(), &identity)
  });
  let (theta1, mu1, s1) = (p1[0].exp(), p1[1].exp(), p1[2].exp());

  // Stage 2 weights by `Ŝ⁻¹` at the stage-1 estimate, Hansen's efficient choice.
  let (rows1, _) = moments(theta1, mu1, s1);
  let mut s_hat = [[0.0_f64; 4]; 4];
  for row in &rows1 {
    for i in 0..4 {
      for j in 0..4 {
        s_hat[i][j] += row[i] * row[j];
      }
    }
  }
  let inv_t = 1.0 / rows1.len() as f64;
  for (i, row) in s_hat.iter_mut().enumerate() {
    for entry in row.iter_mut() {
      *entry *= inv_t;
    }
    // Ridge regularisation guards against a near-singular Ŝ on short series.
    row[i] += 1e-10;
  }
  let w2 = invert4(&s_hat).unwrap_or(identity);

  let (p2, it2, conv2) = nelder_mead([p1[0], p1[1], p1[2]], 2000, |p| {
    objective(p[0].exp(), p[1].exp(), p[2].exp(), &w2)
  });
  let (theta, mu, sigma) = (p2[0].exp(), p2[1].exp(), p2[2].exp());

  let j_stat = (rows1.len() as f64) * objective(theta, mu, sigma, &w2);
  let j_pvalue = 1.0 - chi2_cdf_1dof(j_stat);

  Ok(GmmCirFit {
    theta,
    mu,
    sigma,
    j_stat,
    j_pvalue,
    iterations: it1 + it2,
    converged: conv1 && conv2,
  })
}

/// 4×4 Gauss-Jordan matrix inverse. Returns `None` on a singular pivot.
fn invert4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
  let mut a = *m;
  let mut inv = {
    let mut e = [[0.0; 4]; 4];
    for (i, row) in e.iter_mut().enumerate() {
      row[i] = 1.0;
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
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::gamma::SimdGamma;
  use stochastic_rs_distributions::poisson::SimdPoisson;
  use stochastic_rs_stochastic::traits::ProcessExt;

  use super::*;

  /// Exact CIR step `c χ'²_d(λ)`: `c = σ²(1−a)/(4θ)`, `d = 4θμ/σ²`, `λ = 4θa X_t/(σ²(1−a))`,
  /// the non-central χ² drawn as a Poisson mixture of central ones (Gammas).
  fn cir_exact_step(
    x_t: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    pois_seed: u64,
    gamma_seed: u64,
  ) -> f64 {
    let a = (-theta * dt).exp();
    let s2 = sigma * sigma;
    let c = s2 * (1.0 - a) / (4.0 * theta);
    let d = 4.0 * theta * mu / s2;
    let lambda = x_t * 4.0 * theta * a / (s2 * (1.0 - a));
    // χ'²_d(λ) = χ²_{d + 2N}, N ~ Poisson(λ/2); central χ²_k = Gamma(k/2, 2).
    let n_pois: f64 = f64::from(
      SimdPoisson::<u32>::new(lambda / 2.0)
        .seeded(&Deterministic::new(pois_seed))
        .sample(),
    );
    let shape = d / 2.0 + n_pois;
    let chi2 = if shape > 0.0 {
      SimdGamma::<f64>::new(shape, 2.0)
        .seeded(&Deterministic::new(gamma_seed))
        .sample()
    } else {
      0.0
    };
    c * chi2
  }

  fn simulate_cir_path(
    theta: f64,
    mu: f64,
    sigma: f64,
    x0: f64,
    dt: f64,
    n: usize,
    seed: u64,
  ) -> Array1<f64> {
    let mut path = Array1::<f64>::zeros(n);
    path[0] = x0;
    for t in 1..n {
      // Each step gets its own derived seeds so the path stays reproducible
      // without threading a mutable engine through the helper.
      path[t] = cir_exact_step(
        path[t - 1],
        theta,
        mu,
        sigma,
        dt,
        seed ^ (t as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        seed ^ (t as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9),
      );
    }
    path
  }

  /// Best of three pinned paths: the level's band sits near one standard error, so one path alone
  /// is a coin flip.
  #[test]
  fn gmm_cir_recovers_params_from_exact_path() {
    let (tht, mt, st) = (2.0, 0.04, 0.25);
    let dt = 1.0 / 252.0;
    let fits = [7u64, 13, 29].map(|seed| {
      let path = simulate_cir_path(tht, mt, st, 0.04, dt, 8000, seed);
      (seed, fit_cir_gmm(path.view(), dt).unwrap())
    });
    let ok = fits.iter().any(|(_, r)| {
      r.converged
        && (r.mu - mt).abs() / mt < 0.12
        && (r.theta - tht).abs() / tht < 0.5
        && (r.sigma - st).abs() / st < 0.3
    });
    let report = fits
      .iter()
      .map(|(s, r)| format!("seed {s}: θ={:.4} μ={:.4} σ={:.4}", r.theta, r.mu, r.sigma))
      .collect::<Vec<_>>()
      .join("; ");
    assert!(ok, "no seed recovered (θ={tht}, μ={mt}, σ={st}) — {report}");
  }

  /// The fit's `theta` is the speed and `mu` the level, as in `Cir`, so the conversion hands them
  /// over.
  #[test]
  fn a_gmm_fit_round_trips_into_the_cir_it_names() {
    let truth = Cir::<f64, _>::new(
      5.0,
      0.04,
      0.2,
      10_000,
      Some(0.04),
      Some(100.0),
      VarianceBoundary::Reflect,
      Deterministic::new(5),
    );
    let path = truth.sample();
    let fit = fit_cir_gmm(path.view(), 100.0 / 9_999.0).unwrap();
    assert!((fit.theta - 5.0).abs() < 1.5, "speed {}", fit.theta);
    assert!((fit.mu - 0.04).abs() < 0.01, "level {}", fit.mu);
    let rebuilt = fit.to_cir(
      10_000,
      Some(0.04),
      Some(100.0),
      VarianceBoundary::Reflect,
      Deterministic::new(6),
    );
    assert_eq!(
      (rebuilt.theta, rebuilt.mu, rebuilt.sigma),
      (fit.theta, fit.mu, fit.sigma)
    );
    assert_eq!(rebuilt.sample().len(), 10_000);
  }

  /// The Hansen J p-value should be large (fail to reject the over-id
  /// restriction) when the model is correctly specified.
  #[test]
  fn gmm_cir_j_test_does_not_reject_correct_model() {
    let dt = 1.0 / 252.0;
    let path = simulate_cir_path(1.5, 0.05, 0.3, 0.05, dt, 6000, 11);
    let res = fit_cir_gmm(path.view(), dt).unwrap();
    assert!(
      res.j_pvalue > 0.01,
      "correctly-specified CIR should not be rejected at 1%, J p-value = {}",
      res.j_pvalue
    );
  }

  /// `Ŝ` sums one outer product per transition, so its four moments need four transitions.
  #[test]
  fn gmm_cir_rejects_a_short_series() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, 0.05]);
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::InsufficientData { needed: 5, got: 4 }
    );
  }

  /// The mean, the autocorrelation and the speed are finite, but the residuals near 1.1e154
  /// square past the largest double: `σ₀` is `∞` before its clamp to 10.
  #[test]
  fn gmm_cir_answers_an_overflowing_sigma_start_with_a_numerical_error() {
    let s = Array1::from(vec![1.2e154, 0.0, 1.2e154, 0.0, 1.2e154]);
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::Numerical {
        stage: "starting point"
      }
    );
  }

  #[test]
  fn gmm_cir_rejects_a_negative_observation() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, -0.01, 0.04, 0.05]);
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::InvalidObservation {
        index: 3,
        value: -0.01,
        constraint: "finite and >= 0"
      }
    );
  }

  /// The sample mean of these finite points overflows, so the moment start is not finite.
  #[test]
  fn gmm_cir_answers_an_overflowing_start_with_a_numerical_error() {
    let s = Array1::from(vec![1e308, 1.5e308, 1e308, 0.5e308, 1.2e308]);
    assert_eq!(
      fit_cir_gmm(s.view(), 1.0 / 252.0).unwrap_err(),
      StatsError::Numerical {
        stage: "starting point"
      }
    );
  }

  #[test]
  #[should_panic(expected = "dt must satisfy `dt.is_finite() && dt > 0.0`, got dt = 0.0")]
  fn gmm_cir_panics_on_bad_dt() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, 0.05]);
    let _ = fit_cir_gmm(s.view(), 0.0);
  }
}
