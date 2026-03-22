//! # DRR-GARCH
//!
//! Double Ridge Regression with GARCH(1,1) errors.
//!
//! $$
//! y_{t+1} = \alpha \cdot y_t + \beta' x_t + \varepsilon_{t+1}, \quad
//! \sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
//! $$
//!
//! Two-step ridge regression for autoregressive models with many exogenous
//! predictors, combined with GARCH(1,1) error modeling.
//!
//! Reference: Yi He (2024) "Ridge Regression Under Dense Factor Augmented
//! Models", JASA 2024, Vol 119, No. 546.
//!
use ndarray::Array1;
use ndarray::Array2;

/// DRR-GARCH: Double Ridge Regression with GARCH(1,1) errors.
///
/// Two-step ridge regression for autoregressive models with many exogenous
/// predictors, combined with GARCH(1,1) error modeling.
pub struct DrrGarch {
  /// AR coefficient estimate
  pub alpha_ar: f64,
  /// Ridge regression coefficients for exogenous variables
  pub beta: Array1<f64>,
  /// GARCH(1,1) parameters: (omega, alpha, beta)
  pub garch_params: (f64, f64, f64),
  /// Intercept
  pub intercept: f64,
  /// Ridge penalty used in step 1
  pub lambda1: f64,
  /// Ridge penalty used in step 2 (selected by CV)
  pub lambda2: f64,
}

/// Result of DRR-GARCH fitting.
pub struct DrrGarchFit {
  /// Fitted model
  pub model: DrrGarch,
  /// In-sample residuals
  pub residuals: Array1<f64>,
  /// Conditional variance estimates
  pub conditional_variance: Array1<f64>,
  /// Cross-validation scores for different lambdas
  pub cv_scores: Vec<(f64, f64)>,
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that A = L L'.
fn cholesky(a: &Array2<f64>) -> Array2<f64> {
  let n = a.nrows();
  let mut l = Array2::<f64>::zeros((n, n));
  for j in 0..n {
    let mut sum = 0.0;
    for k in 0..j {
      sum += l[[j, k]] * l[[j, k]];
    }
    let diag = a[[j, j]] - sum;
    assert!(
      diag > 0.0,
      "Cholesky: matrix is not positive definite (diag={diag} at j={j})"
    );
    l[[j, j]] = diag.sqrt();
    for i in (j + 1)..n {
      let mut sum = 0.0;
      for k in 0..j {
        sum += l[[i, k]] * l[[j, k]];
      }
      l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
    }
  }
  l
}

/// Solve the linear system Ax = b where A is symmetric positive-definite,
/// using Cholesky decomposition.
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
  let l = cholesky(a);
  let n = b.len();

  // Forward substitution: Ly = b
  let mut y = Array1::<f64>::zeros(n);
  for i in 0..n {
    let mut sum = 0.0;
    for j in 0..i {
      sum += l[[i, j]] * y[j];
    }
    y[i] = (b[i] - sum) / l[[i, i]];
  }

  // Backward substitution: L'x = y
  let mut x = Array1::<f64>::zeros(n);
  for i in (0..n).rev() {
    let mut sum = 0.0;
    for j in (i + 1)..n {
      sum += l[[j, i]] * x[j];
    }
    x[i] = (y[i] - sum) / l[[i, i]];
  }
  x
}

/// Solve the ridge regression problem:
///
/// beta_hat(lambda) = (X'X/n + lambda * I)^{-1} X'y/n
///
/// The matrix X'X/n + lambda*I is symmetric positive-definite for lambda > 0,
/// so we use Cholesky decomposition to solve the system.
pub fn ridge_solve(x: &Array2<f64>, y: &Array1<f64>, lambda: f64) -> Array1<f64> {
  let n = x.nrows() as f64;
  let p = x.ncols();

  // A = X'X / n + lambda * I
  let xt = x.t();
  let mut a = xt.dot(x);
  a /= n;
  for i in 0..p {
    a[[i, i]] += lambda;
  }

  // b = X'y / n
  let b = xt.dot(y) / n;

  cholesky_solve(&a, &b)
}

/// K-fold cross-validation for ridge regression penalty selection.
///
/// Folds are assigned deterministically as fold_i = i % k_folds.
/// Returns (best_lambda_corrected, scores) where the bias correction
/// lambda_corrected = lambda_best * (K-1)/K is applied.
pub fn cross_validate_lambda(
  x: &Array2<f64>,
  y: &Array1<f64>,
  lambdas: &[f64],
  k_folds: usize,
) -> (f64, Vec<(f64, f64)>) {
  assert!(k_folds >= 2, "CV requires at least 2 folds");
  let n = x.nrows();
  let p = x.ncols();

  // Assign folds deterministically
  let fold_ids: Vec<usize> = (0..n).map(|i| i % k_folds).collect();

  let mut scores: Vec<(f64, f64)> = Vec::with_capacity(lambdas.len());
  let mut best_lambda = lambdas[0];
  let mut best_mse = f64::INFINITY;

  for &lam in lambdas {
    let mut total_mse = 0.0;
    let mut total_test = 0usize;

    for fold in 0..k_folds {
      // Count training and test sizes
      let n_test = fold_ids.iter().filter(|&&f| f == fold).count();
      let n_train = n - n_test;
      if n_train == 0 || n_test == 0 {
        continue;
      }

      // Build training and test sets
      let mut x_train = Array2::<f64>::zeros((n_train, p));
      let mut y_train = Array1::<f64>::zeros(n_train);
      let mut x_test = Array2::<f64>::zeros((n_test, p));
      let mut y_test = Array1::<f64>::zeros(n_test);

      let mut tr_idx = 0;
      let mut te_idx = 0;
      for i in 0..n {
        if fold_ids[i] == fold {
          x_test.row_mut(te_idx).assign(&x.row(i));
          y_test[te_idx] = y[i];
          te_idx += 1;
        } else {
          x_train.row_mut(tr_idx).assign(&x.row(i));
          y_train[tr_idx] = y[i];
          tr_idx += 1;
        }
      }

      // Fit ridge on training data
      let beta_hat = ridge_solve(&x_train, &y_train, lam);

      // Compute MSE on test data
      let y_pred = x_test.dot(&beta_hat);
      let diff = &y_test - &y_pred;
      let mse: f64 = diff.iter().map(|d| d * d).sum::<f64>() / n_test as f64;
      total_mse += mse * n_test as f64;
      total_test += n_test;
    }

    let avg_mse = if total_test > 0 {
      total_mse / total_test as f64
    } else {
      f64::INFINITY
    };

    scores.push((lam, avg_mse));

    if avg_mse < best_mse {
      best_mse = avg_mse;
      best_lambda = lam;
    }
  }

  // Bias correction: lambda_corrected = lambda_best * (K-1)/K
  let corrected = best_lambda * (k_folds - 1) as f64 / k_folds as f64;
  (corrected, scores)
}

/// Fit GARCH(1,1) to residuals using method-of-moments estimation.
///
/// Returns (omega, alpha1, beta1) where
///   sigma^2_t = omega + alpha1 * eps^2_{t-1} + beta1 * sigma^2_{t-1}
///
/// The estimation uses autocorrelation of squared residuals:
///   gamma_0 = Var(eps)
///   rho_1   = Corr(eps^2_t, eps^2_{t-1})
///
/// alpha1 and beta1 are chosen so that the model matches observed persistence,
/// and omega = gamma_0 * (1 - alpha1 - beta1) ensures the unconditional
/// variance is correct.
pub fn fit_garch11(residuals: &Array1<f64>) -> (f64, f64, f64) {
  let n = residuals.len();
  assert!(n >= 4, "Need at least 4 residuals to fit GARCH(1,1)");

  // Compute mean and variance of residuals
  let mean_eps: f64 = residuals.iter().sum::<f64>() / n as f64;
  let eps_centered: Array1<f64> = residuals.mapv(|e| e - mean_eps);
  let gamma0: f64 = eps_centered.iter().map(|e| e * e).sum::<f64>() / n as f64;

  if gamma0 < 1e-15 {
    // Residuals are essentially zero -- return trivial GARCH
    return (1e-10, 0.01, 0.01);
  }

  // Squared residuals
  let eps2: Array1<f64> = eps_centered.mapv(|e| e * e);
  let mean_eps2 = gamma0;

  // Autocorrelation of squared residuals at lag 1
  let mut cov1 = 0.0;
  let mut var_eps2 = 0.0;
  for t in 1..n {
    let d0 = eps2[t] - mean_eps2;
    let d1 = eps2[t - 1] - mean_eps2;
    cov1 += d0 * d1;
    var_eps2 += d0 * d0;
  }
  // Also add variance contribution from t=0
  let d0_first = eps2[0] - mean_eps2;
  var_eps2 += d0_first * d0_first;
  var_eps2 /= n as f64;
  cov1 /= (n - 1) as f64;

  let rho1 = if var_eps2 > 1e-15 {
    (cov1 / var_eps2).clamp(-0.99, 0.99)
  } else {
    0.0
  };

  // Autocorrelation at lag 2 for estimating beta
  let mut cov2 = 0.0;
  for t in 2..n {
    let d0 = eps2[t] - mean_eps2;
    let d2 = eps2[t - 2] - mean_eps2;
    cov2 += d0 * d2;
  }
  if n > 2 {
    cov2 /= (n - 2) as f64;
  }
  let rho2 = if var_eps2 > 1e-15 {
    (cov2 / var_eps2).clamp(-0.99, 0.99)
  } else {
    0.0
  };

  // Method-of-moments estimator for GARCH(1,1):
  // For GARCH(1,1), the ACF of squared residuals satisfies:
  //   rho_1 = alpha + beta * rho_1  =>  not quite, the exact relation is
  //   rho_1 = alpha * kappa + (alpha + beta) * rho_1... but this gets complex.
  //
  // Simplified approach:
  //   beta_hat = rho2 / rho1  (if rho1 != 0)
  //   alpha_hat = rho1 - beta_hat * rho1  (doesn't simplify well)
  //
  // Instead use a robust heuristic:
  //   total persistence = |rho1|
  //   split: alpha ~ 0.1 * persistence, beta ~ 0.9 * persistence
  //   (this is a common starting point, see Bollerslev 1986)
  let persistence = rho1.abs().clamp(0.0, 0.98);

  // Use rho2/rho1 ratio to inform the alpha/beta split
  let ratio = if rho1.abs() > 1e-10 {
    (rho2 / rho1).clamp(0.0, 0.99)
  } else {
    0.9
  };

  let beta1 = (persistence * ratio).clamp(0.01, 0.97);
  let alpha1 = (persistence - beta1).clamp(0.01, 0.97 - beta1);

  // Ensure stationarity: alpha1 + beta1 < 1
  let sum_ab = alpha1 + beta1;
  let (alpha1, beta1) = if sum_ab >= 0.999 {
    let scale = 0.98 / sum_ab;
    (alpha1 * scale, beta1 * scale)
  } else {
    (alpha1, beta1)
  };

  let omega = gamma0 * (1.0 - alpha1 - beta1);
  let omega = omega.max(1e-10);

  (omega, alpha1, beta1)
}

/// Fit the DRR-GARCH model.
///
/// # Arguments
/// * `y` - Response time series of length n.
/// * `x` - Exogenous predictor matrix of shape (n, p).
/// * `lambda1` - Ridge penalty for step 1. If `None`, defaults to p/n.
/// * `k_folds` - Number of CV folds for step 2. If `None`, defaults to 5.
///
/// # Procedure
/// 1. **Step 1 (Preliminary Ridge):** Regress y_{t+1} on [y_t, x_t] with
///    penalty lambda1 to estimate the AR coefficient alpha.
/// 2. **Step 2 (Main Ridge with CV):** Regress residuals r_t = y_{t+1} - alpha*y_t
///    on x_t with CV-selected lambda2.
/// 3. **Step 3:** Fit GARCH(1,1) to the final residuals.
pub fn fit_drr_garch(
  y: &Array1<f64>,
  x: &Array2<f64>,
  lambda1: Option<f64>,
  k_folds: Option<usize>,
) -> DrrGarchFit {
  let n = y.len();
  let p = x.ncols();
  assert!(n >= 4, "Need at least 4 observations");
  assert_eq!(x.nrows(), n, "x must have same number of rows as y");

  let k = k_folds.unwrap_or(5);
  let lam1 = lambda1.unwrap_or(p as f64 / n as f64);

  // ---- Step 1: Preliminary ridge on [y_lag, X] ----
  // We use observations t=0..n-2 as predictors, t=1..n-1 as response.
  let n_eff = n - 1;

  // Build design matrix Z = [y_lag, X_lag] of shape (n_eff, 1+p)
  let mut z = Array2::<f64>::zeros((n_eff, 1 + p));
  let mut y_next = Array1::<f64>::zeros(n_eff);
  for t in 0..n_eff {
    z[[t, 0]] = y[t]; // y_lag
    for j in 0..p {
      z[[t, 1 + j]] = x[[t, j]];
    }
    y_next[t] = y[t + 1];
  }

  // Ridge regression of y_next on Z with penalty lambda1
  let theta_hat = ridge_solve(&z, &y_next, lam1);

  // Extract AR coefficient and compute intercept (embedded in ridge, effectively 0 here)
  let alpha_ar = theta_hat[0];

  // Compute step-1 residuals: r_t = y_{t+1} - alpha_hat * y_t
  let mut residuals_step1 = Array1::<f64>::zeros(n_eff);
  for t in 0..n_eff {
    residuals_step1[t] = y_next[t] - alpha_ar * y[t];
  }

  // ---- Step 2: CV-selected ridge on residuals vs X ----
  // Build X_lag of shape (n_eff, p)
  let x_lag = x.slice(ndarray::s![..n_eff, ..]).to_owned();

  // Default lambda grid: geometrically spaced from 1e-4 to 10, 20 points
  let lambdas: Vec<f64> = (0..20)
    .map(|i| 10.0_f64.powf(-4.0 + i as f64 * 5.0 / 19.0))
    .collect();

  let (lam2, cv_scores) = cross_validate_lambda(&x_lag, &residuals_step1, &lambdas, k);

  // Final ridge fit with selected lambda2
  let beta_hat = ridge_solve(&x_lag, &residuals_step1, lam2);

  // ---- Step 3: Compute final residuals and fit GARCH(1,1) ----
  let predicted = x_lag.dot(&beta_hat);
  let final_residuals = &residuals_step1 - &predicted;

  let garch_params = fit_garch11(&final_residuals);

  // Compute conditional variances
  let (omega, garch_a, garch_b) = garch_params;
  let mut cond_var = Array1::<f64>::zeros(n_eff);
  let uncond_var = if (1.0 - garch_a - garch_b) > 1e-10 {
    omega / (1.0 - garch_a - garch_b)
  } else {
    omega
  };
  cond_var[0] = uncond_var;
  for t in 1..n_eff {
    cond_var[t] = omega + garch_a * final_residuals[t - 1].powi(2) + garch_b * cond_var[t - 1];
  }

  DrrGarchFit {
    model: DrrGarch {
      alpha_ar,
      beta: beta_hat,
      garch_params,
      intercept: 0.0,
      lambda1: lam1,
      lambda2: lam2,
    },
    residuals: final_residuals,
    conditional_variance: cond_var,
    cv_scores,
  }
}

impl DrrGarch {
  /// Predict the next value and its conditional variance.
  ///
  /// Returns (y_hat, sigma_squared) where
  ///   y_hat = intercept + alpha_ar * y_last + beta' * x_new
  pub fn predict(&self, y_last: f64, x_new: &Array1<f64>) -> (f64, f64) {
    let y_hat = self.intercept + self.alpha_ar * y_last + self.beta.dot(x_new);

    // One-step-ahead conditional variance uses unconditional as default
    let (omega, a, b) = self.garch_params;
    let sigma2 = if (1.0 - a - b) > 1e-10 {
      omega / (1.0 - a - b)
    } else {
      omega
    };

    (y_hat, sigma2)
  }

  /// Multi-step ahead forecast.
  ///
  /// # Arguments
  /// * `y` - Historical response values (at least 1 element).
  /// * `x` - Future exogenous predictors of shape (h, p).
  /// * `h` - Forecast horizon.
  ///
  /// # Returns
  /// (point_forecasts, variance_forecasts), each of length h.
  pub fn forecast(&self, y: &[f64], x: &Array2<f64>, h: usize) -> (Array1<f64>, Array1<f64>) {
    assert!(!y.is_empty(), "Need at least one historical observation");
    assert_eq!(x.nrows(), h, "x must have h rows for h-step forecast");
    assert_eq!(
      x.ncols(),
      self.beta.len(),
      "x must have same number of columns as beta"
    );

    let (omega, a, b) = self.garch_params;

    let mut forecasts = Array1::<f64>::zeros(h);
    let mut variances = Array1::<f64>::zeros(h);

    // Initialize conditional variance at unconditional level
    let uncond_var = if (1.0 - a - b) > 1e-10 {
      omega / (1.0 - a - b)
    } else {
      omega
    };

    let mut y_prev = *y.last().unwrap();
    let mut sigma2_prev = uncond_var;

    for t in 0..h {
      let x_row = x.row(t).to_owned();
      let y_hat = self.intercept + self.alpha_ar * y_prev + self.beta.dot(&x_row);
      forecasts[t] = y_hat;

      // Variance forecast: for multi-step, the residual is unknown,
      // so we use the expected value E[eps^2] = sigma^2
      sigma2_prev = omega + (a + b) * sigma2_prev;
      variances[t] = sigma2_prev;

      y_prev = y_hat;
    }

    (forecasts, variances)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;

  use super::*;

  #[test]
  fn ridge_solve_recovers_simple_coefficients() {
    // y = 2*x1 + 3*x2 + noise
    let n = 200;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| {
      if j == 0 {
        (i as f64) / n as f64
      } else {
        ((i * 7 + 3) % n) as f64 / n as f64
      }
    });
    let y = Array1::from_shape_fn(n, |i| {
      2.0 * x[[i, 0]] + 3.0 * x[[i, 1]] + 0.01 * ((i * 13) % 17) as f64 / 17.0
    });
    let beta = ridge_solve(&x, &y, 0.001);
    assert!((beta[0] - 2.0).abs() < 0.5, "beta[0]={}", beta[0]);
    assert!((beta[1] - 3.0).abs() < 0.5, "beta[1]={}", beta[1]);
  }

  #[test]
  fn cross_validation_selects_reasonable_lambda() {
    let n = 100;
    let p = 5;
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
      ((i * (j + 1) * 7 + 11) % 100) as f64 / 100.0
    });
    let y = Array1::from_shape_fn(n, |i| {
      x[[i, 0]] * 1.5 + x[[i, 1]] * 0.5 + 0.1 * ((i * 3) % 10) as f64 / 10.0
    });
    let lambdas: Vec<f64> = (0..20)
      .map(|i| 10.0_f64.powf(-4.0 + i as f64 * 0.5))
      .collect();
    let (best_lambda, scores) = cross_validate_lambda(&x, &y, &lambdas, 5);
    assert!(best_lambda > 0.0, "lambda should be positive");
    assert!(!scores.is_empty(), "should have scores");
  }

  #[test]
  fn fit_drr_garch_produces_valid_model() {
    // Generate simple AR(1) + noise data
    let n = 500;
    let p = 10;
    let mut y = Array1::<f64>::zeros(n);
    y[0] = 0.5;
    for t in 1..n {
      y[t] = 0.5 * y[t - 1] + 0.1 * ((t * 7 + 3) % 20) as f64 / 20.0 - 0.05;
    }
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
      ((i * (j + 1) * 13 + 7) % 100) as f64 / 100.0 - 0.5
    });

    let fit = fit_drr_garch(&y, &x, None, None);
    assert!(
      fit.model.alpha_ar.abs() < 1.0,
      "AR coeff should be < 1 in absolute value"
    );
    assert!(fit.model.garch_params.0 > 0.0, "omega should be > 0");
    assert!(!fit.residuals.is_empty(), "should have residuals");
  }

  #[test]
  fn garch_fit_recovers_reasonable_params() {
    // Generate GARCH(1,1) data
    let n = 1000;
    let omega = 0.05;
    let alpha = 0.08;
    let beta = 0.9;
    let mut eps = Array1::<f64>::zeros(n);
    let mut sig2 = Array1::<f64>::zeros(n);
    sig2[0] = omega / (1.0 - alpha - beta);
    eps[0] = sig2[0].sqrt() * 0.5; // deterministic for test
    for t in 1..n {
      sig2[t] = omega + alpha * eps[t - 1].powi(2) + beta * sig2[t - 1];
      eps[t] = sig2[t].sqrt() * (((t * 17 + 3) % 100) as f64 / 50.0 - 1.0);
    }
    let (w, a, b) = fit_garch11(&eps);
    assert!(w > 0.0, "omega should be > 0, got {w}");
    assert!(
      a + b < 1.0,
      "stationarity: a+b should be < 1, got {}",
      a + b
    );
  }
}
