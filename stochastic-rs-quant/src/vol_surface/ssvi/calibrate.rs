use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::params::SsviParams;
use super::params::SsviSlice;
use crate::LossMetric;
use crate::calibration::Calibrated;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::LmOptions;
use crate::calibration::least_squares::minimize;
use crate::traits::RealExt;

/// SSVI's global $(\rho, \eta, \gamma)$ fitted to every slice at once by Levenberg–Marquardt; the
/// loss is in total variance, and a parameter the projection moves stops the fit at its bound.
pub fn calibrate_ssvi<T: RealExt>(
  slices: &[SsviSlice<T>],
  initial: Option<SsviParams<T>>,
) -> Calibrated<SsviParams<T>> {
  let init_f64 = initial
    .map(|p| p.as_f64())
    .unwrap_or(SsviParams::<f64>::new(-0.3, 0.5, 0.5));

  let slices_f64: Vec<SsviSliceF64> = slices
    .iter()
    .map(|s| SsviSliceF64 {
      log_moneyness: s
        .log_moneyness
        .iter()
        .map(|x| x.to_f64().unwrap_or(0.0))
        .collect(),
      total_variance: s
        .total_variance
        .iter()
        .map(|x| x.to_f64().unwrap_or(0.0))
        .collect(),
      theta: s.theta.to_f64().unwrap_or(0.0),
    })
    .collect();

  let theta_max = f64::max_or_nan_fold(
    slices_f64
      .iter()
      .map(|s| s.theta)
      .filter(|t| t.is_finite() && *t > 0.0),
  )
  .max_or_nan(0.0);

  let problem = SsviLmProblem {
    slices: slices_f64,
    params: init_f64.into_dvector(),
  };

  let (result, report) = minimize(
    problem,
    LmOptions {
      tolerance: Some(1e-12),
      patience: 200,
      pivoted_qr: false,
      ..LmOptions::default()
    },
  );

  let mut p64 = SsviParams::<f64>::from_dvector(result.params.view());
  let unprojected = p64.into_dvector();
  // Enforce Gatheral-Jacquier 2014 power-law calendar-spread-free admissibility
  // (eq.4.4) over the calibration's observed θ range. Falls back to butterfly-only
  // projection when no positive θ is available (degenerate input).
  if theta_max > 0.0 {
    p64.project_with_theta_range(theta_max);
  } else {
    p64.project();
  }

  let slices = result.slices.iter();
  let points = slices.flat_map(|s| {
    s.log_moneyness
      .iter()
      .zip(&s.total_variance)
      .map(|p| (s.theta, p))
  });
  let (market, model): (Vec<f64>, Vec<f64>) = points
    .map(|(theta, (&k, &w))| (w, p64.total_variance(k, theta)))
    .unzip();
  let params = SsviParams {
    rho: T::from_f64_fast(p64.rho),
    eta: T::from_f64_fast(p64.eta),
    gamma: T::from_f64_fast(p64.gamma),
  };
  let (before, after) = (unprojected.to_vec(), p64.into_dvector().to_vec());
  report
    .fit(params, &market, &model, &LossMetric::ALL, Vec::new())
    .projected(&["rho", "eta", "gamma"], &before, &after)
}

#[derive(Clone, Debug)]
struct SsviSliceF64 {
  log_moneyness: Vec<f64>,
  total_variance: Vec<f64>,
  theta: f64,
}

struct SsviLmProblem {
  slices: Vec<SsviSliceF64>,
  params: Array1<f64>,
}

impl LeastSquaresProblem for SsviLmProblem {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    self.params.assign(&params);
  }

  fn params(&self) -> Array1<f64> {
    self.params.clone()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let p = SsviParams::<f64>::from_dvector(self.params.view());
    let n: usize = self.slices.iter().map(|s| s.log_moneyness.len()).sum();
    let mut r = Array1::zeros(n);
    let mut idx = 0;
    for slice in &self.slices {
      for i in 0..slice.log_moneyness.len() {
        r[idx] = p.total_variance(slice.log_moneyness[i], slice.theta) - slice.total_variance[i];
        idx += 1;
      }
    }
    Some(r)
  }

  /// Closed-form Jacobian of the SSVI total-variance residual w.r.t. the global
  /// parameters $(\rho, \eta, \gamma)$.
  ///
  /// Let $\phi(\theta) = \eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}$ and
  /// $r = \sqrt{(\phi k + \rho)^2 + 1 - \rho^2}$. Then:
  ///
  /// $$
  /// \begin{aligned}
  /// \partial_{\rho} w &= \tfrac{\theta}{2}\left(\phi k + \tfrac{\phi k}{r}\right) \\
  /// \partial_{\phi} w &= \tfrac{\theta k}{2}\left(\rho + \tfrac{\phi k + \rho}{r}\right) \\
  /// \partial_{\eta} w &= \partial_{\phi} w \cdot \phi / \eta \\
  /// \partial_{\gamma} w &= -\partial_{\phi} w \cdot \phi \ln\frac{\theta}{1+\theta}.
  /// \end{aligned}
  /// $$
  fn jacobian(&self) -> Option<Array2<f64>> {
    let p = SsviParams::<f64>::from_dvector(self.params.view());
    let n: usize = self.slices.iter().map(|s| s.log_moneyness.len()).sum();
    let mut jac = Array2::zeros((n, 3));

    let mut idx = 0;
    for slice in &self.slices {
      let theta = slice.theta;
      if theta <= 0.0 || !theta.is_finite() {
        for _ in 0..slice.log_moneyness.len() {
          jac[(idx, 0)] = 0.0;
          jac[(idx, 1)] = 0.0;
          jac[(idx, 2)] = 0.0;
          idx += 1;
        }
        continue;
      }
      let phi = p.phi(theta);
      let log_ratio = (theta / (1.0 + theta)).ln();

      for i in 0..slice.log_moneyness.len() {
        let k = slice.log_moneyness[i];
        let pk_plus_rho = phi * k + p.rho;
        let r2 = pk_plus_rho * pk_plus_rho + 1.0 - p.rho * p.rho;
        let r = r2.max(1e-30).sqrt();

        let dw_drho = 0.5 * theta * (phi * k + (phi * k) / r);
        let dw_dphi = 0.5 * theta * k * (p.rho + pk_plus_rho / r);

        let dw_deta = if p.eta.abs() > 1e-30 {
          dw_dphi * phi / p.eta
        } else {
          0.0
        };
        let dw_dgamma = -dw_dphi * phi * log_ratio;

        jac[(idx, 0)] = dw_drho;
        jac[(idx, 1)] = dw_deta;
        jac[(idx, 2)] = dw_dgamma;

        idx += 1;
      }
    }

    Some(jac)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::calibration::least_squares::checks::jacobian_error;

  /// The unconstrained SSVI Jacobian at a start, at the truth and at a far point.
  #[test]
  fn the_jacobian_is_the_slope_of_the_residuals() {
    let truth = SsviParams::<f64> {
      rho: -0.4,
      eta: 1.2,
      gamma: 0.4,
    };
    let log_moneyness = vec![-0.3, -0.1, 0.0, 0.1, 0.3];
    let slices = [0.02, 0.05, 0.12]
      .into_iter()
      .map(|theta| SsviSliceF64 {
        total_variance: log_moneyness
          .iter()
          .map(|&k| truth.total_variance(k, theta))
          .collect(),
        log_moneyness: log_moneyness.clone(),
        theta,
      })
      .collect();
    let params = truth.into_dvector();
    let mut problem = SsviLmProblem { slices, params };
    let points = [
      ndarray::array![0.0, 1.0, 0.5],
      ndarray::array![-0.4, 1.2, 0.4],
      ndarray::array![0.9, 4.0, 0.9],
    ];
    for point in points {
      let error = jacobian_error(&mut problem, point.view(), |_, _| 1e-6);
      assert!(error < 1e-8, "{point}: {error:e}");
    }
  }
}
