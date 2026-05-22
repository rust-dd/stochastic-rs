use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::params::SsviParams;
use super::params::SsviSlice;
use crate::traits::FloatExt;

/// Calibrate SSVI global parameters $(\rho, \eta, \gamma)$ to multiple
/// maturity slices simultaneously.
pub fn calibrate_ssvi<T: FloatExt>(
  slices: &[SsviSlice<T>],
  initial: Option<SsviParams<T>>,
) -> SsviParams<T> {
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

  let problem = SsviLmProblem {
    slices: slices_f64,
    params: init_f64.into_dvector(),
  };

  let (result, _report) = LevenbergMarquardt::new()
    .with_patience(200)
    .with_tol(1e-12)
    .minimize(problem);

  let mut p64 = SsviParams::<f64>::from_dvector(&result.params);
  p64.project();

  SsviParams {
    rho: T::from_f64_fast(p64.rho),
    eta: T::from_f64_fast(p64.eta),
    gamma: T::from_f64_fast(p64.gamma),
  }
}

#[derive(Clone, Debug)]
struct SsviSliceF64 {
  log_moneyness: Vec<f64>,
  total_variance: Vec<f64>,
  theta: f64,
}

struct SsviLmProblem {
  slices: Vec<SsviSliceF64>,
  params: DVector<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for SsviLmProblem {
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;
  type JacobianStorage = Owned<f64, Dyn, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params.copy_from(params);
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let p = SsviParams::<f64>::from_dvector(&self.params);
    let n: usize = self.slices.iter().map(|s| s.log_moneyness.len()).sum();
    let mut r = DVector::zeros(n);
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
  /// Replaces the rc.0/rc.1 1-sided forward-difference Jacobian (`h = 1e-7`,
  /// O(h) error + 3 extra `total_variance` evaluations per data point) with
  /// the analytic derivative — O(eps) error, no extra evals.
  ///
  /// Let $\phi(\theta) = \eta \theta^{-\gamma}$ and
  /// $r = \sqrt{(\phi k + \rho)^2 + 1 - \rho^2}$. Then:
  ///
  /// $$
  /// \begin{aligned}
  /// \partial_{\rho} w &= \tfrac{\theta}{2}\left(\phi k + \tfrac{\phi k}{r}\right) \\
  /// \partial_{\phi} w &= \tfrac{\theta k}{2}\left(\rho + \tfrac{\phi k + \rho}{r}\right) \\
  /// \partial_{\eta} w &= \partial_{\phi} w \cdot \phi / \eta \\
  /// \partial_{\gamma} w &= -\partial_{\phi} w \cdot \phi \ln \theta.
  /// \end{aligned}
  /// $$
  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let p = SsviParams::<f64>::from_dvector(&self.params);
    let n: usize = self.slices.iter().map(|s| s.log_moneyness.len()).sum();
    let mut jac = DMatrix::zeros(n, 3);

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
      let phi = p.eta * theta.powf(-p.gamma);
      let log_theta = theta.ln();

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
        let dw_dgamma = -dw_dphi * phi * log_theta;

        jac[(idx, 0)] = dw_drho;
        jac[(idx, 1)] = dw_deta;
        jac[(idx, 2)] = dw_dgamma;

        idx += 1;
      }
    }

    Some(jac)
  }
}
