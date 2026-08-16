//! # GjrGarch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^p(\alpha_i+\gamma_i\mathbf 1_{\{X_{t-i}<0\}})X_{t-i}^2
//! +\sum_{j=1}^q\beta_j\sigma_{t-j}^2
//! $$
//!
//! This is the variance-level threshold recursion of Glosten,
//! Jagannathan, Runkle (1993) — commonly called "GJR-GARCH" — not
//! Zakoian's (1994) original TGARCH, which thresholds the conditional
//! *standard deviation* `sigma_t` rather than `sigma_t^2`. The two
//! specifications are not algebraically equivalent. This type was
//! previously named `Tgarch`, a name that described Zakoian's model
//! rather than this one; the old name is kept as a deprecated alias (see
//! [`Tgarch`]), but the type itself has always implemented the GJR
//! recursion above. Zakoian's own thresholded-standard-deviation
//! recursion is not implemented under either name.
//!
//! References:
//! - Glosten L. R., Jagannathan R., Runkle D. E. (1993) — *On the
//!   Relation between the Expected Value and the Volatility of the
//!   Nominal Excess Return on Stocks*, Journal of Finance 48(5),
//!   1779–1801, DOI: 10.1111/j.1540-6261.1993.tb05128.x.
//! - Zakoian J.-M. (1994) — *Threshold Heteroskedastic Models*, Journal
//!   of Economic Dynamics and Control 18(5), 931–955,
//!   DOI: 10.1016/0165-1889(94)90039-6.
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Implements the Glosten-Jagannathan-Runkle (1993) GJR-GARCH(p,q) model:
///
/// \[
///   \sigma_t^2
///     = \omega
///       + \sum_{i=1}^p \Bigl[\alpha_i X_{t-i}^2
///                              + \gamma_i X_{t-i}^2 \mathbf{1}_{\{X_{t-i}<0\}}\Bigr]
///       + \sum_{j=1}^q \beta_j \sigma_{t-j}^2,
///   \quad X_t = \sigma_t \cdot z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Parameters
/// - `omega`: Constant term (\(\omega\)).
/// - `alpha`: Array \(\{\alpha_1, \ldots, \alpha_p\}\) for the positive part of squared residuals.
/// - `gamma`: Array \(\{\gamma_1, \ldots, \gamma_p\}\) for the threshold effect (negative residuals).
///   Must have the same length as `alpha`.
/// - `beta`:  Array \(\{\beta_1, \ldots, \beta_q\}\) for the past variance terms.
/// - `n`:     Length of the time series to generate.
/// - `m`:     Optional batch size (unused by default).
///
/// # Notes
/// - Stationarity constraints typically include: \(\sum \alpha_i + \tfrac{1}{2}\sum \gamma_i + \sum \beta_j < 1\).
/// - We do a simple unconditional variance initialization for \(\sigma_0^2\).
#[derive(Debug, Clone)]
pub struct GjrGarch<T: FloatExt, S: SeedExt = Unseeded> {
  /// Constant term in conditional variance dynamics.
  pub omega: T,
  /// Arch coefficients α_i (positive-squared-residual loading), length p.
  pub alpha: Array1<T>,
  /// Threshold extra loading γ_i applied when `X_{t-i}<0`, length p
  /// (must match `alpha.len()`).
  pub gamma: Array1<T>,
  /// Garch coefficients β_j (past-variance persistence loading), length q.
  pub beta: Array1<T>,
  /// Length of the generated time series.
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> GjrGarch<T, S> {
  pub fn new(
    omega: T,
    alpha: Array1<T>,
    gamma: Array1<T>,
    beta: Array1<T>,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(omega > T::zero(), "GjrGarch requires omega > 0");
    assert!(
      alpha.len() == gamma.len(),
      "GjrGarch requires alpha.len() == gamma.len()"
    );
    Self {
      omega,
      alpha,
      gamma,
      beta,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for GjrGarch<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GjrGarchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GjrGarchSampler<T> {
    GjrGarchSampler {
      n: self.n,
      omega: self.omega,
      alpha: self.alpha.clone(),
      gamma: self.gamma.clone(),
      beta: self.beta.clone(),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }
}

/// Reusable [`GjrGarch`] sampling state: owns the standard-normal innovation
/// source and the variance coefficients so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct GjrGarchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  gamma: Array1<T>,
  beta: Array1<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GjrGarchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice(slice);
    }

    // Scratch array for sigma_t^2 (the output buffer holds X_t)
    let mut sigma2 = Array1::<T>::zeros(n);
    let var_floor = T::from_f64_fast(1e-12);

    // Sum up alpha + 0.5 gamma + beta for unconditional variance approximation
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_gamma_half = self.gamma.iter().cloned().sum::<T>() * T::from_f64_fast(0.5);
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = T::one() - sum_alpha - sum_gamma_half - sum_beta;
    assert!(
      denom > T::zero(),
      "GjrGarch requires sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1 for finite unconditional variance"
    );

    for t in 0..n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum over p lags
        for i in 1..=p {
          if t >= i {
            let x_lag = out[t - i];
            // Threshold indicator
            let indicator = if x_lag < T::zero() {
              T::one()
            } else {
              T::zero()
            };

            // alpha_i * X_{t-i}^2 + gamma_i * X_{t-i}^2 * indicator
            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.gamma[i - 1] * x_lag.powi(2) * indicator;
          }
        }

        // Sum over q lags
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }

        sigma2[t] = var_t;
      }
      assert!(
        sigma2[t].is_finite() && sigma2[t] > T::zero(),
        "GjrGarch produced non-positive or non-finite conditional variance at t={}",
        t
      );
      // X_t = sigma_t * z_t
      out[t] = sigma2[t].max(var_floor).sqrt() * z[t];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for GjrGarchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("GjrGarch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

/// Deprecated alias for [`GjrGarch`]. The name `Tgarch` suggested Zakoian's
/// (1994) TGARCH, but this type has always implemented the
/// variance-threshold recursion of Glosten, Jagannathan, Runkle (1993)
/// instead — a different, non-algebraically-equivalent model (see the
/// module docs above). Zakoian's own model is not implemented in this
/// crate.
#[deprecated(
  since = "2.7.0",
  note = "renamed to `GjrGarch`: this type implements Glosten-Jagannathan-Runkle (1993), not Zakoian's (1994) TGARCH"
)]
pub use GjrGarch as Tgarch;

py_process_1d!(PyTgarch, GjrGarch,
  sig: (omega, alpha, gamma_, beta, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, gamma_: Vec<f64>, beta: Vec<f64>, n: usize)
);
