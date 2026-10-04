//! `BivariateExt` — bivariate copula trait.

use std::cell::RefCell;
use std::cmp::Ordering;

use ndarray::Array1;
use ndarray::Axis;
use ndarray::stack;
use roots::SimpleConvergency;
use roots::find_root_brent;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::bivariate::CopulaType as BivariateCopulaType;
use crate::error::CopulaError;

/// Upper- and lower-tail dependence coefficients
/// $$
/// \lambda_L = \lim_{u\to0^+} \frac{C(u,u)}{u}, \qquad
/// \lambda_U = \lim_{u\to1^-} \frac{1 - 2u + C(u,u)}{1 - u}.
/// $$
/// Generic in `T` so the value type travels with the struct; every
/// [`BivariateExt`] impl in this crate is `f64`-based, so
/// [`BivariateExt::tail_dependence`] always returns `TailDependence<f64>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailDependence<T> {
  pub lower: T,
  pub upper: T,
}

pub trait BivariateExt {
  fn r#type(&self) -> BivariateCopulaType;

  fn tau(&self) -> Option<f64>;

  fn set_tau(&mut self, tau: f64);

  fn theta(&self) -> Option<f64>;

  fn theta_bounds(&self) -> (f64, f64);

  fn invalid_thetas(&self) -> Vec<f64>;

  fn set_theta(&mut self, theta: f64);

  fn check_theta(&self) -> Result<(), CopulaError> {
    let (lower, upper) = self.theta_bounds();
    let theta = self.theta().ok_or(CopulaError::NotFitted)?;
    let invalid = self.invalid_thetas();

    if !(lower <= theta && theta <= upper) || invalid.contains(&theta) {
      return Err(CopulaError::InvalidParameter {
        name: "theta",
        value: theta,
        constraint: format!("{lower} <= theta <= {upper}, theta not in {invalid:?}"),
      });
    }

    Ok(())
  }

  fn compute_theta(&self) -> f64;

  #[doc(hidden)]
  fn _compute_theta(&mut self) {
    self.set_theta(self.compute_theta());
    let _ = self.check_theta();
  }

  /// Closed-form upper/lower tail-dependence coefficients for the current
  /// `theta`. Required — not defaulted — because a silent `(0.0, 0.0)`
  /// fallback would be a correctness bug for every family with nonzero
  /// tail dependence (Clayton, Gumbel, Joe, Galambos, Hüsler-Reiss,
  /// Marshall-Olkin, Student-t). See each family's module doc for the
  /// formula and its source.
  ///
  /// # Panics
  ///
  /// Implementations panic with a message beginning `"tail_dependence
  /// requires a valid theta"` if the copula's shape parameter is unset or
  /// outside its valid domain — see
  /// [`BivariateExt::assert_theta_valid_for_tail_dependence`]. Every
  /// other formula-producing method (`pdf`/`cdf`/`partial_derivative`/
  /// `percent_point`) is gated by `check_fit()?`; `tail_dependence` has
  /// no `Result` to propagate through, so it panics instead. This matters
  /// because `_compute_theta` discards its own `check_theta()` result —
  /// `fit()` on data whose empirical tau falls outside a family's
  /// domain (e.g. negative tau for Gumbel) silently leaves `theta` out of
  /// bounds, and without this guard `tail_dependence` would silently
  /// return a nonsensical (even negative) coefficient.
  fn tail_dependence(&self) -> TailDependence<f64>;

  /// Panics with a message beginning `"tail_dependence requires a valid
  /// theta"` unless `theta` is set and satisfies
  /// [`BivariateExt::theta_bounds`] / [`BivariateExt::invalid_thetas`].
  /// Every [`BivariateExt::tail_dependence`] impl in this crate calls this
  /// first (or an equivalent family-specific check, for families like
  /// Marshall-Olkin that accept parameters outside the `theta` field).
  fn assert_theta_valid_for_tail_dependence(&self) {
    if let Err(e) = self.check_theta() {
      panic!("tail_dependence requires a valid theta: {e}");
    }
  }

  /// Archimedean generator $\varphi$, $C(u,v) = \varphi^{-1}(\varphi(u) + \varphi(v))$; the default
  /// is `Unsupported`, named by the `r#type()` label, for a family with no such form.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, CopulaError> {
    Err(CopulaError::Unsupported(format!(
      "{:?} is not Archimedean: no generator",
      self.r#type()
    )))
  }

  fn sample(&self, n: usize) -> Result<ndarray::Array2<f64>, CopulaError> {
    self.sample_with_uniform(SimdUniform::<f64>::new(0.0, 1.0).seeded(&Unseeded), n)
  }

  /// Deterministic sampler. Returns the same paths for a fixed `seed`.
  fn sample_with_seed(&self, n: usize, seed: u64) -> Result<ndarray::Array2<f64>, CopulaError> {
    self.sample_with_uniform(
      SimdUniform::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed)),
      n,
    )
  }

  #[doc(hidden)]
  fn sample_with_uniform(
    &self,
    mut ud: Seeded<SimdUniform<f64>>,
    n: usize,
  ) -> Result<ndarray::Array2<f64>, CopulaError> {
    // The family's `check_fit` is the gate: a copula parameterised outside `theta` passes it,
    // a `tau` without a `theta` does not.
    self.check_fit()?;

    let mut v = Array1::<f64>::zeros(n);
    ud.fill_slice(v.as_slice_mut().unwrap());
    let mut c = Array1::<f64>::zeros(n);
    ud.fill_slice(c.as_slice_mut().unwrap());
    let u = self.percent_point(&c, &v)?;

    Ok(stack![Axis(1), u, v])
  }

  fn fit(&mut self, X: &ndarray::Array2<f64>) -> Result<(), CopulaError> {
    if X.nrows() < 2 {
      return Err(CopulaError::InsufficientData {
        needed: 2,
        got: X.nrows(),
      });
    }
    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    self.check_marginal(&U)?;
    self.check_marginal(&V)?;

    let (tau, ..) = kendalls::tau_b_with_comparator(&U.to_vec(), &V.to_vec(), |a, b| {
      a.partial_cmp(b).unwrap_or(Ordering::Greater)
    })
    .map_err(|_| CopulaError::InsufficientData {
      needed: 2,
      got: U.len(),
    })?;

    self.set_tau(tau);
    self._compute_theta();

    Ok(())
  }

  fn check_fit(&self) -> Result<(), CopulaError> {
    if self.theta().is_none() {
      return Err(CopulaError::NotFitted);
    }

    self.check_theta()?;
    Ok(())
  }

  #[doc(hidden)]
  fn check_marginal(&self, u: &Array1<f64>) -> Result<(), CopulaError> {
    if !u.iter().all(|x| (0.0..=1.0).contains(x)) {
      return Err(CopulaError::MarginalOutOfRange);
    }

    let mut empirical_cdf = u.to_vec();
    empirical_cdf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let empirical_cdf = Array1::from(empirical_cdf);
    let uniform = Array1::linspace(0.0, 1.0, u.len());
    let ks = (empirical_cdf - uniform).fold(0.0_f64, |acc, &d| acc.max(d.abs()));

    if ks > 1.627 / (u.len() as f64).sqrt() {
      return Err(CopulaError::MarginalNotUniform);
    }

    Ok(())
  }

  fn pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError>;

  fn log_pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError>;

  /// Inverse conditional: returns `u` such that `P(U ≤ u | V = v) = p`.
  /// This is the canonical quantile-function name for this trait; see also
  /// [`ppf`](Self::ppf), a SciPy-compatible alias.
  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.percent_point_numerical(y, V)
  }

  /// Brent-root-finding numerical inversion of `partial_derivative_scalar`
  /// (`#[doc(hidden)]`, so not itself a doc link target) that backs the
  /// default [`percent_point`](Self::percent_point). Exposed
  /// under its own name so a family that overrides `percent_point` (to
  /// special-case a degenerate parameter, say) has a way to fall back to
  /// this generic implementation — calling `Self::percent_point` from
  /// inside an override of that same method would just recurse into the
  /// override instead of reaching this body.
  ///
  /// A quantile whose root lies at or below the bracket floor `f64::EPSILON`
  /// saturates to that floor — the answer is below resolution, which is a
  /// value, not a failure. A failing `partial_derivative` or a solve that
  /// does not converge is an `Err`: neither has a quantile to report, and a
  /// fabricated one would be indistinguishable from a real deep-tail draw.
  fn percent_point_numerical(
    &self,
    y: &Array1<f64>,
    V: &Array1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let n = y.len();
    let mut results = Array1::zeros(n);

    for i in 0..n {
      let y_i = y[i];
      let v_i = V[i];

      // The root-finder's closure cannot return a `Result`, so the first
      // inner failure is parked here and re-raised after the solve.
      let inner_err: RefCell<Option<CopulaError>> = RefCell::new(None);
      let h = |u: f64| match self.partial_derivative_scalar(u, v_i) {
        Ok(d) => d - y_i,
        Err(e) => {
          let mut slot = inner_err.borrow_mut();
          if slot.is_none() {
            *slot = Some(e);
          }
          f64::NAN
        }
      };

      let lo = f64::EPSILON;
      let f_lo = h(lo);
      if let Some(e) = inner_err.borrow_mut().take() {
        return Err(e);
      }
      if f_lo >= 0.0 {
        results[i] = lo;
        continue;
      }

      let mut convergency = SimpleConvergency {
        eps: f64::EPSILON,
        max_iter: 50,
      };
      let root = find_root_brent(lo, 1.0, h, &mut convergency);
      if let Some(e) = inner_err.borrow_mut().take() {
        return Err(e);
      }
      results[i] = root.map_err(|e| {
        CopulaError::Numerical(format!(
          "{:?} h-inverse did not converge at (y={y_i}, v={v_i}): {e:?}",
          self.r#type()
        ))
      })?;
    }

    Ok(results)
  }

  /// `ppf` is a SciPy-compatible alias for [`percent_point`](Self::percent_point).
  fn ppf(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.percent_point(y, V)
  }

  fn partial_derivative(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    let n = X.nrows();
    let mut X_prime = X.clone();
    let mut delta = Array1::zeros(n);
    for i in 0..n {
      delta[i] = if X[[i, 1]] > 0.5 { -0.0001 } else { 0.0001 };
      X_prime[[i, 1]] = X[[i, 1]] + delta[i];
    }

    let f = self.cdf(X)?;
    let f_prime = self.cdf(&X_prime)?;

    let mut deriv = Array1::zeros(n);
    for i in 0..n {
      deriv[i] = (f_prime[i] - f[i]) / delta[i];
    }

    Ok(deriv)
  }

  #[doc(hidden)]
  fn partial_derivative_scalar(&self, U: f64, V: f64) -> Result<f64, CopulaError> {
    self.check_fit()?;
    let X = stack![Axis(1), Array1::from(vec![U]), Array1::from(vec![V])];
    let out = self.partial_derivative(&X);

    Ok(*out?.get(0).unwrap())
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;
  use crate::bivariate::amh::Amh;
  use crate::bivariate::clayton::Clayton;

  /// Overrides no provided method, so every default body answers for it; `pdf` and `cdf` fail.
  struct DummyNonArchimedean {
    theta: Option<f64>,
  }

  impl BivariateExt for DummyNonArchimedean {
    fn r#type(&self) -> BivariateCopulaType {
      BivariateCopulaType::Fgm
    }

    fn tau(&self) -> Option<f64> {
      None
    }

    fn set_tau(&mut self, _tau: f64) {}

    fn theta(&self) -> Option<f64> {
      self.theta
    }

    fn theta_bounds(&self) -> (f64, f64) {
      (-1.0, 1.0)
    }

    fn invalid_thetas(&self) -> Vec<f64> {
      vec![]
    }

    fn set_theta(&mut self, _theta: f64) {}

    fn compute_theta(&self) -> f64 {
      0.0
    }

    fn tail_dependence(&self) -> TailDependence<f64> {
      TailDependence {
        lower: 0.0,
        upper: 0.0,
      }
    }

    fn pdf(&self, _x: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
      Err(CopulaError::Unsupported(
        "not implemented for DummyNonArchimedean".into(),
      ))
    }

    fn cdf(&self, _x: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
      Err(CopulaError::Unsupported(
        "not implemented for DummyNonArchimedean".into(),
      ))
    }
  }

  /// Parameterised outside `theta`, like `MarshallOlkin::with_alpha_beta`; the h-function is `u`.
  struct ParameterisedWithoutTheta;

  impl BivariateExt for ParameterisedWithoutTheta {
    fn r#type(&self) -> BivariateCopulaType {
      BivariateCopulaType::MarshallOlkin
    }

    fn tau(&self) -> Option<f64> {
      None
    }

    fn set_tau(&mut self, _tau: f64) {}

    fn theta(&self) -> Option<f64> {
      None
    }

    fn theta_bounds(&self) -> (f64, f64) {
      (0.0, 1.0)
    }

    fn invalid_thetas(&self) -> Vec<f64> {
      vec![]
    }

    fn set_theta(&mut self, _theta: f64) {}

    fn compute_theta(&self) -> f64 {
      0.0
    }

    fn tail_dependence(&self) -> TailDependence<f64> {
      TailDependence {
        lower: 0.0,
        upper: 0.0,
      }
    }

    fn check_fit(&self) -> Result<(), CopulaError> {
      Ok(())
    }

    fn pdf(&self, x: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
      Ok(Array1::ones(x.nrows()))
    }

    fn cdf(&self, x: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
      Ok(&x.column(0) * &x.column(1))
    }

    fn partial_derivative(&self, x: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
      Ok(x.column(0).to_owned())
    }
  }

  /// A type with no `generator` override reaches the trait default, which names its `r#type()`.
  #[test]
  fn generator_default_returns_anchored_not_archimedean_err() {
    let dummy = DummyNonArchimedean { theta: None };
    assert_eq!(
      dummy.generator(&array![0.5_f64, 0.8]).unwrap_err(),
      CopulaError::Unsupported("Fgm is not Archimedean: no generator".into())
    );
  }

  /// An unfitted copula is `NotFitted` before any solve, and a failing `partial_derivative` keeps
  /// its own variant: neither becomes a panic or a fabricated `EPSILON` quantile.
  #[test]
  fn percent_point_numerical_propagates_inner_errors_instead_of_panicking() {
    let (y, v) = (array![0.5_f64], array![0.5_f64]);
    let unfitted = DummyNonArchimedean { theta: None };
    assert_eq!(
      unfitted.percent_point_numerical(&y, &v).unwrap_err(),
      CopulaError::NotFitted
    );
    let fitted = DummyNonArchimedean { theta: Some(0.5) };
    assert_eq!(
      fitted.percent_point_numerical(&y, &v).unwrap_err(),
      CopulaError::Unsupported("not implemented for DummyNonArchimedean".into())
    );
  }

  /// A target below the bracket floor is a saturation, not a failure: the
  /// h-function of any copula at `u = EPSILON` is at least `y = 1e-300`, so
  /// the quantile is below resolution and the floor itself is the honest
  /// answer.
  #[test]
  fn percent_point_numerical_saturates_at_the_bracket_floor() {
    let mut amh = Amh::new();
    amh.set_tau(0.25);
    amh._compute_theta();
    let out = amh
      .percent_point_numerical(&array![1e-300_f64], &array![0.5_f64])
      .expect("saturation is a value");
    assert_eq!(
      out[0],
      f64::EPSILON,
      "expected the bracket floor, got {}",
      out[0]
    );
  }

  /// Round trip on the numerical path: `u = h^{-1}(y | v)` must satisfy
  /// `h(u | v) = y`. Amh has a closed-form `partial_derivative` but no
  /// `percent_point` override, so this exercises exactly the Brent body.
  #[test]
  fn percent_point_numerical_round_trips_through_the_h_function() {
    let mut amh = Amh::new();
    amh.set_tau(0.25);
    amh._compute_theta();
    for &(y, v) in &[(0.1_f64, 0.3_f64), (0.5, 0.5), (0.9, 0.7), (0.05, 0.95)] {
      let u = amh
        .percent_point_numerical(&array![y], &array![v])
        .expect("solve")[0];
      let back = amh.partial_derivative_scalar(u, v).expect("h eval");
      assert!(
        (back - y).abs() < 1e-9,
        "round trip failed: h(h_inv({y}|{v})|{v}) = {back}"
      );
    }
  }

  /// `#[doc(hidden)]` hides `_compute_theta` / `check_marginal` /
  /// `partial_derivative_scalar` from rendered docs but must not restrict
  /// who can call them — they stay reachable in-crate exactly like
  /// `sample_with_uniform` already was. Compile-guard test: this would
  /// stop compiling if any of the three ever became `pub(crate)` (or
  /// otherwise less visible) by mistake.
  #[test]
  fn doc_hidden_methods_remain_callable_in_crate() {
    let mut c = Clayton::new();
    c.set_tau(0.5);
    c._compute_theta();
    assert!(c.theta().is_some());

    let u = array![0.1_f64, 0.5, 0.9];
    assert!(c.check_marginal(&u).is_ok());

    let scalar = c.partial_derivative_scalar(0.3, 0.6).unwrap();
    assert!((0.0..=1.0).contains(&scalar));
  }

  #[test]
  fn sampling_needs_theta_not_tau() {
    let c = Clayton {
      theta: Some(2.0),
      ..Clayton::new()
    };
    let uv = c
      .sample_with_seed(1_000, 7)
      .expect("theta alone is enough to sample");
    assert_eq!(uv.dim(), (1_000, 2));

    assert_eq!(
      Clayton::new().sample_with_seed(10, 7).unwrap_err(),
      CopulaError::NotFitted
    );
  }

  #[test]
  fn sampling_gates_on_the_family_check_fit() {
    let uv = ParameterisedWithoutTheta
      .sample_with_seed(64, 7)
      .expect("check_fit passes without a theta");
    assert_eq!(uv.dim(), (64, 2));
    assert!(uv.iter().all(|x| (0.0..=1.0).contains(x)));
  }
}
