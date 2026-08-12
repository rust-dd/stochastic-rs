//! `BivariateExt` — bivariate copula trait.

use std::cmp::Ordering;
use std::error::Error;

use ndarray::Array1;
use ndarray::Axis;
use ndarray::stack;
use roots::SimpleConvergency;
use roots::find_root_brent;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::bivariate::CopulaType as BivariateCopulaType;

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

  fn check_theta(&self) -> Result<(), String> {
    let (lower, upper) = self.theta_bounds();
    let theta = self
      .theta()
      .ok_or_else(|| "theta is not set; call set_theta or fit first".to_string())?;
    let invalid = self.invalid_thetas();

    if !(lower <= theta && theta <= upper) || invalid.contains(&theta) {
      return Err(format!(
        "Theta must be in the interval [{}, {}] and not in {:?}",
        lower, upper, invalid
      ));
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

  /// Archimedean generator $\varphi_\theta(t)$, satisfying $C(u,v) =
  /// \varphi^{-1}(\varphi(u) + \varphi(v))$. Overridden with a closed form
  /// by the six Archimedean families (AMH, Clayton, Frank, Gumbel,
  /// Independence, Joe); every other family has no Archimedean
  /// representation, so the default here — derived from `r#type()`'s
  /// `Debug` label — returns the anchored
  /// `"<Type> is not Archimedean — generator not defined"` without each
  /// family hand-writing an identical stub.
  fn generator(&self, _t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Err(
      format!(
        "{:?} is not Archimedean — generator not defined",
        self.r#type()
      )
      .into(),
    )
  }

  fn sample(&self, n: usize) -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
    self.sample_with_uniform(
      stochastic_rs_distributions::uniform::SimdUniform::<f64>::new(0.0, 1.0, &Unseeded),
      n,
    )
  }

  /// Deterministic sampler. Returns the same paths for a fixed `seed`.
  fn sample_with_seed(&self, n: usize, seed: u64) -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
    self.sample_with_uniform(
      stochastic_rs_distributions::uniform::SimdUniform::<f64>::new(
        0.0,
        1.0,
        &Deterministic::new(seed),
      ),
      n,
    )
  }

  #[doc(hidden)]
  fn sample_with_uniform(
    &self,
    ud: stochastic_rs_distributions::uniform::SimdUniform<f64>,
    n: usize,
  ) -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
    if self.tau().is_none() {
      return Err("Tau is not defined".into());
    }

    let tau = self.tau().unwrap();

    if !(-1.0..1.0).contains(&tau) {
      return Err("Tau must be in the interval (-1, 1)".into());
    }

    let mut v = Array1::<f64>::zeros(n);
    ud.fill_slice(v.as_slice_mut().unwrap());
    let mut c = Array1::<f64>::zeros(n);
    ud.fill_slice(c.as_slice_mut().unwrap());
    let u = self.percent_point(&c, &v)?;

    Ok(stack![Axis(1), u, v])
  }

  fn fit(&mut self, X: &ndarray::Array2<f64>) -> Result<(), Box<dyn Error>> {
    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    self.check_marginal(&U)?;
    self.check_marginal(&V)?;

    let (tau, ..) = kendalls::tau_b_with_comparator(&U.to_vec(), &V.to_vec(), |a, b| {
      a.partial_cmp(b).unwrap_or(Ordering::Greater)
    })?;

    self.set_tau(tau);
    self._compute_theta();

    Ok(())
  }

  fn check_fit(&self) -> Result<(), Box<dyn Error>> {
    if self.theta().is_none() {
      return Err("Fit the copula first".into());
    }

    self.check_theta()?;
    Ok(())
  }

  #[doc(hidden)]
  fn check_marginal(&self, u: &Array1<f64>) -> Result<(), String> {
    if !u.iter().all(|x| (0.0..=1.0).contains(x)) {
      return Err("Marginal values must be in the interval [0, 1]".into());
    }

    let mut empirical_cdf = u.to_vec();
    empirical_cdf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let empirical_cdf = Array1::from(empirical_cdf);
    let uniform = Array1::linspace(0.0, 1.0, u.len());
    let ks = (empirical_cdf - uniform).fold(0.0_f64, |acc, &d| acc.max(d.abs()));

    if ks > 1.627 / (u.len() as f64).sqrt() {
      return Err("Marginal values do not follow a uniform distribution".into());
    }

    Ok(())
  }

  fn pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  /// Inverse conditional: returns `u` such that `P(U ≤ u | V = v) = p`.
  /// This is the canonical quantile-function name for this trait; see also
  /// [`ppf`](Self::ppf), a SciPy-compatible alias.
  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.percent_point_numerical(y, V)
  }

  /// Brent-root-finding numerical inversion of
  /// [`partial_derivative_scalar`](Self::partial_derivative_scalar) that
  /// backs the default [`percent_point`](Self::percent_point). Exposed
  /// under its own name so a family that overrides `percent_point` (to
  /// special-case a degenerate parameter, say) has a way to fall back to
  /// this generic implementation — calling `Self::percent_point` from
  /// inside an override of that same method would just recurse into the
  /// override instead of reaching this body.
  fn percent_point_numerical(
    &self,
    y: &Array1<f64>,
    V: &Array1<f64>,
  ) -> Result<Array1<f64>, Box<dyn Error>> {
    let n = y.len();
    let mut results = Array1::zeros(n);

    for i in 0..n {
      let y_i = y[i];
      let v_i = V[i];

      let f = |u| self.partial_derivative_scalar(u, v_i).unwrap() - y_i;
      let mut convergency = SimpleConvergency {
        eps: f64::EPSILON,
        max_iter: 50,
      };
      let min = find_root_brent(f64::EPSILON, 1.0, f, &mut convergency);
      results[i] = min.unwrap_or(f64::EPSILON);
    }

    Ok(results)
  }

  /// `ppf` is a SciPy-compatible alias for [`percent_point`](Self::percent_point).
  fn ppf(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.percent_point(y, V)
  }

  fn partial_derivative(
    &self,
    X: &ndarray::Array2<f64>,
  ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let n = X.nrows();
    let mut X_prime = X.clone();
    let mut delta = Array1::zeros(n);
    for i in 0..n {
      delta[i] = if X[[i, 1]] > 0.5 { -0.0001 } else { 0.0001 };
      X_prime[[i, 1]] = X[[i, 1]] + delta[i];
    }

    let f = self.cdf(X).unwrap();
    let f_prime = self.cdf(&X_prime).unwrap();

    let mut deriv = Array1::zeros(n);
    for i in 0..n {
      deriv[i] = (f_prime[i] - f[i]) / delta[i];
    }

    Ok(deriv)
  }

  #[doc(hidden)]
  fn partial_derivative_scalar(&self, U: f64, V: f64) -> Result<f64, Box<dyn Error>> {
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
  use crate::bivariate::clayton::Clayton;

  /// Minimal non-Archimedean stand-in that does **not** override
  /// `generator` at all — proves the trait-default body (not any family's
  /// own hand-written stub) is what actually answers the call.
  struct DummyNonArchimedean;

  impl BivariateExt for DummyNonArchimedean {
    fn r#type(&self) -> BivariateCopulaType {
      BivariateCopulaType::Fgm
    }

    fn tau(&self) -> Option<f64> {
      None
    }

    fn set_tau(&mut self, _tau: f64) {}

    fn theta(&self) -> Option<f64> {
      None
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

    fn pdf(&self, _x: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
      Err("not implemented for DummyNonArchimedean".into())
    }

    fn cdf(&self, _x: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
      Err("not implemented for DummyNonArchimedean".into())
    }
  }

  /// `generator()`'s default body — reached here by a type that has no
  /// override whatsoever — returns the anchored "<Type> is not Archimedean
  /// — generator not defined" message built from `r#type()`'s `Debug`
  /// label, matching the pattern the 7 real non-Archimedean families used
  /// to hand-write individually.
  #[test]
  fn generator_default_returns_anchored_not_archimedean_err() {
    let dummy = DummyNonArchimedean;
    let t = array![0.5_f64, 0.8];
    let msg = dummy.generator(&t).unwrap_err().to_string();
    assert!(
      msg.contains("is not Archimedean — generator not defined"),
      "unexpected message: {msg}"
    );
    assert!(
      msg.starts_with("Fgm"),
      "expected the r#type() Debug label as prefix, got: {msg}"
    );
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
}
