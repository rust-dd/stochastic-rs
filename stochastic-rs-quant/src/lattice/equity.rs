//! Cox–Ross–Rubinstein trees as a [`crate::traits::ModelPricer`], so they plug into the vol-surface
//! pipeline beside the Fourier and SABR pricers.

use ndarray::Array1;

use super::tree::BinomialTree;
use crate::OptionStyle;
use crate::OptionType;
use crate::traits::ModelPricer;
use crate::traits::RealExt;
use crate::traits::VanillaEuropeanCall;

/// Cox–Ross–Rubinstein binomial model for European and American options, on
/// $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and $p = (e^{(r-q)\Delta t} - d)/(u - d)$.
#[derive(Debug, Clone, Copy)]
pub struct CrrModel<T: RealExt> {
  /// Lognormal volatility $\sigma$.
  pub sigma: T,
  /// Number of tree steps.
  pub steps: usize,
}

impl<T: RealExt> CrrModel<T> {
  /// Panics unless `sigma` is finite and `>= 0` and `steps >= 1`.
  pub fn new(sigma: T, steps: usize) -> Self {
    assert!(
      sigma.is_finite() && sigma >= T::zero(),
      "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    assert!(
      steps >= 1,
      "steps must satisfy `steps >= 1`, got steps = {steps:?}"
    );
    Self { sigma, steps }
  }

  /// The rollback of both styles: the payoff at `tau = 0`; NaN past expiry, for a NaN query, a
  /// spot or strike that is not positive, and where `p` leaves `[0, 1]`, as it does at `sigma = 0`.
  ///
  /// $$
  /// V=e^{-r\Delta t}\bigl(pV_u+(1-p)V_d\bigr),\qquad
  /// V^{\mathrm{Am}}=\max\bigl(V,\ \text{intrinsic}\bigr),\qquad
  /// p=\frac{e^{(r-q)\Delta t}-d}{u-d}
  /// $$
  fn price_rollback(
    &self,
    s: T,
    k: T,
    r: T,
    q: T,
    tau: T,
    option_type: OptionType,
    option_style: OptionStyle,
  ) -> T {
    let intrinsic = |state: T| match option_type {
      OptionType::Call => (state - k).max_or_nan(T::zero()),
      OptionType::Put => (k - state).max_or_nan(T::zero()),
    };
    if tau == T::zero() && !(r.is_nan() || q.is_nan()) {
      return intrinsic(s);
    }
    let dt = tau / T::from_usize_(self.steps);
    let sqrt_dt = dt.sqrt();
    let up = (self.sigma * sqrt_dt).exp();
    let down = T::one() / up;
    let drift = ((r - q) * dt).exp();
    let p = (drift - down) / (up - down);
    let positive = s > T::zero() && k > T::zero();
    if !(tau > T::zero() && positive && p >= T::zero() && p <= T::one()) {
      return T::nan();
    }
    let tree = BinomialTree::from_crr(s, up, down, p, self.steps, dt);
    let discount = (-r * dt).exp();

    let terminal_states = tree.states.last().expect("tree has at least one level");
    let mut values = Array1::from_iter(terminal_states.iter().map(|&state| intrinsic(state)));

    for level in (0..tree.up_probabilities.len()).rev() {
      let states = &tree.states[level];
      let probabilities = &tree.up_probabilities[level];
      let mut step_values = Array1::<T>::zeros(level + 1);
      for node in 0..=level {
        let p_node = probabilities[node];
        let continuation =
          discount * (p_node * values[node + 1] + (T::one() - p_node) * values[node]);
        step_values[node] = match option_style {
          OptionStyle::American => continuation.max_or_nan(intrinsic(states[node])),
          OptionStyle::European => continuation,
        };
      }
      values = step_values;
    }
    values[0]
  }

  fn price_european(&self, s: T, k: T, r: T, q: T, tau: T, option_type: OptionType) -> T {
    self.price_rollback(s, k, r, q, tau, option_type, OptionStyle::European)
  }

  /// American price by comparing continuation with exercise at every node.
  ///
  /// Reference: Cox, Ross & Rubinstein (1979), "Option Pricing: A Simplified Approach", JFE 7(3)
  pub fn price_american(&self, s: T, k: T, r: T, q: T, tau: T, option_type: OptionType) -> T {
    self.price_rollback(s, k, r, q, tau, option_type, OptionStyle::American)
  }
}

impl ModelPricer for CrrModel<f64> {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_european(s, k, r, q, tau, OptionType::Call)
  }

  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_european(s, k, r, q, tau, OptionType::Put)
  }
}

/// [`price_call`](ModelPricer::price_call) rolls back the European payoff, not
/// [`price_american`](CrrModel::price_american), so it is the vanilla call on `Se^{(r−q)τ}`.
impl VanillaEuropeanCall for CrrModel<f64> {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::OptionStyle;
  use crate::pricing::BjerksundStensland2002Pricer;
  use crate::pricing::finite_difference::FiniteDifferenceMethod;
  use crate::pricing::finite_difference::FiniteDifferencePricer;

  #[test]
  fn crr_call_recovers_black_scholes_at_high_steps() {
    let model = CrrModel::new(0.2_f64, 200);
    let bs_call = 10.4506; // Black-Scholes call at S=100, K=100, r=0.05, q=0, T=1, sigma=0.2
    let crr_call = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    assert!((crr_call - bs_call).abs() < 0.05, "got {}", crr_call);
  }

  #[test]
  fn crr_put_call_parity() {
    let model = CrrModel::new(0.25_f64, 100);
    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let q = 0.02;
    let tau = 0.5;
    let c = model.price_call(s, k, r, q, tau);
    let p = model.price_put(s, k, r, q, tau);
    let parity = c - p - s * (-q * tau).exp() + k * (-r * tau).exp();
    assert!(parity.abs() < 0.01, "parity residual {parity}");
  }

  /// Without dividends an American call is never exercised early, so it rolls back to the European.
  /// Reference: Merton (1973), Bell J. Econ. Manag. Sci. 4(1), Theorem 2
  #[test]
  fn crr_american_call_no_dividend_equals_european() {
    let model = CrrModel::new(0.2_f64, 500);
    let (s, k, r, q, tau) = (100.0, 100.0, 0.05, 0.0, 1.0);
    let european = model.price_call(s, k, r, q, tau);
    let american = model.price_american(s, k, r, q, tau, OptionType::Call);
    assert!(
      (american - european).abs() < 1e-10,
      "american {american} vs european {european}"
    );
  }

  /// With `q > 0` early exercise of a call can pay, so `price_rollback`'s clamp binds; checked
  /// against [`BjerksundStensland2002Pricer`].
  #[test]
  fn crr_american_call_with_dividends_exceeds_european() {
    let (s, k, r, q, sigma, tau) = (100.0, 100.0, 0.03, 0.07, 0.2, 1.0);
    let model = CrrModel::new(sigma, 1000);
    let european = model.price_call(s, k, r, q, tau);
    let american = model.price_american(s, k, r, q, tau, OptionType::Call);
    assert!(
      american > european,
      "expected strict early-exercise premium: american {american} vs european {european}"
    );

    let reference = BjerksundStensland2002Pricer::new(sigma).price_call(s, k, r, q, tau);
    let rel_err = (american - reference).abs() / reference;
    assert!(
      rel_err < 1e-2,
      "crr {american} vs bjerksund-stensland {reference} (rel err {rel_err})"
    );
  }

  #[test]
  fn crr_american_put_geq_european_put() {
    let model = CrrModel::new(0.2_f64, 500);
    let (s, k, r, q, tau) = (90.0, 100.0, 0.05, 0.0, 1.0);
    let european = model.price_put(s, k, r, q, tau);
    let american = model.price_american(s, k, r, q, tau, OptionType::Put);
    assert!(
      american > european,
      "expected strict early-exercise premium: american {american} vs european {european}"
    );
  }

  /// Reference: Bjerksund, P. & Stensland, G. (2002), "Closed Form Valuation
  /// of American Options", NHH discussion paper 2002/09 — closed-form
  /// American approximation cross-checked against the CRR rollback.
  #[test]
  fn crr_american_put_matches_bjerksund_stensland() {
    let (s, k, r, q, sigma, tau) = (100.0, 110.0, 0.05, 0.0, 0.2, 1.0);
    let model = CrrModel::new(sigma, 1000);
    let crr = model.price_american(s, k, r, q, tau, OptionType::Put);

    let reference = BjerksundStensland2002Pricer::new(sigma).price_put(s, k, r, q, tau);

    let rel_err = (crr - reference).abs() / reference;
    assert!(
      rel_err < 1e-2,
      "crr {crr} vs bjerksund-stensland {reference} (rel err {rel_err})"
    );
  }

  /// Reference: `FiniteDifferencePricer` Crank-Nicolson American solve
  /// (projected max-with-intrinsic clamp per time step).
  #[test]
  fn crr_american_put_matches_fd() {
    let (s, k, r, sigma, tau) = (100.0, 110.0, 0.05, 0.2, 1.0);
    let model = CrrModel::new(sigma, 1000);
    let crr = model.price_american(s, k, r, 0.0, tau, OptionType::Put);

    let fd = FiniteDifferencePricer::new(
      sigma,
      4000,
      400,
      OptionStyle::American,
      FiniteDifferenceMethod::CrankNicolson,
    )
    .price_put(s, k, r, 0.0, tau);

    let rel_err = (crr - fd).abs() / fd;
    assert!(rel_err < 5e-3, "crr {crr} vs fd {fd} (rel err {rel_err})");
  }

  #[test]
  fn crr_american_converges_in_steps() {
    let (s, k, r, q, sigma, tau) = (90.0_f64, 100.0, 0.05, 0.0, 0.2, 1.0);
    let price = |steps: usize| {
      CrrModel::<f64>::new(sigma, steps).price_american(s, k, r, q, tau, OptionType::Put)
    };
    let coarse = (price(200) - price(100)).abs();
    let fine = (price(1600) - price(800)).abs();
    assert!(
      fine < coarse,
      "expected tighter bracket at higher step counts: fine {fine} vs coarse {coarse}"
    );
  }
}
