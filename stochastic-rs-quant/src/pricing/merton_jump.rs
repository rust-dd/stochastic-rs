//! Merton (1976) jump-diffusion priced as a Poisson-weighted series of Black-Scholes terms.
//!
//! $$
//! V=\sum_{n=0}^{\infty}e^{-\lambda\tau}\frac{(\lambda\tau)^n}{n!}V_{BS}(\sigma_n),
//! \qquad\sigma_n=\sqrt{d^2+z^2\tfrac{n}{\tau}}
//! $$
//!
//! `d`, `z`: diffusive and jump log-size std from `sigma`, `gamma`; jumps of `E(Y) = 1` keep `r`.
use super::bsm::BsmCoc;
use super::bsm::BsmPricer;
use crate::CallPut;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

mod greeks;

/// Merton (1976) jump diffusion; a query discounts at `r`, carries at [`BsmPricer::b`]`(r, q)`, and
/// where `b = r − q` (`Merton1973`, `GarmanKohlhagen1983`) carry `b₀` at `r₀` is `(r₀, r₀ − b₀)`.
///
/// ```
/// use stochastic_rs_quant::pricing::bsm::BsmCoc;
/// use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = Merton1976Pricer::new(0.2, 0.5, 0.4, 10, BsmCoc::Bsm1973);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.0, 0.5);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Merton1976Pricer {
  /// Volatility
  pub sigma: f64,
  /// Expected number of jumps
  pub lambda: f64,
  /// Percentage of the volatility due to jumps
  pub gamma: f64,
  /// Iteration limit
  pub m: usize,
  /// Cost of carry
  pub b: BsmCoc,
}

/// Where a Poisson term's query sits: at `σ_n = 0` its `d₁` is `±∞` or `0/0`, and the two cases
/// have different `σ → 0⁺` limits, so no blanket `0` serves both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TermRegime {
  /// $\sigma_n > 0$, or a query the closed forms already answer with `NaN`
  /// and must go on answering with one. The closed form *is* the value.
  Ordinary,
  /// `σ_n = 0` off the forward: `d₁ → ±∞` and the CDFs saturate; only `gamma` is `0/0`, with
  /// limit `0`, since `φ(d₁) ~ e^{−c²/2σ²}` beats `1/σ`.
  Saturated,
  /// `σ_n = 0` at the forward `Se^{bτ} = K`: each closed form is `0/0`; both CDFs tend to `½`, so
  /// the price and first-order Greeks have finite limits and `gamma` diverges.
  AtTheForward,
}

impl Merton1976Pricer {
  /// Panics unless `sigma` and `lambda` are finite and `≥ 0`, `gamma ∈ [0, 1]` and `m ≥ 1`; at
  /// `lambda = 0` the model is Black-Scholes at volatility `sigma`.
  pub fn new(sigma: f64, lambda: f64, gamma: f64, m: usize, b: BsmCoc) -> Self {
    assert!(
      sigma.is_finite() && sigma >= 0.0,
      "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = {sigma:?}"
    );
    assert!(
      lambda.is_finite() && lambda >= 0.0,
      "lambda must satisfy `lambda.is_finite() && lambda >= 0.0`, got lambda = {lambda:?}"
    );
    assert!(
      (0.0..=1.0).contains(&gamma),
      "gamma must satisfy `(0.0..=1.0).contains(&gamma)`, got gamma = {gamma:?}"
    );
    assert!(m >= 1, "m must satisfy `m >= 1`, got m = {m:?}");
    Self {
      sigma,
      lambda,
      gamma,
      m,
      b,
    }
  }

  /// Poisson-weighted series $\sum_{n=0}^{m-1} w_n \cdot V_{BS}(\sigma_n)$ with running-product
  /// weights, so `m` may exceed the ≈ 21 terms an integer `n!` survives; the payoff at `tau = 0`.
  pub fn price_call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    if tau == 0.0 {
      return self.base_bsm().price_call_put(s, k, r, q, tau);
    }
    let mut call = 0.0;
    let mut put = 0.0;

    for i in 0..self.m {
      let weight = self.poisson_weight(i, tau);
      let term = Self::term_call_put(&self.term_bsm(i, tau), s, k, r, q, tau);
      call += term.call * weight;
      put += term.put * weight;
    }

    CallPut { call, put }
  }

  /// Jump-size std `z` from giving a `gamma` share of the variance `σ²` to jumps; `0` at
  /// `lambda == 0`, the value at that point, not the `λ → 0⁺` limit (Black-Scholes at `σ√(1 − γ)`).
  fn jump_size_std(&self) -> f64 {
    if self.lambda == 0.0 {
      return 0.0;
    }
    (self.sigma.powi(2) * self.gamma / self.lambda).sqrt()
  }

  /// `√(σ² − λz²)` with `λz² = σ²γ` taken directly, as squaring `jump_size_std` back rounds below 0
  /// at `gamma = 1`; `sigma` at `lambda = 0`, NaN where `z` is not real (a literal past `new`).
  fn diffusive_std(&self) -> f64 {
    let jump_variance_rate = if self.lambda == 0.0 {
      0.0
    } else if self.jump_size_std().is_nan() {
      f64::NAN
    } else {
      self.sigma.powi(2) * self.gamma
    };
    (self.sigma.powi(2) - jump_variance_rate).sqrt()
  }

  /// `σ_n = √(d² + z²n/τ)`: the diffusion runs all of `τ` whatever the jump count, so `σ_0 = d`,
  /// and the Poisson mean of `σ_n²τ` is `σ²τ`, which makes `sigma` the total volatility.
  /// Reference: Merton (1976), J. Financ. Econ. 3, 125–144, eq. (18)
  /// Reference: Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., §6.9.1
  fn term_vol(&self, n: usize, tau: f64) -> f64 {
    (self.diffusive_std().powi(2) + self.jump_size_std().powi(2) * n as f64 / tau).sqrt()
  }

  /// `e^{−λτ}(λτ)^n/n!` with `(λτ)^n/n!` as a running `f64` product: an integer `n!` overflows
  /// `usize` past `n ≈ 20`, while the weight stays in `[0, 1]`.
  fn poisson_weight(&self, n: usize, tau: f64) -> f64 {
    let lt = self.lambda * tau;
    let ratio = (0..n).fold(1.0, |acc, i| acc * lt / (i as f64 + 1.0));
    (-lt).exp() * ratio
  }

  /// The no-jump Black-Scholes limit, at total volatility `sigma`.
  fn base_bsm(&self) -> BsmPricer {
    BsmPricer::new(self.sigma, self.b)
  }

  /// Built past `BsmPricer::new`: `σ_n` comes from the query and overflows to `+∞` as `τ → 0⁺`.
  fn term_bsm(&self, n: usize, tau: f64) -> BsmPricer {
    BsmPricer {
      sigma: self.term_vol(n, tau),
      b: self.b,
    }
  }

  /// Classifies a term's query ([`TermRegime`]); a query failing `0 < tau < ∞`, `s > 0` or `k > 0`
  /// (a NaN fails, `+∞` passes) stays `Ordinary`, so the closed form answers it, not a limit.
  fn term_regime(term: &BsmPricer, s: f64, k: f64, r: f64, q: f64, tau: f64) -> TermRegime {
    if term.sigma != 0.0 || !(tau.is_finite() && tau > 0.0 && s > 0.0 && k > 0.0) {
      return TermRegime::Ordinary;
    }
    if term.d1_d2(s, k, r, q, tau).0.is_nan() {
      TermRegime::AtTheForward
    } else {
      TermRegime::Saturated
    }
  }

  /// Both legs of one Poisson term. A zero-volatility term at the forward `S e^{bτ} = K` is `0/0`
  /// and takes its `σ → 0⁺` limit `±½(S e^{(b−r)τ} − K e^{−rτ})`, so it cannot poison the sum.
  fn term_call_put(term: &BsmPricer, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    if Self::term_regime(term, s, k, r, q, tau) == TermRegime::AtTheForward {
      let half_carry = 0.5 * s * ((term.b(r, q) - r) * tau).exp();
      let half_disc = 0.5 * k * (-r * tau).exp();
      return CallPut {
        call: half_carry - half_disc,
        put: half_disc - half_carry,
      };
    }
    term.price_call_put(s, k, r, q, tau)
  }
}

impl ModelPricer for Merton1976Pricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).call
  }

  /// Not the parity default: each term carries at `e^{(b−r)τ}`, the default's `e^{−qτ}` only when
  /// `b = r − q`, false for [`BsmCoc::Bsm1973`] at `q ≠ 0` and for `Black1976` and `Asay1982`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).put
  }
}

/// Each Poisson term is a European vanilla priced by [`BsmPricer`], whose carry this inherits.
impl VanillaEuropeanCall for Merton1976Pricer {
  /// `Se^{bτ}` under `self.b`, from [`BsmPricer::vanilla_call_forward`], so the forward and the
  /// series' terms cannot carry differently.
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.base_bsm().vanilla_call_forward(s, r, q, tau)
  }
}

#[cfg(test)]
mod tests;
