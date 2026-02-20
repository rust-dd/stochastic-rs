//! # Svcgmy (CGMYSV discrete-time approximation)
//!
//! Paper model (Kim 2021):
//! $$
//! L_t = Z_{V_t} + \rho v_t, \quad V_t = \int_0^t v_s ds,
//! $$
//! where $Z$ is a standard CGMY process (independent of $v$) and $v$ follows CIR:
//! $$
//! dv_t=\kappa(\eta-v_t)dt+\zeta\sqrt{v_t}dW_t.
//! $$
//!
//! This implementation generates the discrete-time approximation on a grid
//! $t_m = m\Delta t$, using Algorithm 1 in the paper.
//!
//! Notes:
//! - `rho` is a **loading** on $v_t$ (not a correlation), so it is not restricted to [-1, 1].
//! - Series indices follow Algorithm 1: **j = 1..J**, with **Γ0 = 0**.
//!
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use scilib::math::basic::gamma;

use crate::distributions::exp::SimdExp;
use crate::distributions::uniform::SimdUniform;
use crate::stats::non_central_chi_squared;
use crate::stochastic::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// CGMY Stochastic Volatility process (CGMYSV)
///
/// Paper: <https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf>
pub struct SVCGMY<T: FloatExt> {
  /// Positive tempering parameter λ+ > 0
  pub lambda_plus: T,
  /// Negative tempering parameter λ− > 0
  pub lambda_minus: T,
  /// Activity parameter α (0 < α < 2)
  pub alpha: T,

  /// CIR mean reversion κ > 0
  pub kappa: T,
  /// CIR long-term level η >= 0
  pub eta: T,
  /// CIR vol-of-vol ζ > 0
  pub zeta: T,

  /// Loading parameter ρ (no [-1,1] restriction)
  pub rho: T,

  /// Number of time steps (M+1 points including t=0)
  pub n: usize,
  /// Truncation level J (number of series terms)
  pub j: usize,

  /// Initial value (interpreted as L0)
  pub x0: Option<T>,
  /// Initial variance v0
  pub v0: Option<T>,
  /// Time horizon T
  pub t: Option<T>,
}

impl<T: FloatExt> SVCGMY<T> {
  pub fn new(
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    kappa: T,
    eta: T,
    zeta: T,
    rho: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(lambda_plus > T::zero(), "lambda_plus must be positive");
    assert!(lambda_minus > T::zero(), "lambda_minus must be positive");
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha must be in (0, 2)"
    );
    assert!(kappa > T::zero(), "kappa must be positive");
    assert!(eta >= T::zero(), "eta must be non-negative");
    assert!(zeta > T::zero(), "zeta must be positive");
    assert!(n >= 2, "n must be >= 2");

    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }

    Self {
      lambda_plus,
      lambda_minus,
      alpha,
      kappa,
      eta,
      zeta,
      rho,
      n,
      j,
      x0,
      v0,
      t,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for SVCGMY<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let mut x = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);

    x[0] = self.x0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());
    // y = L - rho * v  =>  L = y + rho * v
    y[0] = x[0] - self.rho * v[0];

    let f2 = T::from_usize_(2);

    // C = (Γ(2-α) (λ+^(α-2) + λ-^(α-2)))^{-1}
    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = T::one()
      / (T::from_f64_fast(g)
        * (self.lambda_plus.powf(self.alpha - f2) + self.lambda_minus.powf(self.alpha - f2)));

    // CIR exact-step constants (paper)
    let c = (f2 * self.kappa) / ((T::one() - (-self.kappa * dt).exp()) * self.zeta.powi(2));
    let df = T::from_usize_(4) * self.kappa * self.eta / self.zeta.powi(2);

    // 1) Simulate v on the grid via noncentral chi-square
    for i in 1..self.n {
      let ncp = f2 * c * v[i - 1] * (-self.kappa * dt).exp();
      let xi = non_central_chi_squared::sample(df, ncp, &mut rng);
      v[i] = xi / (f2 * c);
    }

    // 2) Series random variables (Algorithm 1 uses j=1..J with Γ0=0)
    let J = self.j;
    let size = J + 1; // index 0 is reserved (Γ0=0)

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    // U_j ~ Unif(0,1), E_j ~ Exp(1), τ_j ~ Unif(0,T)
    let U = Array1::<T>::random(size, &uniform);
    let E = Array1::<T>::random(size, exp);
    let tau = Array1::<T>::random(size, &uniform) * t_max;

    // Γ_0=0, Γ_j = Γ_{j-1} + E'_j; we reuse Poisson-generator-as-arrival-times for Γ_j
    let P = Poisson::new(T::one(), Some(size), None).sample();

    // c(τ_j) = C * v_{k-1} where (k-1)dt < τ_j <= k dt
    let mut c_tau = Array1::<T>::zeros(size);
    for j in 1..=J {
      let tau_j = tau[j];
      let k = ((tau_j / dt).ceil()).min(T::from_usize_(self.n - 1));
      let v_km1 = if k == T::zero() {
        v[0]
      } else {
        v[k.to_usize().unwrap() - 1]
      };
      c_tau[j] = C * v_km1;
    }

    // 3) Build Y on the grid (Algorithm 1)
    for i in 1..self.n {
      // b_m = - v_{m-1} (λ+^(α-1) - λ-^(α-1)) / ((1-α)(λ+^(α-2)+λ-^(α-2)))
      let numerator = v[i - 1]
        * (self.lambda_plus.powf(self.alpha - T::one())
          - self.lambda_minus.powf(self.alpha - T::one()));
      let denominator = (T::one() - self.alpha)
        * (self.lambda_plus.powf(self.alpha - f2) + self.lambda_minus.powf(self.alpha - f2));
      let b = -numerator / denominator;

      let mut jump_component = T::zero();

      let t_1 = T::from_usize_(i - 1) * dt;
      let t = T::from_usize_(i) * dt;

      for j in 1..=J {
        if tau[j] > t_1 && tau[j] <= t {
          // V_j is chosen as λ+ or -λ- with prob 1/2
          let v_j = if rng.random_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          // min term: ((α Γ_j)/(2 c(τ_j) T))^{-1/α}  ∧  E_j U_j^{1/α} / |V_j|
          let num = self.alpha * P[j];
          let den = f2 * c_tau[j] * t_max;
          let term1 = (num / den).powf(-T::one() / self.alpha);

          let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();
          let min_term = term1.min(term2);

          jump_component += min_term * (v_j / v_j.abs());
        }
      }

      y[i] = y[i - 1] + jump_component + b * dt;
    }

    // 4) L ≈ Y + ρ v  (paper Eq. (9))
    for i in 1..self.n {
      x[i] = y[i] + self.rho * v[i];
    }

    x
  }
}

py_process_1d!(PySVCGMY, SVCGMY,
  sig: (lambda_plus, lambda_minus, alpha, kappa, eta, zeta, rho, n, j, x0=None, v0=None, t=None, dtype=None),
  params: (lambda_plus: f64, lambda_minus: f64, alpha: f64, kappa: f64, eta: f64, zeta: f64, rho: f64, n: usize, j: usize, x0: Option<f64>, v0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn x0_shift_is_preserved_when_no_jumps_and_zero_loading_and_zero_var() {
    // If J=0, rho=0, and v_t ≡ 0, then y has no jumps and b=0, so x remains x0.
    // Setting eta=0 and v0=0 makes CIR stay at 0 in this exact scheme.
    let model = SVCGMY::new(
      2.0_f64, // lambda_plus
      2.0_f64, // lambda_minus
      0.5,     // alpha
      1.0,     // kappa
      0.0,     // eta  (important: keep v_t = 0)
      0.2,     // zeta
      0.0,     // rho
      8,       // n
      0,       // j (J=0)
      Some(5.0),
      Some(0.0),
      Some(1.0),
    );

    let x = model.sample();
    assert!(x.iter().all(|v| (*v - 5.0).abs() < 1e-12));
  }

  #[test]
  #[should_panic(expected = "alpha must be in (0, 2)")]
  fn invalid_alpha_panics() {
    let _ = SVCGMY::new(
      2.0_f64,
      2.0,
      2.5,
      1.0,
      0.04,
      0.2,
      0.0,
      8,
      0,
      Some(0.0),
      Some(0.01),
      Some(1.0),
    );
  }
}
