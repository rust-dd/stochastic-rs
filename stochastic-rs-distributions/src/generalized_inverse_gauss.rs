//! # Generalized Inverse Gaussian (GIG)
//!
//! $$
//! f(x) = \frac{(\psi/\chi)^{\lambda/2}}{2K_\lambda(\sqrt{\chi\psi})}\,x^{\lambda-1}
//! \exp\!\Bigl(-\tfrac12\bigl(\tfrac{\chi}{x} + \psi x\bigr)\Bigr),\qquad x > 0,\qquad
//! \mathbb E X^k = (\chi/\psi)^{k/2}
//! \frac{K_{\lambda+k}(\sqrt{\chi\psi})}{K_\lambda(\sqrt{\chi\psi})},
//! $$
//!
//! The GH family's clock, the inverse Gaussian at `λ = −½`, no closed-form cdf; at `χ, ψ > 0` drawn
//! as `Y/√(ψ/χ)` by Hörmann and Leydold's three regimes of `GIGrvg`, `1/Y` at `λ < 0`.
//!
//! Reference: Hörmann, W., Leydold, J. (2014), "Generating generalized inverse Gaussian random variates", *Statistics and Computing* 24(4), 547-557, DOI 10.1007/s11222-013-9387-3.
//! Reference: Dagpunar, J.S. (1989), "An easily implemented generalised inverse Gaussian generator", *Communications in Statistics — Simulation and Computation* 18(2), 703-710, DOI 10.1080/03610918908812785.
//! Reference: Jørgensen, B. (1982), *Statistical Properties of the Generalized Inverse Gaussian Distribution*, Lecture Notes in Statistics 9, Springer, DOI 10.1007/978-1-4612-5698-4.
//! Reference: Olver, F.W.J. et al. (eds.), *NIST Digital Library of Mathematical Functions*, eq. 10.30.2 (the mgf at its boundary).

use std::mem::MaybeUninit;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::as_uninit_mut;

use super::SimdFloatExt;
use crate::generalized_inverse_gauss::cumulants::Cumulants;
use crate::seeded::StreamState;
use crate::special::KDenominator;
use crate::special::LARGE_ORDER;
use crate::special::PLAIN_LOG_SIZE;
use crate::special::RatioParts;
use crate::special::exp_parts;
use crate::special::ldexp;
use crate::special::ln_bessel_ke;
use crate::special::ln_quotient;
use crate::special::ln_shift;
use crate::special::ratio_regime;
use crate::special::split_pow2;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Under this `χψ` a ratio's limit `g_ν → ν` may hold within `2⁻¹¹²`, which [`ratio_moment::limit_moment`] then tests.
const LIMIT_CLOCK: f64 = 7.703_719_777_548_943e-34;

/// From here a mean, whose doubles miss by up to `43ε` on the ratios' route, is formed again in two parts ([`near_max`]).
pub(crate) const NEAR_MAX: f64 = f64::MAX * (1.0 - 64.0 * f64::EPSILON);

/// Which Hörmann–Leydold generator the parameters select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Regime {
  /// Algorithm 1: three-piece hat for $\lambda < 1$ and small $\beta$.
  Hat,
  /// Algorithm 2: ratio-of-uniforms without mode shift.
  RatioOfUniforms,
  /// Algorithm 3: ratio-of-uniforms with mode shift (Dagpunar–Lehner).
  RatioOfUniformsShifted,
}

/// Precomputed generator state for the quasi-density $g(y \mid \lambda, \beta)$
/// with $\lambda \ge 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Setup {
  lambda: f64,
  beta: f64,
  regime: Regime,
  /// $\log g(m)$, the normalisation that keeps every hat in `(0, 1]`.
  log_g_mode: f64,
  m: f64,
  x0: f64,
  x_star: f64,
  k2: f64,
  k3: f64,
  a1: f64,
  a2: f64,
  a3: f64,
  u_minus: f64,
  u_plus: f64,
}

impl Setup {
  fn new(lambda: f64, beta: f64) -> Self {
    let log_g = |x: f64| (lambda - 1.0) * x.ln() - 0.5 * beta * (x + 1.0 / x);
    let regime = if lambda > 2.0 || beta > 3.0 {
      Regime::RatioOfUniformsShifted
    } else if lambda >= 1.0 - 2.25 * beta * beta || beta > 0.2 {
      Regime::RatioOfUniforms
    } else {
      Regime::Hat
    };
    let mut s = Self {
      lambda,
      beta,
      regime,
      log_g_mode: 0.0,
      m: 0.0,
      x0: 0.0,
      x_star: 0.0,
      k2: 0.0,
      k3: 0.0,
      a1: 0.0,
      a2: 0.0,
      a3: 0.0,
      u_minus: 0.0,
      u_plus: 0.0,
    };
    match regime {
      Regime::Hat => {
        s.m = beta / ((1.0 - lambda) + ((1.0 - lambda).powi(2) + beta * beta).sqrt());
        s.log_g_mode = log_g(s.m);
        s.x0 = beta / (1.0 - lambda);
        s.x_star = s.x0.max(2.0 / beta);
        s.a1 = s.x0;
        if s.x0 < 2.0 / beta {
          s.k2 = (-beta - s.log_g_mode).exp();
          s.a2 = if lambda == 0.0 {
            s.k2 * (2.0 / (beta * beta)).ln()
          } else {
            s.k2 * ((2.0 / beta).powf(lambda) - s.x0.powf(lambda)) / lambda
          };
        }
        s.k3 = ((lambda - 1.0) * s.x_star.ln() - s.log_g_mode).exp();
        s.a3 = 2.0 * s.k3 * (-s.x_star * beta / 2.0).exp() / beta;
      }
      Regime::RatioOfUniforms => {
        s.m = beta / ((1.0 - lambda) + ((1.0 - lambda).powi(2) + beta * beta).sqrt());
        s.log_g_mode = log_g(s.m);
        let x_plus = ((1.0 + lambda) + ((1.0 + lambda).powi(2) + beta * beta).sqrt()) / beta;
        s.u_plus = x_plus * (0.5 * (log_g(x_plus) - s.log_g_mode)).exp();
      }
      Regime::RatioOfUniformsShifted => {
        s.m = (((lambda - 1.0).powi(2) + beta * beta).sqrt() + (lambda - 1.0)) / beta;
        s.log_g_mode = log_g(s.m);
        let a = -2.0 * (lambda + 1.0) / beta - s.m;
        let b = 2.0 * (lambda - 1.0) * s.m / beta - 1.0;
        let c = s.m;
        let p = b - a * a / 3.0;
        let q = 2.0 * a.powi(3) / 27.0 - a * b / 3.0 + c;
        let phi = (-(q / 2.0) * (-27.0 / p.powi(3)).sqrt())
          .clamp(-1.0, 1.0)
          .acos();
        let radius = (-4.0 * p / 3.0).sqrt();
        let x_minus = radius * (phi / 3.0 + 4.0 * std::f64::consts::PI / 3.0).cos() - a / 3.0;
        let x_plus = radius * (phi / 3.0).cos() - a / 3.0;
        s.u_minus = (x_minus - s.m) * (0.5 * (log_g(x_minus) - s.log_g_mode)).exp();
        s.u_plus = (x_plus - s.m) * (0.5 * (log_g(x_plus) - s.log_g_mode)).exp();
      }
    }
    s
  }

  /// $\log g(x) - \log g(m)$.
  #[inline]
  fn log_g_normalised(&self, x: f64) -> f64 {
    (self.lambda - 1.0) * x.ln() - 0.5 * self.beta * (x + 1.0 / x) - self.log_g_mode
  }

  /// One draw from $g(\cdot \mid \lambda, \beta)$ using uniforms and (for the
  /// shifted ratio-of-uniforms) nothing else.
  fn draw(&self, mut uniform: impl FnMut() -> f64) -> f64 {
    match self.regime {
      Regime::Hat => loop {
        let u = uniform();
        let v = uniform() * (self.a1 + self.a2 + self.a3);
        let (x, log_h) = if v <= self.a1 {
          (self.x0 * v / self.a1, 0.0)
        } else if v <= self.a1 + self.a2 {
          let v = v - self.a1;
          let x = if self.lambda == 0.0 {
            self.beta * (v * self.beta.exp()).exp()
          } else {
            (self.x0.powf(self.lambda) + v * self.lambda / self.k2).powf(1.0 / self.lambda)
          };
          (x, self.k2.ln() + (self.lambda - 1.0) * x.ln())
        } else {
          let v = v - (self.a1 + self.a2);
          let x = -2.0 / self.beta
            * ((-self.x_star * self.beta / 2.0).exp() - v * self.beta / (2.0 * self.k3)).ln();
          (x, self.k3.ln() - x * self.beta / 2.0)
        };
        if x > 0.0 && u.ln() + log_h <= self.log_g_normalised(x) {
          return x;
        }
      },
      Regime::RatioOfUniforms => loop {
        let u = uniform() * self.u_plus;
        let v = uniform();
        // `v = 0`, outside the region `0 < v ≤ √g(u/v)`, comes from an f32 uniform once in 8.4e6
        // draws; at `λ < 1` its `−∞ ≤ −∞` test would pass and leave `u/v = +∞` (or 0) as a draw.
        if v <= 0.0 {
          continue;
        }
        let x = u / v;
        if 2.0 * v.ln() <= self.log_g_normalised(x) {
          return x;
        }
      },
      Regime::RatioOfUniformsShifted => loop {
        let u = self.u_minus + uniform() * (self.u_plus - self.u_minus);
        let v = uniform();
        // The same exclusion: a `λ > 2` shape fails the test on its own (`log g` at ∞ is NaN), but
        // this regime also takes `λ < 1` at `β > 3`, accepted as above.
        if v <= 0.0 {
          continue;
        }
        let x = u / v + self.m;
        if x > 0.0 && 2.0 * v.ln() <= self.log_g_normalised(x) {
          return x;
        }
      },
    }
  }
}

/// The GIG(λ, χ, ψ) law: density within `ε(S + 16)` to `r = √(λ² + χψ) = 50`, `S` its logs' sizes
/// summed, then `4ε(|ln f| + 20)`, plus `k√r·1e-31` at `k` deviations past `r = 3e34`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGig<T> {
  lambda: T,
  chi: T,
  psi: T,
  setup: Setup,
  /// $\alpha = \sqrt{\psi/\chi}$, the scale taken out of the quasi-density.
  scale_out: f64,
  invert: bool,
}

impl<T: SimdFloatExt> SimdGig<T> {
  /// Construct a GIG$(\lambda, \chi, \psi)$.
  pub fn new(lambda: T, chi: T, psi: T) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      chi.is_finite(),
      "chi must satisfy `chi.is_finite()`, got chi = {chi:?}"
    );
    assert!(
      psi.is_finite(),
      "psi must satisfy `psi.is_finite()`, got psi = {psi:?}"
    );
    assert!(
      chi > T::zero(),
      "chi must satisfy `chi > T::zero()`, got chi = {chi:?}"
    );
    assert!(
      psi > T::zero(),
      "psi must satisfy `psi > T::zero()`, got psi = {psi:?}"
    );
    let lambda_f = lambda.to_f64().unwrap();
    let chi_f = chi.to_f64().unwrap();
    let psi_f = psi.to_f64().unwrap();
    let beta = (chi_f * psi_f).sqrt();
    Self {
      lambda,
      chi,
      psi,
      setup: Setup::new(lambda_f.abs(), beta),
      scale_out: (psi_f / chi_f).sqrt(),
      invert: lambda_f < 0.0,
    }
  }

  /// The index `λ`.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// The weight `χ` of the `1/x` term.
  pub fn chi(&self) -> T {
    self.chi
  }

  /// The weight `ψ` of the `x` term.
  pub fn psi(&self) -> T {
    self.psi
  }

  /// A quasi-density draw `y` mapped to the law: `1/y` for negative `λ`, then the scale `1/α`.
  #[inline]
  fn rescale(&self, y: f64) -> T {
    let y = if self.invert { 1.0 / y } else { y };
    T::from_f64_fast(y / self.scale_out)
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [MaybeUninit<T>]) {
    for x in out.iter_mut() {
      x.write(
        self.rescale(
          self
            .setup
            .draw(|| T::sample_uniform_simd(rng).to_f64().unwrap()),
        ),
      );
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.rescale(self.setup.draw(|| T::sample_uniform(rng).to_f64().unwrap()))
  }

  fn params(&self) -> (f64, f64, f64) {
    (
      self.lambda.to_f64().unwrap(),
      self.chi.to_f64().unwrap(),
      self.psi.to_f64().unwrap(),
    )
  }

  /// $\mathbb E X^k = (\chi/\psi)^{k/2} K_{\lambda+k}(b)/K_\lambda(b)$, $b = \sqrt{\chi\psi}$: a
  /// product of Bessel ratios under order 50, of the saddle's `w_jʲe^{rest}` over runs past it.
  pub fn raw_moment(&self, k: u32) -> f64 {
    let (lambda, chi, psi) = self.params();
    if chi * psi < LIMIT_CLOCK
      && let Some(v) = ratio_moment::limit_moment(lambda, (chi, psi), k, 1.0)
    {
      return v;
    }
    let roots = (chi.sqrt(), psi.sqrt());
    let v = Self::raw_moment_at((lambda, roots.0, roots.1), k);
    if v < NEAR_MAX {
      return v;
    }
    near_max::moment(self, k, v)
  }

  /// [`Self::raw_moment`] at `(λ, √χ, √ψ)`, which the GH mixture forms from `δ` and `γ` without squaring them.
  pub(crate) fn raw_moment_at(clock: (f64, f64, f64), k: u32) -> f64 {
    Self::scaled_raw_moment_at(clock, k, 1.0)
  }

  /// `factor·E Xᵏ` at `(λ, √χ, √ψ)`: the ratios' product below order 50, the saddle's from it;
  /// `factor`'s power of two, and past the doubles `w_k = 2ʲm`'s, kept apart.
  pub(crate) fn scaled_raw_moment_at(
    (lambda, root_chi, root_psi): (f64, f64, f64),
    k: u32,
    factor: f64,
  ) -> f64 {
    if factor == 0.0 {
      return 0.0;
    }
    let order = f64::from(k);
    if let Some((wk, rest)) = ratio_moment::saddle_route(lambda, order, (root_chi, root_psi)) {
      let moment = wk.powf(order) * rest.exp();
      let value = factor * moment;
      if moment.is_normal() && value.is_normal() {
        return value;
      }
      // Past the doubles `w_k = 2ʲm` and the factor apart, so no `ε·|ln w_k|` enters, as a logarithm's would.
      let (m, j) = ratio_moment::saddle_scale(lambda, order, (root_chi, root_psi));
      let ((b, n), (e, mk)) = (split_pow2(factor.abs()), split_pow2(m.powf(order)));
      let p = (i64::from(j) * i64::from(k) + i64::from(b + e)).clamp(-100_000, 100_000) as i32;
      return factor.signum() * exp_parts((rest, 0.0), n * mk, p);
    }
    let (m, e) = ratio_moment::ratio_moment(lambda, (root_chi, root_psi), k);
    let (b, n) = split_pow2(factor.abs());
    factor.signum() * ldexp(n * m, b + e)
  }

  /// The interior mgf's ratio at `t`, `shifted = ψ − 2t > 0`.
  fn mgf_ratio(&self, t: f64, shifted: f64) -> RatioParts {
    let (lambda, chi, psi) = self.params();
    let (root_chi, root_psi, root_shifted) = (chi.sqrt(), psi.sqrt(), shifted.sqrt());
    let dx = root_chi * (2.0 * t) / (root_psi + root_shifted);
    (
      (0.0, psi, shifted, 0.5 * lambda),
      (lambda, root_chi * root_shifted),
      KDenominator::new(lambda, root_chi * root_psi),
      dx,
    )
  }

  /// The cumulants `κₙ = sⁿρⁿ⁻¹kₙ`, `s = 2ᵉm`, at `(λ, √χ, √ψ)`, each factor representable where
  /// `κₙ` is not; `(χ, ψ)` where both squares are exact doubles.
  pub(crate) fn cumulants_at(
    lambda: f64,
    roots: (f64, f64),
    squares: Option<(f64, f64)>,
  ) -> Cumulants {
    cumulants::cumulants(lambda, roots, squares)
  }

  fn cumulants(&self) -> Cumulants {
    let (lambda, chi, psi) = self.params();
    Self::cumulants_at(lambda, (chi.sqrt(), psi.sqrt()), Some((chi, psi)))
  }
}

impl<T: SimdFloatExt> Sealed for SimdGig<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGig<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGig<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [MaybeUninit<T>]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    // SAFETY: `fill_parts` writes every element of the buffer.
    buf.pop(|b| self.fill_parts(rng, unsafe { as_uninit_mut(b) }))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGig<T> {
  /// One scalar Hörmann–Leydold draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdGig<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if x <= 0.0 || x == f64::INFINITY {
      return Some(0.0);
    }
    let (lambda, chi, psi) = self.params();
    let (root_chi, root_psi) = (chi.sqrt(), psi.sqrt());
    let b = root_chi * root_psi;
    let r = lambda.hypot(b);
    if r >= LARGE_ORDER {
      return Some(concentrated::density(
        lambda,
        (chi, psi),
        (root_chi, root_psi),
        r,
        x,
      ));
    }
    let (power, ln_ke) = (
      0.5 * lambda * ln_quotient(psi, chi),
      ln_bessel_ke(lambda, b),
    );
    let (tail, spread) = ((lambda - 1.0) * x.ln(), 0.5 * (chi / x + psi * x));
    if !(power.abs() + ln_ke.abs() + b + tail.abs() + spread <= PLAIN_LOG_SIZE && b.is_normal()) {
      return Some(small_order::density(lambda, chi, psi, x));
    }
    let log_norm = power - std::f64::consts::LN_2 - (ln_ke - b);
    Some((log_norm + tail - spread).exp())
  }

  fn mean(&self) -> Option<f64> {
    Some(self.raw_moment(1))
  }

  /// $\bigl(\lambda - 1 + \sqrt{(\lambda-1)^2 + \chi\psi}\bigr)/\psi$.
  fn mode(&self) -> Option<f64> {
    let (lambda, chi, psi) = self.params();
    Some((lambda - 1.0 + ((lambda - 1.0).powi(2) + chi * psi).sqrt()) / psi)
  }

  fn variance(&self) -> Option<f64> {
    let ((m, e), rho, ([_, k2, _, _], _), _) = self.cumulants();
    Some(ldexp(m * (rho * k2) * m, 2 * e))
  }

  fn skewness(&self) -> Option<f64> {
    let (_, rho, ([_, k2, k3, _], t), _) = self.cumulants();
    Some(ldexp(rho.sqrt() * (k3 / k2.powf(1.5)), t))
  }

  /// Excess kurtosis, `κ₄/κ₂²`.
  fn kurtosis(&self) -> Option<f64> {
    let (_, rho, ([_, k2, _, k4], t), _) = self.cumulants();
    Some(ldexp(rho * (k4 / (k2 * k2)), 2 * t))
  }

  /// The mgf for `t < ψ/2`; at `t = ψ/2` its limit for `λ < 0`, else there and past it `+∞`:
  ///
  /// $$
  /// \Bigl(\frac{\psi}{\psi - 2t}\Bigr)^{\lambda/2}
  /// \frac{K_\lambda(\sqrt{\chi(\psi - 2t)})}{K_\lambda(\sqrt{\chi\psi})}
  /// \quad\to\quad \Bigl(\frac{\sqrt{\chi\psi}}{2}\Bigr)^\lambda
  /// \frac{\Gamma(-\lambda)}{2K_\lambda(\sqrt{\chi\psi})}.
  /// $$
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t == f64::NEG_INFINITY {
      return Some(0.0);
    }
    let (lambda, chi, psi) = self.params();
    let shifted = psi - 2.0 * t;
    if shifted < 0.0 || (shifted == 0.0 && lambda >= 0.0) {
      return Some(f64::INFINITY);
    }
    if lambda.abs() >= LARGE_ORDER {
      return Some(ln_shift(lambda, t, (chi, psi, shifted)).exp());
    }
    if shifted == 0.0 {
      return Some(small_order::boundary_mgf(lambda, chi, psi));
    }
    let (power, num, den, dx) = self.mgf_ratio(t, shifted);
    Some(
      ratio_regime(power, num, &den, dx).unwrap_or_else(|| small_order::mgf(lambda, chi, psi, t)),
    )
  }
}

py_distribution!(PyGig, SimdGig,
  sig: (lambda, chi, psi, seed=None, dtype=None),
  params: (lambda: f64, chi: f64, psi: f64)
);

mod concentrated;
mod cumulants;
pub(crate) mod near_max;
mod ratio_moment;
mod small_order;

#[cfg(test)]
mod tests;
