//! # Generalized Inverse Gaussian (GIG)
//!
//! $$
//! f(x) = \frac{(\psi/\chi)^{\lambda/2}}{2K_\lambda(\sqrt{\chi\psi})}\,x^{\lambda-1}
//! \exp\!\Bigl(-\tfrac12\bigl(\tfrac{\chi}{x} + \psi x\bigr)\Bigr),\qquad x > 0,
//! $$
//!
//! with $\lambda \in \mathbb R$ and, for the sampler, $\chi > 0$, $\psi > 0$
//! (the boundary cases are the gamma and inverse-gamma laws). The mixing
//! law behind the generalized hyperbolic family; the inverse Gaussian is
//! $\lambda = -1/2$.
//!
//! ## Sampling
//!
//! Hörmann and Leydold's uniformly fast generator on the two-parameter
//! quasi-density $g(y \mid \lambda, \beta) = y^{\lambda-1}e^{-\beta(y + 1/y)/2}$
//! with $\beta = \sqrt{\chi\psi}$, rescaled by $\alpha = \sqrt{\psi/\chi}$ as
//! $X = Y/\alpha$, and $1/Y$ for negative $\lambda$: their Algorithm 1
//! (three-piece hat with rejection constant below 2.73) for $\lambda < 1$ and
//! small $\beta$, the ratio-of-uniforms without mode shift (Algorithm 2)
//! in the $T_{-1/2}$-concave middle range, and the Dagpunar–Lehner
//! ratio-of-uniforms with mode shift (Algorithm 3, Cardano roots) for large
//! $\lambda$ or $\beta$ — the regime split of their `GIGrvg` reference
//! implementation.
//!
//! Raw moments are Bessel ratios, $\mathbb E X^k = (\chi/\psi)^{k/2}
//! K_{\lambda+k}(\sqrt{\chi\psi}) / K_\lambda(\sqrt{\chi\psi})$; there is no
//! closed-form CDF or quantile.
//!
//! References:
//! - Hörmann, W., Leydold, J. (2014), "Generating generalized inverse
//!   Gaussian random variates", *Statistics and Computing* 24(4), 547-557.
//!   DOI: 10.1007/s11222-013-9387-3
//! - Dagpunar, J.S. (1989), "An easily implemented generalised inverse
//!   Gaussian generator", *Communications in Statistics — Simulation and
//!   Computation* 18(2), 703-710. DOI: 10.1080/03610918908812785
//! - Jørgensen, B. (1982), *Statistical Properties of the Generalized
//!   Inverse Gaussian Distribution*, Lecture Notes in Statistics 9,
//!   Springer. DOI: 10.1007/978-1-4612-5698-4

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::special::bessel_k::bessel_ke;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

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
        // The ratio-of-uniforms region is `0 < v ≤ sqrt(g(u/v))`, so `v = 0`
        // is not a point of it — but a single-precision uniform hands back
        // an exact zero once in about 8.4 million draws, and then `u/v` is
        // `+inf` while `2 ln v` is `-inf`. For `λ < 1` the normalised log
        // density at infinity is `-inf` too, so the test passes and the
        // infinity leaves as a draw (as a zero, once the `λ < 0` branch
        // inverts it — either way outside the strictly positive support).
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
        // The same `v = 0` exclusion. A `λ > 2` shape escapes it on its
        // own — `log g` at infinity is a NaN there and the test fails —
        // but this regime also takes every `λ < 1` with `β > 3`, which has
        // the `-inf ≤ -inf` acceptance of the branch above.
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

/// Generalized inverse Gaussian distribution GIG$(\lambda, \chi, \psi)$ with
/// `chi > 0` and `psi > 0`; a [`Seeded`](crate::Seeded) stream draws it in bulk.
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
    let lambda_f = lambda.to_f64().unwrap();
    let chi_f = chi.to_f64().unwrap();
    let psi_f = psi.to_f64().unwrap();
    assert!(chi_f > 0.0, "GIG: chi must be positive");
    assert!(psi_f > 0.0, "GIG: psi must be positive");
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

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.rescale(
        self
          .setup
          .draw(|| T::sample_uniform_simd(rng).to_f64().unwrap()),
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

  /// $\mathbb E X^k = (\chi/\psi)^{k/2} K_{\lambda+k}(b)/K_\lambda(b)$,
  /// $b = \sqrt{\chi\psi}$.
  pub fn raw_moment(&self, k: u32) -> f64 {
    let (lambda, chi, psi) = self.params();
    let b = (chi * psi).sqrt();
    (chi / psi).powf(0.5 * k as f64) * bessel_ke(lambda + k as f64, b) / bessel_ke(lambda, b)
  }
}

impl<T: SimdFloatExt> Sealed for SimdGig<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGig<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    let stream_seed = seed.next_seed();
    (
      StreamState {
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGig<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [T]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| self.fill_parts(rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGig<T> {
  /// One scalar Hörmann–Leydold draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdGig<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let (lambda, chi, psi) = self.params();
    let b = (chi * psi).sqrt();
    let log_norm =
      0.5 * lambda * (psi / chi).ln() - std::f64::consts::LN_2 - (bessel_ke(lambda, b).ln() - b);
    (log_norm + (lambda - 1.0) * x.ln() - 0.5 * (chi / x + psi * x)).exp()
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!("DistributionExt::cdf for SimdGig has no closed form")
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!("DistributionExt::inv_cdf for SimdGig has no closed form")
  }

  fn mean(&self) -> f64 {
    self.raw_moment(1)
  }

  /// $\bigl(\lambda - 1 + \sqrt{(\lambda-1)^2 + \chi\psi}\bigr)/\psi$.
  fn mode(&self) -> f64 {
    let (lambda, chi, psi) = self.params();
    (lambda - 1.0 + ((lambda - 1.0).powi(2) + chi * psi).sqrt()) / psi
  }

  fn variance(&self) -> f64 {
    let m1 = self.raw_moment(1);
    self.raw_moment(2) - m1 * m1
  }

  fn skewness(&self) -> f64 {
    let (m1, m2, m3) = (self.raw_moment(1), self.raw_moment(2), self.raw_moment(3));
    let var = m2 - m1 * m1;
    (m3 - 3.0 * m1 * m2 + 2.0 * m1.powi(3)) / var.powf(1.5)
  }

  /// Excess kurtosis.
  fn kurtosis(&self) -> f64 {
    let (m1, m2, m3, m4) = (
      self.raw_moment(1),
      self.raw_moment(2),
      self.raw_moment(3),
      self.raw_moment(4),
    );
    let var = m2 - m1 * m1;
    (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.powi(4)) / (var * var) - 3.0
  }

  /// $(\psi/(\psi - 2t))^{\lambda/2}\,K_\lambda(\sqrt{\chi(\psi - 2t)})/K_\lambda(\sqrt{\chi\psi})$
  /// for $t < \psi/2$, `NaN` beyond.
  fn moment_generating_function(&self, t: f64) -> f64 {
    let (lambda, chi, psi) = self.params();
    let shifted = psi - 2.0 * t;
    if shifted <= 0.0 {
      return f64::NAN;
    }
    let b = (chi * psi).sqrt();
    let bt = (chi * shifted).sqrt();
    (psi / shifted).powf(0.5 * lambda) * bessel_ke(lambda, bt) / bessel_ke(lambda, b)
      * (b - bt).exp()
  }
}

py_distribution!(PyGig, SimdGig,
  sig: (lambda, chi, psi, seed=None, dtype=None),
  params: (lambda: f64, chi: f64, psi: f64)
);

#[cfg(test)]
mod tests;
