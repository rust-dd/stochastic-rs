//! # Variance Gamma
//!
//! Madan, Carr and Chang's VG law at unit time — a Brownian motion with
//! drift $\theta$ and volatility $\sigma$ evaluated at a gamma time with
//! unit mean and variance $\nu$, shifted by $\mu$:
//!
//! $$
//! X = \mu + \theta G + \sigma\sqrt{G}\,Z,\qquad G \sim \mathrm{Gamma}(1/\nu,\ \nu),\ Z \sim \mathcal N(0,1),
//! $$
//!
//! $$
//! f(x) = \frac{2\,e^{\theta(x-\mu)/\sigma^2}}{\sigma\sqrt{2\pi}\,\nu^{1/\nu}\,\Gamma(1/\nu)}
//! \left(\frac{|x-\mu|}{\sqrt{2\sigma^2/\nu + \theta^2}}\right)^{\frac1\nu - \frac12}
//! K_{\frac1\nu - \frac12}\!\left(\frac{|x-\mu|\sqrt{2\sigma^2/\nu + \theta^2}}{\sigma^2}\right),
//! \qquad
//! \varphi(u) = e^{iu\mu}\bigl(1 - i\theta\nu u + \tfrac12\sigma^2\nu u^2\bigr)^{-1/\nu}
//! $$
//!
//! (Madan–Carr–Chang 1998, eq. 23 and eq. 2 at $t = 1$). The mean is
//! $\mu + \theta$, the variance $\sigma^2 + \nu\theta^2$; skewness and excess
//! kurtosis follow from the gamma mixture's cumulants, reducing to $0$ and
//! $3\nu$ when $\theta = 0$. At $x = \mu$ the density is finite only for
//! $\nu < 2$. There is no closed-form CDF or quantile.
//!
//! Reference: Madan, D.B., Carr, P.P., Chang, E.C. (1998), "The Variance
//! Gamma Process and Option Pricing", *European Finance Review* 2(1),
//! 79-105. DOI: 10.1023/A:1009703431535

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::gamma::GammaState;
use super::gamma::SimdGamma;
use super::normal::SimdNormal;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::special::LARGE_ORDER;
use crate::special::debye_shape;
use crate::special::ln_1p_mx;
use crate::special::ln_bessel_ke;
use crate::special::ln_gamma;
use crate::special::ln_quotient;
use crate::special::stirling_tail;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_VG_THRESHOLD: usize = 16;

/// Variance-gamma distribution: volatility `σ > 0`, gamma-clock variance rate `ν > 0`, drift `θ`, location `μ`; a [`Seeded`](crate::Seeded) stream draws it.
/// Its density is within `ε·Σ|ln terms|` (`1e-12` to `ν = 1/400` at `σ ∈ [0.1, 1]`, `|θ| ≤ ½`); NaN past every digit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdVarianceGamma<T> {
  sigma: T,
  nu: T,
  theta: T,
  mu: T,
  gamma: SimdGamma<T>,
}

/// A VG stream: the gamma-clock and normal sub-streams and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct VgState<T: SimdFloatExt, R: SimdRngExt> {
  gamma: GammaState<T, R>,
  normal: StreamState<T, R, 64>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdVarianceGamma<T> {
  /// Construct a VG$(\sigma, \nu, \theta, \mu)$ over an internal gamma
  /// subordinator with shape $1/\nu$ and scale $\nu$.
  pub fn new(sigma: T, nu: T, theta: T, mu: T) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    assert!(
      nu > T::zero(),
      "nu must satisfy `nu > T::zero()`, got nu = {nu:?}"
    );
    let shape = T::one() / nu;
    assert!(
      shape.is_finite(),
      "nu must satisfy `(1 / nu).is_finite()`, got nu = {nu:?}"
    );
    Self {
      sigma,
      nu,
      theta,
      mu,
      gamma: SimdGamma::new(shape, nu),
    }
  }

  /// The volatility `σ`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// The variance rate `ν` of the gamma clock.
  pub fn nu(&self) -> T {
    self.nu
  }

  /// The drift `θ`.
  pub fn theta(&self) -> T {
    self.theta
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    gamma: &mut GammaState<T, R>,
    normal: &mut StreamState<T, R, 64>,
    out: &mut [T],
  ) {
    if out.len() < SMALL_VG_THRESHOLD {
      for x in out.iter_mut() {
        let g = self.gamma.next(gamma);
        let z = SimdNormal::<T>::standard().next(normal);
        *x = self.mu + self.theta * g + self.sigma * g.sqrt() * z;
      }
      return;
    }
    let mu = T::splat(self.mu);
    let theta = T::splat(self.theta);
    let sigma = T::splat(self.sigma);
    let mut gbuf = [T::zero(); 64];
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gamma.fill(gamma, &mut gbuf);
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf);
      for (sub, (g8, z8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        gbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(zbuf.as_chunks::<8>().0.iter()),
      ) {
        let g = T::simd_from_array(*g8);
        let z = T::simd_from_array(*z8);
        let x = mu + theta * g + sigma * T::simd_sqrt(g) * z;
        *sub = T::simd_to_array(x);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gamma.fill(gamma, &mut gbuf[..n]);
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      for i in 0..n {
        rem[i] = self.mu + self.theta * gbuf[i] + self.sigma * gbuf[i].sqrt() * zbuf[i];
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let g = self.gamma.draw_with(rng);
    let z = SimdNormal::<T>::standard().draw_with(rng);
    self.mu + self.theta * g + self.sigma * g.sqrt() * z
  }

  fn params(&self) -> (f64, f64, f64, f64) {
    (
      self.sigma.to_f64().unwrap(),
      self.nu.to_f64().unwrap(),
      self.theta.to_f64().unwrap(),
      self.mu.to_f64().unwrap(),
    )
  }

  /// The density at order `1/ν − ½` past `LARGE_ORDER`, from the gamma clock: `ln N(1/ν − ½, d²/σ², θ²/σ² + 2/ν)` less the
  /// clock's `ln Γ(1/ν) + (ln ν)/ν` meets as `ln1pmx(g − 1)/ν − ½ ln g − (d − θg)²/(2σ²g)` at the maximiser `g`.
  fn large_order_pdf((sigma, nu, theta): (f64, f64, f64), d: f64, order: f64) -> f64 {
    if !d.is_finite() {
      return 0.0;
    }
    let (s2, c) = (sigma * sigma, (theta / sigma) * (theta / sigma));
    let (root_chi, psi) = (d.abs() / sigma, c + 2.0 / nu);
    // `g − 1` itself, `(χ − 1 − c)/(R + o + 1 + c)` with `ψ = 2o + 1 + c`, `o` the order: `g` rounds within an ulp of 1 long
    // before `1/ν` stops scaling that ulp.
    let r = order.hypot(root_chi * psi.sqrt());
    let gm1 = (root_chi * root_chi - 1.0 - c) / (r + order + 1.0 + c);
    let g = 1.0 + gm1;
    let gap = d - theta * g;
    // Debye's `−½ ln(1/ν − ½)` with the clock's `−½ ln ν` is `−½ ln_1p(−ν/2)`.
    let rests = debye_shape(order, root_chi * psi.sqrt())
      - 0.5 * (-0.5 * nu).ln_1p()
      - stirling_tail(nu)
      - 0.5 * std::f64::consts::TAU.ln();
    (std::f64::consts::LN_2 - 0.5 * (std::f64::consts::TAU * s2).ln() + ln_1p_mx(gm1) / nu
      - 0.5 * gm1.ln_1p()
      - 0.5 * (gap / (s2 * g)) * gap
      + rests)
      .exp()
  }
}

impl<T: SimdFloatExt> Sealed for SimdVarianceGamma<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdVarianceGamma<T> {
  type State<R: SimdRngExt> = VgState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (VgState<T, R>, u64) {
    let (gamma, _) = self.gamma.init::<R, S>(seed);
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let basis = seed.next_seed();
    (
      VgState {
        gamma,
        normal,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdVarianceGamma<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut VgState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.gamma, &mut state.normal, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut VgState<T, R>) -> T {
    let VgState { gamma, normal, buf } = state;
    buf.pop(|b| self.fill_parts(gamma, normal, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdVarianceGamma<T> {
  /// `μ + θG + σ√G·Z` from one scalar gamma and one scalar normal draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdVarianceGamma<T> {
  /// Madan–Carr–Chang eq. 23, evaluated in log space through the scaled
  /// Bessel function; at `x = μ` the $y^a K_a(y)$ limit $2^{a-1}\Gamma(a)$
  /// for $a = 1/\nu - 1/2 > 0$, `+∞` otherwise.
  fn pdf(&self, x: f64) -> Option<f64> {
    // The density vanishes at ±∞, where the log-space kernel would read ∞ − ∞.
    if x.is_infinite() {
      return Some(0.0);
    }
    let (sigma, nu, theta, mu) = self.params();
    let order = (2.0 - nu) / (2.0 * nu);
    if order >= LARGE_ORDER {
      return Some(Self::large_order_pdf((sigma, nu, theta), x - mu, order));
    }
    let a = 1.0 / nu - 0.5;
    let root = (2.0 * sigma * sigma / nu + theta * theta).sqrt();
    let (t_sigma, t_nu, t_gamma) = (sigma.ln(), nu.ln() / nu, ln_gamma(1.0 / nu));
    let log_norm =
      std::f64::consts::LN_2 - t_sigma - 0.5 * (2.0 * std::f64::consts::PI).ln() - t_nu - t_gamma;
    let d = x - mu;
    if d == 0.0 {
      if a <= 0.0 {
        return Some(f64::INFINITY);
      }
      let scale = 2.0 * ln_quotient(sigma, root);
      return Some((log_norm + a * scale + (a - 1.0) * std::f64::consts::LN_2 + ln_gamma(a)).exp());
    }
    let y = d.abs() * root / (sigma * sigma);
    // `θd/σ² − y` as `−2|d|/(ν(root + |θ|)) − |d|(|θ| − θ·sgn d)/σ²`, of one sign, where `2σ²/ν ≪ θ²` would leave two
    // large terms to cancel.
    let tilt = -2.0 * d.abs() / (nu * (root + theta.abs()))
      - d.abs() / sigma * ((theta.abs() - theta * d.signum()) / sigma);
    Some((log_norm + a * ln_quotient(d.abs(), root) + ln_bessel_ke(a, y) + tilt).exp())
  }

  fn mean(&self) -> Option<f64> {
    let (_, _, theta, mu) = self.params();
    Some(mu + theta)
  }

  fn variance(&self) -> Option<f64> {
    let (sigma, nu, theta, _) = self.params();
    Some(sigma * sigma + nu * theta * theta)
  }

  /// $(2\theta^3\nu^2 + 3\theta\sigma^2\nu)/(\sigma^2 + \nu\theta^2)^{3/2}$.
  fn skewness(&self) -> Option<f64> {
    let (sigma, nu, theta, _) = self.params();
    let var = sigma * sigma + nu * theta * theta;
    Some((2.0 * theta.powi(3) * nu * nu + 3.0 * theta * sigma * sigma * nu) / var.powf(1.5))
  }

  /// Excess kurtosis `κ₄/κ₂²` from the gamma clock's cumulants, `κ₄ = 3νσ⁴(2ρ² + 4ρ + 1)`, `ρ = νθ²/σ²`: $3\nu(2 - (1 + \rho)^{-2})$.
  fn kurtosis(&self) -> Option<f64> {
    let (sigma, nu, theta, _) = self.params();
    let rho = nu * (theta / sigma) * (theta / sigma);
    Some(3.0 * nu * (2.0 - 1.0 / ((1.0 + rho) * (1.0 + rho))))
  }

  /// $e^{\mu t}(1 - \theta\nu t - \tfrac12\sigma^2\nu t^2)^{-1/\nu}$ where the base is positive;
  /// `+∞` at and beyond its roots, where `E[e^{tX}]` diverges.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let (sigma, nu, theta, mu) = self.params();
    let base = 1.0 - theta * nu * t - 0.5 * sigma * sigma * nu * t * t;
    if base <= 0.0 {
      Some(f64::INFINITY)
    } else {
      Some((mu * t).exp() * base.powf(-1.0 / nu))
    }
  }

  /// $e^{iu\mu}(1 - i\theta\nu u + \tfrac12\sigma^2\nu u^2)^{-1/\nu}$.
  fn characteristic_function(&self, u: f64) -> Option<num_complex::Complex64> {
    use num_complex::Complex64;
    let (sigma, nu, theta, mu) = self.params();
    let base = Complex64::new(1.0 + 0.5 * sigma * sigma * nu * u * u, -theta * nu * u);
    Some(Complex64::new(0.0, mu * u).exp() * base.powf(-1.0 / nu))
  }
}

#[cfg(test)]
mod tests;

py_distribution!(PyVarianceGamma, SimdVarianceGamma,
  sig: (sigma, nu, theta, mu, seed=None, dtype=None),
  params: (sigma: f64, nu: f64, theta: f64, mu: f64)
);
