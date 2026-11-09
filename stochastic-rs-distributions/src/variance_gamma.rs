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

use std::mem::MaybeUninit;

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
use crate::special::PLAIN_LOG_SIZE;
use crate::special::add_parts;
use crate::special::debye_shape;
use crate::special::exp_parts;
use crate::special::ldexp;
use crate::special::ln_1p_mx;
use crate::special::ln_bessel_ke;
use crate::special::ln_gamma;
use crate::special::ln_quotient;
use crate::special::mul_parts;
use crate::special::stirling_tail;
use crate::special::two_prod;
use crate::special::two_sum;
use crate::special::two_sum_far;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;
use crate::traits::distribution::as_uninit_mut;

const SMALL_VG_THRESHOLD: usize = 16;

/// The VG(σ, ν, θ, μ) law, `ν` its clock's rate: density within `ε(S + 16)` to order `1/ν − ½ = 50`
/// (`S` its logs' summed sizes), then `4ε(|ln f| + |ln σ| + 20)` or `ε(|ln f| + |ln σ| + 400)/16`.
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
    out: &mut [MaybeUninit<T>],
  ) {
    if out.len() < SMALL_VG_THRESHOLD {
      for x in out.iter_mut() {
        let g = self.gamma.next(gamma);
        let z = SimdNormal::<T>::standard().next(normal);
        x.write(self.mu + self.theta * g + self.sigma * g.sqrt() * z);
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
      self.gamma.fill(gamma, as_uninit_mut(&mut gbuf));
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
        *sub = T::simd_to_array(x).map(MaybeUninit::new);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gamma.fill(gamma, as_uninit_mut(&mut gbuf[..n]));
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      for i in 0..n {
        rem[i].write(self.mu + self.theta * gbuf[i] + self.sigma * gbuf[i].sqrt() * zbuf[i]);
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
  #[inline(always)]
  fn large_order_pdf(
    (sigma, nu, theta): (f64, f64, f64),
    (d, d_lo): (f64, f64),
    order: f64,
    p: i32,
  ) -> f64 {
    if !d.is_finite() {
      return 0.0;
    }
    let c = (theta / sigma) * (theta / sigma);
    let (root_chi, psi) = (d.abs() / sigma, c + 2.0 / nu);
    // `g − 1 = (χ − 1 − c)/D`, `D = R + o + 1 + c`, `ψ = 2o + 1 + c`, `o` the order, `χ − c = (d − θ)(d + θ)/σ²` from the exact
    // `d − θ`; past `q = θ(d + θ)/σ²D = ½`, where `θ(g − 1)` cancels `d − θ`, the gap `d − θg` is `((d − θ)(o + 1 + Δ) + θ)/D`.
    let (h, h_lo) = two_sum_far(d, -theta);
    if !(nu > limits::NORMAL_CLOCK && c * nu < limits::NARROW_DIFFUSION && root_chi <= FAR)
      || !(h.is_finite() && (d + theta).is_finite())
    {
      return Self::extreme_large_order((sigma, nu, theta), (d, d_lo), order, p);
    }
    let near = h + (h_lo + d_lo);
    let r = order.hypot(root_chi * psi.sqrt());
    let den = r + order + 1.0 + c;
    let gm1 = ((near / sigma) * ((d + theta) / sigma) - 1.0) / den;
    // Below `g = −⅓` `ln_1p_mx(g)` leaves its series and `1 + g` carries `g`'s rounding, which `1/ν` multiplies.
    if gm1 < -1.0 / 3.0 {
      return large_order::density((sigma, nu, theta), (d, d_lo), p);
    }
    let q = (theta / sigma) * ((d + theta) / sigma) / den;
    let gap = if q < 0.5 {
      (near - theta * gm1) / sigma
    } else {
      let tilt = (theta / sigma) * (d / sigma) / r;
      // `Δ = R − θd/σ²`, by `R² − (θd/σ²)² = o² + 2χ/ν` where `θd > 0`.
      let excess = if tilt > 0.0 {
        ((order / r) * order + (root_chi / r) * root_chi * (2.0 / nu)) / (1.0 + tilt)
      } else {
        r * (1.0 - tilt)
      };
      (near / sigma) * ((order + 1.0 + excess) / den) + (theta / sigma) / den
    };
    // Debye's `−½ ln(1/ν − ½)` with the clock's `−½ ln ν` is `−½ ln_1p(−ν/2)`.
    let rests = debye_shape(order, root_chi * psi.sqrt())
      - 0.5 * (-0.5 * nu).ln_1p()
      - stirling_tail(nu)
      - std::f64::consts::TAU.ln();
    let ln_sigma = sigma.ln();
    let a = std::f64::consts::LN_2 - ln_sigma + ln_1p_mx(gm1) / nu
      - 0.5 * gm1.ln_1p()
      - 0.5 * gap * gap / (1.0 + gm1)
      + rests;
    // Past `|a| = 760` the plain exponent, a few `ε` of it off, settles `0` or `+∞` alone.
    if a.abs() + ln_sigma.abs() > PARTS_FROM && a.abs() < 760.0 {
      return large_order::density((sigma, nu, theta), (d, d_lo), p);
    }
    exp_pow2(a, p)
  }

  /// [`Self::large_order_pdf`] past its scales: a normal clock from `ν = 2⁻¹⁶⁰` down, `μ + θG` from `νθ²/σ² = 2¹⁰⁰`, `0` past
  /// `|d|/σ = FAR`, and the law of `X/2`, exactly scaled, where `d ∓ θ` overflows.
  #[cold]
  #[inline(never)]
  fn extreme_large_order(
    (sigma, nu, theta): (f64, f64, f64),
    (d, d_lo): (f64, f64),
    order: f64,
    p: i32,
  ) -> f64 {
    if nu <= limits::NORMAL_CLOCK {
      return limits::normal_clock((sigma, nu, theta), (d, d_lo), p);
    }
    if (theta / sigma) * (theta / sigma) * nu >= limits::NARROW_DIFFUSION {
      return limits::gamma_clock((nu, theta), (d, d_lo), p);
    }
    if (d / sigma).abs() > FAR {
      return 0.0;
    }
    Self::large_order_pdf(
      (0.5 * sigma, nu, 0.5 * theta),
      (0.5 * d, 0.5 * d_lo),
      order,
      p - 1,
    )
  }

  /// The density at `x` times `2ᵖ`; where `x − μ` passes MAX, on `X/2 ~ VG(σ/2, ν, θ/2, μ/2)`,
  /// whose halved subnormal `σ` moves it by `σ²/νθ²`, under `2⁻¹⁰⁰` where the gap holds a density.
  fn density_at((sigma, nu, theta, mu): (f64, f64, f64, f64), x: f64, p: i32) -> f64 {
    if (x - mu).is_infinite() {
      let half = if sigma.is_normal() {
        0.5 * sigma
      } else {
        sigma
      };
      return Self::density_at((half, nu, 0.5 * theta, 0.5 * mu), 0.5 * x, p - 1);
    }
    let order = (2.0 - nu) / (2.0 * nu);
    if order >= LARGE_ORDER {
      return Self::large_order_pdf((sigma, nu, theta), two_sum_far(x, -mu), order, p);
    }
    let a = 1.0 / nu - 0.5;
    let root = (2.0 * sigma * sigma / nu + theta * theta).sqrt();
    let (t_sigma, t_nu, t_gamma) = (sigma.ln(), nu.ln() / nu, ln_gamma(1.0 / nu));
    let log_norm =
      std::f64::consts::LN_2 - t_sigma - 0.5 * (2.0 * std::f64::consts::PI).ln() - t_nu - t_gamma;
    let size = t_sigma.abs() + t_nu.abs() + t_gamma.abs() + 1.62;
    let d = x - mu;
    if d == 0.0 {
      if a <= 0.0 {
        return f64::INFINITY;
      }
      let (scale, gamma_a) = (2.0 * ln_quotient(sigma, root), ln_gamma(a));
      let at_mu = size + (a * scale).abs() + gamma_a.abs();
      if at_mu.is_nan() || at_mu > PLAIN_LOG_SIZE {
        return small_order::density((sigma, nu, theta), (0.0, 0.0), p);
      }
      return exp_pow2(
        log_norm + a * scale + (a - 1.0) * std::f64::consts::LN_2 + gamma_a,
        p,
      );
    }
    let y = d.abs() * root / (sigma * sigma);
    // `θd/σ² − y` as `−2|d|/(ν(root + |θ|)) − |d|(|θ| − θ·sgn d)/σ²`, of one sign, where
    // `2σ²/ν ≪ θ²` would leave two large terms to cancel.
    let tilt = -2.0 * d.abs() / (nu * (root + theta.abs()))
      - d.abs() / sigma * ((theta.abs() - theta * d.signum()) / sigma);
    let (power, kernel) = (a * ln_quotient(d.abs(), root), ln_bessel_ke(a, y));
    if !(size + power.abs() + kernel.abs() - tilt <= PLAIN_LOG_SIZE && y.is_normal()) {
      return small_order::density((sigma, nu, theta), two_sum_far(x, -mu), p);
    }
    exp_pow2(log_norm + power + kernel + tilt, p)
  }

  /// `ln b`, `b = 1 − X`, `X = θνt + ½σ²νt²`: plain where `PLAIN` bounds `X`'s rounding, else in two parts, `ln_1p(−X)` below
  /// `X = ½` and `1 − X` in two parts, exactly within `2⁻⁴⁰` of `X`'s terms; `None` where `b ≤ 0`.
  fn ln_base(sigma: f64, nu: f64, theta: f64, t: f64) -> Option<f64> {
    let plain_nt = nu * t;
    let (drift, spread) = (theta * plain_nt, 0.5 * sigma * sigma * t * plain_nt);
    // A product of nonzero factors that leaves the normal range rounds away its digits, which `1 − X` may still carry.
    let normal = (sigma * sigma).is_normal() && plain_nt.is_normal() && spread.is_normal();
    if t != 0.0 && !t.is_nan() && !(normal && (theta == 0.0 || drift.is_normal())) {
      return small_order::ln_base_far((sigma, nu, theta), t);
    }
    let x = drift + spread;
    let b = 1.0 - x;
    if drift.abs() + spread <= PLAIN * nu * b.min(1.0) {
      return Some(if x < 0.5 { (-x).ln_1p() } else { b.ln() });
    }
    let nt = two_prod(nu, t);
    let drift = mul_parts(nt, (theta, 0.0));
    let spread = mul_parts(mul_parts(two_prod(sigma, sigma), nt), (0.5 * t, 0.0));
    let (x, x_lo) = add_parts(drift, spread);
    if !x_lo.is_finite() {
      return if t.is_nan() {
        Some(t)
      } else {
        small_order::ln_base_far((sigma, nu, theta), t)
      };
    }
    if x < 0.5 {
      return Some((-x).ln_1p());
    }
    // `1 − x` is exact from `x = ½` to 2, so this rounds once.
    let b = (1.0 - x) - x_lo;
    let b = if b.abs() > TIGHT * (drift.0.abs() + spread.0.abs()) {
      b
    } else {
      exact_base((sigma, nu, theta), t)
    };
    (b > 0.0).then(|| b.ln())
  }
}

/// Past this `|d|/σ`, with `νθ²/σ²` under `2¹⁰⁰` and `ν` over `2⁻¹⁶⁰`, the point lies `2⁴⁰⁰` of the law's scale out.
const FAR: f64 = 3.273_390_607_896_142e150;

/// Past this `|ln f| + |ln σ|` the large-order density sums its exponent in two parts, where the plain sum's rounding, a few
/// `ε` of its largest terms, would near `1e-12`.
const PARTS_FROM: f64 = 512.0;

/// Up to this `(|θνt| + ½σ²νt²)/(ν min(b, 1))` the plain `X`'s rounding moves `ln(b)/ν` by under `2.5·PLAIN·ε`.
const PLAIN: f64 = 64.0;

/// Past this fraction of `X`'s terms the two-part base, within about `6ε²` of their size, errs by under `2⁻⁶³` of itself.
const TIGHT: f64 = 1.0 / (1u64 << 40) as f64;

/// `e^a·2ᵖ`, at `p ≠ 0` rounded once, so a subnormal density of the law of `X/2ᵏ` does not round twice.
fn exp_pow2(a: f64, p: i32) -> f64 {
  if p == 0 {
    a.exp()
  } else {
    exp_parts((a, 0.0), 1.0, p)
  }
}

/// `1 − θνt − ½σ²νt²` exactly: Dekker's products' parts summed by Shewchuk's Grow-Expansion (1997, Thm 10, zero elimination),
/// whose nonoverlapping parts add from the smallest to within an ulp.
fn exact_base((sigma, nu, theta): (f64, f64, f64), t: f64) -> f64 {
  let mut parts = [0.0_f64; 24];
  let mut len = 0;
  let mut grow = |value: f64| {
    let (mut q, mut kept) = (value, 0);
    for i in 0..len {
      let (s, e) = two_sum(q, parts[i]);
      q = s;
      if e != 0.0 {
        parts[kept] = e;
        kept += 1;
      }
    }
    if q != 0.0 {
      parts[kept] = q;
      kept += 1;
    }
    len = kept;
  };
  grow(1.0);
  let (nt, s) = (two_prod(nu, t), two_prod(sigma, sigma));
  for a in [nt.0, nt.1] {
    let (p, e) = two_prod(a, theta);
    grow(-p);
    grow(-e);
    for c in [s.0, s.1] {
      let (q, f) = two_prod(a, c);
      for d in [q, f] {
        let (r, g) = two_prod(d, 0.5 * t);
        grow(-r);
        grow(-g);
      }
    }
  }
  parts[..len].iter().sum()
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
  fn fill<R: SimdRngExt>(&self, state: &mut VgState<T, R>, out: &mut [MaybeUninit<T>]) {
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
    Some(Self::density_at(self.params(), x, 0))
  }

  fn mean(&self) -> Option<f64> {
    let (_, _, theta, mu) = self.params();
    Some(mu + theta)
  }

  fn variance(&self) -> Option<f64> {
    let (sigma, nu, theta, _) = self.params();
    let variance = sigma * sigma + nu * theta * theta;
    // Two subnormal terms round twice; at `2⁶⁰⁰` the sum rounds once, `νθ² < 2⁻¹⁰²²` holding `θ` under 3.
    if variance < f64::MIN_POSITIVE {
      let (s, t) = (ldexp(sigma, 300), ldexp(theta, 300));
      return Some(ldexp(s * s + nu * t * t, -600));
    }
    Some(variance)
  }

  /// `κ₃/κ₂^{3/2}` from the gamma clock's cumulants, `t(3 + 2ρ)/(1 + ρ)^{3/2}`, `t = νθ/σ`, `ρ = tθ/σ`, past `ρ = 1` as
  /// `sgn θ·√ν(2 + 3/ρ)/(1 + 1/ρ)^{3/2}`: no power of a parameter leaves the range where the value does not.
  fn skewness(&self) -> Option<f64> {
    let (sigma, nu, theta, _) = self.params();
    let ratio = theta / sigma;
    // `νθ/σ` in an order whose partial product is normal where `θ/σ` is not.
    let t = if ratio.is_normal() || theta == 0.0 {
      nu * ratio
    } else if (nu * theta).is_normal() {
      nu * theta / sigma
    } else {
      theta * (nu / sigma)
    };
    let rho = t * ratio;
    Some(if rho <= 1.0 {
      t * (3.0 + 2.0 * rho) / (1.0 + rho).powf(1.5)
    } else {
      theta.signum() * nu.sqrt() * (2.0 + 3.0 / rho) / (1.0 + 1.0 / rho).powf(1.5)
    })
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
    let Some(ln_base) = Self::ln_base(sigma, nu, theta, t) else {
      return Some(f64::INFINITY);
    };
    // One exponent: a location that cancels the clock past `e^{±709}` would overflow either factor alone.
    let (p, p_lo) = two_prod(mu, t);
    let clock = ln_base / nu;
    if p.is_infinite() && clock == p {
      // Both terms past the doubles: the larger by its logarithm decides the sign of the exponent.
      let larger = mu.abs().ln() + t.abs().ln() > ln_base.abs().ln() - nu.ln();
      return Some(if larger == (p > 0.0) {
        f64::INFINITY
      } else {
        0.0
      });
    }
    let low = if p.is_finite() { p_lo } else { 0.0 };
    Some((p - clock + low).exp())
  }

  /// $e^{iu\mu}(1 - i\theta\nu u + \tfrac12\sigma^2\nu u^2)^{-1/\nu}$.
  fn characteristic_function(&self, u: f64) -> Option<num_complex::Complex64> {
    use num_complex::Complex64;
    let (sigma, nu, theta, mu) = self.params();
    let base = Complex64::new(1.0 + 0.5 * sigma * sigma * nu * u * u, -theta * nu * u);
    Some(Complex64::new(0.0, mu * u).exp() * base.powf(-1.0 / nu))
  }
}

mod large_order;
mod limits;
mod small_order;

#[cfg(test)]
mod tests;

py_distribution!(PyVarianceGamma, SimdVarianceGamma,
  sig: (sigma, nu, theta, mu, seed=None, dtype=None),
  params: (sigma: f64, nu: f64, theta: f64, mu: f64)
);
