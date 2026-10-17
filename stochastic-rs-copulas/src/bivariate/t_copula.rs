//! # Bivariate Student-t copula
//!
//! $$
//! C_{\rho,\nu}(u,v) = T_{\rho,\nu}\!\big(t_\nu^{-1}(u),\, t_\nu^{-1}(v)\big),
//! $$
//! where $T_{\rho,\nu}$ is the bivariate Student-t CDF with correlation
//! $\rho \in (-1, 1)$ and degrees of freedom $\nu > 0$, and $t_\nu^{-1}$
//! the univariate Student-t quantile.
//!
//! - **Kendall's tau:** $\tau = \tfrac{2}{\pi}\arcsin\rho$ (same as Gaussian).
//! - **Tail dependence:** $\lambda_U = \lambda_L = 2\, t_{\nu+1}\!\big(
//!   -\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big)$ — symmetric and strictly
//!   positive for finite $\nu$, in contrast to Gaussian.
//!
//! In the limit $\nu \to \infty$ the t-copula collapses to the Gaussian
//! copula.
//!
//! The bivariate CDF is Dunnett & Sobel's (1954) finite sum for an integer $\nu \le 1000$, else Genz's (2004) eq. (21), its integral by
//! tanh-sinh to $4 \cdot 10^{-16}$; the value inherits the Student-t quantile's and cdf's error near $u = \tfrac12$ and at large or small $\nu$.
//!
//! Reference: Embrechts, P., Lindskog, F., McNeil, A.J. (2003),
//! "Modelling Dependence with Copulas and Applications to Risk
//! Management", in *Handbook of Heavy Tailed Distributions in Finance*,
//! Elsevier, ch. 8.
//! Reference: Dunnett, C.W., Sobel, M. (1954), "A bivariate generalization of Student's t-distribution, with tables for certain special cases", *Biometrika* 41(1/2), 153-169, DOI 10.2307/2333013.
//! Reference: Genz, A. (2004), "Numerical computation of rectangular bivariate and trivariate normal and t probabilities", *Statistics and Computing* 14(3), 251-260, DOI 10.1023/B:STCO.0000035304.20635.31 (equations numbered as in the author's preprint).
//! Reference: Takahasi, H., Mori, M. (1974), "Double exponential formulas for numerical integration", *Publ. RIMS* 9(3), 721-741, DOI 10.2977/prims/1195192451.
//! Reference: Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009), "Pair-copula constructions of multiple dependence", *Insurance: Mathematics and Economics* 44(2), 182-198, App. B.2 (the h-function inverse), DOI 10.1016/j.insmatheco.2007.02.001.

use std::f64::consts::FRAC_2_PI;
use std::f64::consts::FRAC_PI_2;
use std::f64::consts::TAU;
use std::sync::OnceLock;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::ln_gamma;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::error::CopulaError;
use crate::student_t;
use crate::student_t::Law;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::tau_out_of_range;

/// Below this a quantile's square, `ν` added, stays finite: past it the h-function and its inverse take logarithms.
const LARGE: f64 = 1e150;

#[derive(Debug, Clone)]
pub struct TCopula {
  theta: Option<f64>,
  nu: f64,
}

impl Default for TCopula {
  fn default() -> Self {
    Self {
      theta: None,
      nu: 4.0,
    }
  }
}

impl TCopula {
  /// A t copula with correlation `rho` and `nu` degrees of freedom; panics unless `-1 < rho < 1` and `0 < nu < ∞`.
  pub fn new(rho: f64, nu: f64) -> Self {
    Self::try_new(rho, nu).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(rho: f64, nu: f64) -> Result<Self, CopulaError> {
    student_t::check_nu(nu)?;
    let copula = Self {
      theta: Some(rho),
      nu,
    };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The t copula whose Kendall's τ is `tau`, at `nu` degrees of freedom.
  pub fn from_tau(tau: f64, nu: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?, nu)
  }

  pub fn nu(&self) -> f64 {
    self.nu
  }

  /// Panics under the rule of `check_nu`: unless `0 < nu < ∞` and `nu / 2 > 0`.
  pub fn set_nu(&mut self, nu: f64) {
    student_t::check_nu(nu).unwrap_or_else(|e| panic!("{e}"));
    self.nu = nu;
  }

  /// `u = t_ν(t_{ν+1}⁻¹(y)√((ν + x²)(1 − ρ²)/(ν + 1)) + ρx)`, `x = t_ν⁻¹(v)`, in logarithms where a quantile passes
  /// `LARGE`, and the two-point law the `v → 0, 1` limit of `∂_v C` leaves at the edges.
  pub(crate) fn inverse(rho: f64, nu: f64) -> impl Fn(f64, f64) -> f64 {
    let (law, wide) = (Law::new(nu), Law::new(nu + 1.0));
    let one_minus_rho2 = (1.0 - rho) * (1.0 + rho);
    let edge = rho * ((nu + 1.0) / one_minus_rho2).sqrt();
    let ln_spread = 0.5 * (one_minus_rho2 / (nu + 1.0)).ln();
    move |y, v| {
      let (x, q) = (law.quantile(v), wide.quantile(y));
      if v <= 0.0 || v >= 1.0 {
        return if y < 1.0 && q.value <= -x.value.signum() * edge {
          0.0
        } else {
          1.0
        };
      }
      if x.value.abs() < LARGE && q.value.is_finite() {
        let scale = ((nu + x.value * x.value) * one_minus_rho2 / (nu + 1.0)).sqrt();
        return law.cdf_at(q.value * scale + rho * x.value);
      }
      if x.value.abs() < LARGE {
        let ln_scale = 0.5 * (nu + x.value * x.value).ln() + ln_spread;
        return law.cdf(q.value.signum(), q.ln_abs() + ln_scale);
      }
      // `B = xw`, `w = ρ + sgn(x) q √((1 + ν/x²)(1 − ρ²)/(ν + 1))`, whose `ν/x²` is below every rounding unit here.
      let (sign_w, ln_w) = if q.value.is_finite() {
        let w = rho + x.value.signum() * q.value * ln_spread.exp();
        (w.signum(), w.abs().ln())
      } else {
        (x.value.signum() * q.value.signum(), q.ln_abs() + ln_spread)
      };
      law.cdf_scaled(x, x.value.signum() * sign_w, ln_w)
    }
  }

  /// `∂_v C = t_{ν+1}((x − ρy)√((ν + 1)/((ν + y²)(1 − ρ²))))`, `x, y = t_ν⁻¹(u), t_ν⁻¹(v)`, in logarithms where a quantile
  /// passes `LARGE`, and its `v → 0, 1` limit `t_{ν+1}(±ρ√((ν + 1)/(1 − ρ²)))` at the edges.
  pub(crate) fn h(rho: f64, nu: f64) -> impl Fn(f64, f64) -> f64 {
    let (law, wide) = (Law::new(nu), Law::new(nu + 1.0));
    let sqrt_one_minus_rho2 = ((1.0 - rho) * (1.0 + rho)).sqrt();
    let nu_plus_one = nu + 1.0;
    let boundary_arg = rho * nu_plus_one.sqrt() / sqrt_one_minus_rho2;
    let reach = nu_plus_one.sqrt() / sqrt_one_minus_rho2;
    move |u, v| {
      if u >= 1.0 {
        return 1.0;
      }
      if v <= 0.0 || v >= 1.0 {
        return wide.cdf_at(if v <= 0.0 {
          boundary_arg
        } else {
          -boundary_arg
        });
      }
      if u <= 0.0 {
        return 0.0;
      }
      let (x, y) = (law.quantile(u), law.quantile(v));
      if x.value.is_finite() && y.value.abs() < LARGE {
        let gap = x.value - rho * y.value;
        // At the median a tiny `ν` overflows the scale: the argument is 0 there.
        if gap == 0.0 {
          return wide.cdf_at(0.0);
        }
        let scale = (nu_plus_one / (nu + y.value * y.value)).sqrt() / sqrt_one_minus_rho2;
        return wide.cdf_at(gap * scale);
      }
      // `x/√(ν + y²)` and `y/√(ν + y²)` from the logarithms; past `f64::MAX` the first carries the argument alone.
      let (ln_x_over_root, y_over_root) = if y.value.abs() < LARGE {
        let root = (nu + y.value * y.value).sqrt();
        (x.ln_abs() - root.ln(), y.value / root)
      } else {
        let lift = 0.5 * (nu * (-2.0 * y.ln_abs()).exp()).ln_1p();
        (law.ln_ratio(x, y) - lift, y.value.signum() * (-lift).exp())
      };
      let ratio = x.value.signum() * ln_x_over_root.exp();
      if ratio.is_finite() {
        wide.cdf_at((ratio - rho * y_over_root) * reach)
      } else {
        wide.cdf(x.value.signum(), ln_x_over_root + reach.ln())
      }
    }
  }

  /// Bivariate Student-t density $f_{\rho,\nu}(x, y)$.
  fn bivariate_t_pdf(x: f64, y: f64, rho: f64, nu: f64) -> f64 {
    let one_minus_rho2 = 1.0 - rho * rho;
    let log_norm = ln_gamma(0.5 * (nu + 2.0))
      - ln_gamma(0.5 * nu)
      - (nu * std::f64::consts::PI).ln()
      - 0.5 * one_minus_rho2.ln();
    let q = (x * x - 2.0 * rho * x * y + y * y) / (nu * one_minus_rho2);
    let log_kernel = -0.5 * (nu + 2.0) * (1.0 + q).ln();
    (log_norm + log_kernel).exp()
  }
}

impl BivariateExt for TCopula {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::TCopula
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (-1.0, 1.0)
  }

  /// `ρ = ±1` divides by `√(1 − ρ²)`.
  fn invalid_thetas(&self) -> Vec<f64> {
    vec![-1.0, 1.0]
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
        out[i] = 0.0;
        continue;
      }
      let xx = student_t::quantile(u, nu);
      let yy = student_t::quantile(v, nu);
      let num = Self::bivariate_t_pdf(xx, yy, rho, nu);
      let den = student_t::ln_pdf(xx, nu).exp() * student_t::ln_pdf(yy, nu).exp();
      out[i] = num / den;
    }
    Ok(out)
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (rho, nu) = (self.theta.unwrap(), self.nu);
    Ok(
      x.rows()
        .into_iter()
        .map(|r| t_copula_cdf(r[0], r[1], rho, nu))
        .collect(),
    )
  }

  /// `TCopula::h`; NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    Ok(conditional_cdf(x, Self::h(self.theta.unwrap(), self.nu)))
  }

  /// `TCopula::inverse` (Aas et al. App. B.2); NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    conditional_quantiles(y, v, Self::inverse(self.theta.unwrap(), self.nu))
  }

  fn tau_from_theta(&self, theta: f64) -> f64 {
    2.0 / std::f64::consts::PI * theta.asin()
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    if !(tau > -1.0 && tau < 1.0) {
      return Err(tau_out_of_range(tau, "-1.0 < tau < 1.0".into()));
    }
    // `ρ` rounds onto the excluded `±1` once `|τ| > 1 − 6e-9`.
    admitted(self, tau, (std::f64::consts::FRAC_PI_2 * tau).sin())
  }

  /// Symmetric tail dependence $\lambda_L = \lambda_U = 2\,t_{\nu+1}\!\big(
  /// -\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\big)$, strictly positive for every
  /// finite $\nu$ and $\rho \in (-1,1)$ — in contrast to the Gaussian
  /// copula's asymptotic independence
  /// ([`crate::bivariate::gaussian::GaussianCopula`]).
  /// Reference: Embrechts, Lindskog, McNeil (2003), §3.2 (module header).
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    let rho = self.theta.unwrap();
    let nu = self.nu;
    let arg = -(((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt());
    let lambda = 2.0 * student_t::cdf(arg, nu + 1.0);
    Ok(TailDependence {
      lower: lambda,
      upper: lambda,
    })
  }
}

/// Integer `ν` up to this takes the Dunnett–Sobel sums, whose `ν/2` terms cost less than the quadrature's 449 nodes.
const DUNNETT_SOBEL_MAX_NU: f64 = 1000.0;

/// `C_{ρ,ν}(u, v) = T_ν(t_ν⁻¹(u), t_ν⁻¹(v); ρ)` on the closed square, the bivariate cdf of both t copulas: 0 on a zero
/// coordinate, the other coordinate at a unit one.
pub(crate) fn t_copula_cdf(u: f64, v: f64, rho: f64, nu: f64) -> f64 {
  if u <= 0.0 || v <= 0.0 {
    return 0.0;
  }
  if u >= 1.0 {
    return v;
  }
  if v >= 1.0 {
    return u;
  }
  bivariate_t_cdf(
    student_t::quantile(u, nu),
    student_t::quantile(v, nu),
    rho,
    nu,
  )
}

/// `T_ν(h, k; ρ) = P(X ≤ h, Y ≤ k)` of the standard bivariate t; an infinite limit, an overflowed quantile, takes the
/// marginal limit.
fn bivariate_t_cdf(h: f64, k: f64, rho: f64, nu: f64) -> f64 {
  if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
    return 0.0;
  }
  if h == f64::INFINITY {
    return student_t::cdf(k, nu);
  }
  if k == f64::INFINITY {
    return student_t::cdf(h, nu);
  }
  if nu.fract() == 0.0 && nu <= DUNNETT_SOBEL_MAX_NU {
    dunnett_sobel(h, k, rho, nu as u32)
  } else {
    generalized_plackett(h, k, rho, nu)
  }
}

/// Dunnett & Sobel's (1954) sums as Genz's BVTL evaluates them, in `sin α = h/√(ν + h²)`, `cos α = √ν/√(ν + h²)` so no
/// square overflows; the odd leading term is `T₁((h, k)/√ν)`, Genz (2004) eq. (20) at `ν = 1` in closed form.
fn dunnett_sobel(h: f64, k: f64, rho: f64, n: u32) -> f64 {
  let rn = f64::from(n).sqrt();
  let (rh, rk) = (rn.hypot(h), rn.hypot(k));
  let (sa, ca, sb, cb) = (h / rh, rn / rh, k / rk, rn / rk);
  let om = ((1.0 - rho) * (1.0 + rho)).sqrt();
  let (xk, yk, sk) = beta_argument(sb * ca - rho * sa * cb, om * cb);
  let (xh, yh, sh) = beta_argument(sa * cb - rho * sb * ca, om * ca);
  let (c2a, c2b) = (ca * ca, cb * cb);
  if n.is_multiple_of(2) {
    let mut total = om.atan2(-rho) / TAU;
    let (mut gh, mut gk) = (0.25 * sa, 0.25 * sb);
    let (mut ik, mut dk) = (FRAC_2_PI * xk.atan2(yk), FRAC_2_PI * xk * yk);
    let (mut ih, mut dh) = (FRAC_2_PI * xh.atan2(yh), FRAC_2_PI * xh * yh);
    for j in 1..=n / 2 {
      let j = f64::from(j);
      total += gh * (1.0 + sk * ik) + gk * (1.0 + sh * ih);
      ik += dk;
      dk *= 2.0 * j * yk * yk / (2.0 * j + 1.0);
      ih += dh;
      dh *= 2.0 * j * yh * yh / (2.0 * j + 1.0);
      gh *= (2.0 * j - 1.0) / (2.0 * j) * c2a;
      gk *= (2.0 * j - 1.0) / (2.0 * j) * c2b;
    }
    return floor_at_zero(total);
  }
  let (alpha, beta) = (h.atan2(rn), k.atan2(rn));
  let one_minus_z = 2.0 * (0.5 * (alpha - beta)).sin().powi(2) + (1.0 - rho) * ca * cb;
  let one_plus_z = 2.0 * (0.5 * (alpha + beta)).sin().powi(2) + (1.0 + rho) * ca * cb;
  let asin_z = (sa * sb + rho * ca * cb).atan2((one_minus_z * one_plus_z).sqrt());
  let mut total = 0.25 + (alpha + beta + asin_z) / TAU;
  let (mut gh, mut gk) = (sa * ca / TAU, sb * cb / TAU);
  let (mut ik, mut dk, mut ih, mut dh) = (xk, xk, xh, xh);
  for j in 1..=(n - 1) / 2 {
    let j = f64::from(j);
    total += gh * (1.0 + sk * ik) + gk * (1.0 + sh * ih);
    dk *= (2.0 * j - 1.0) * yk * yk / (2.0 * j);
    ik += dk;
    dh *= (2.0 * j - 1.0) * yh * yh / (2.0 * j);
    ih += dh;
    gh *= 2.0 * j / (2.0 * j + 1.0) * c2a;
    gk *= 2.0 * j / (2.0 * j + 1.0) * c2b;
  }
  floor_at_zero(total)
}

/// Both forms cancel just below 0 where the probability is 0 (`−2e-16` for the sums); a NaN passes through.
fn floor_at_zero(t: f64) -> f64 {
  if t < 0.0 { 0.0 } else { t }
}

/// `(√x, √(1 − x), sign d)` of `x = d²/(d² + e²)`, the incomplete beta argument of one Dunnett–Sobel sum.
fn beta_argument(d: f64, e: f64) -> (f64, f64, f64) {
  let r = d.hypot(e);
  (d.abs() / r, e / r, if d < 0.0 { -1.0 } else { 1.0 })
}

/// Genz (2004) eq. (21) at `θ = s(π/2 − φ)`, `s = sign ρ`: `T_ν(b, s) − s/(2π) ∫₀^{acos|ρ|} (1 + c q)^{−ν/2} dφ`, with
/// `q = (a − sb)²/sin²φ + sab/cos²(φ/2)` in `(a, b) = (h, k)/m`, `c = m²/ν`, `m = max(|h|, |k|, √ν)`; `c` may overflow.
fn generalized_plackett(h: f64, k: f64, rho: f64, nu: f64) -> f64 {
  let s = if rho < 0.0 { -1.0 } else { 1.0 };
  let base = if s > 0.0 {
    student_t::cdf(h.min(k), nu)
  } else {
    (student_t::cdf(h, nu) - student_t::cdf(-k, nu)).max(0.0)
  };
  let m = h.abs().max(k.abs()).max(nu.sqrt());
  let (a, b, c) = (h / m, k / m, m * m / nu);
  let (ln_c, inv_c) = (2.0 * m.ln() - nu.ln(), nu / (m * m));
  let (diff2, cross) = ((a - s * b).powi(2), s * a * b);
  let top = rho.abs().acos();
  let integral: f64 = tanh_sinh()
    .iter()
    .map(|&(x, w)| {
      let (sh, ch) = (0.5 * top * x).sin_cos();
      let q = diff2 / (2.0 * sh * ch).powi(2) + cross / (ch * ch);
      let cq = c * q;
      // `ln(1 + cq) = ln c + ln(q + 1/c)` where `cq` overflows, `ln_1p` everywhere else for its accuracy at large `ν`.
      let log = if cq.is_finite() {
        cq.ln_1p()
      } else {
        ln_c + (q + inv_c).ln()
      };
      w * (-0.5 * nu * log).exp()
    })
    .sum();
  floor_at_zero(base - s * top * integral / TAU)
}

/// Tanh-sinh nodes `(x, w)` on `(0, 1)`, step `1/64` over `|t| ≤ 3.5`; at step `1/32` the endpoint layer at `h ≈ sk`
/// leaves `3e-12` once `ν` is in the thousands.
fn tanh_sinh() -> &'static [(f64, f64)] {
  static NODES: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
  NODES.get_or_init(|| {
    (-224..=224)
      .map(|i| {
        let t = f64::from(i) / 64.0;
        let s = FRAC_PI_2 * t.sinh();
        let w = FRAC_PI_2 * t.cosh() / (128.0 * s.cosh().powi(2));
        (1.0 / (1.0 + (-2.0 * s).exp()), w)
      })
      .collect()
  })
}

#[cfg(test)]
mod tests;
