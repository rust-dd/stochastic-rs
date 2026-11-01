//! The truncated normal's draws: rejection on the base normal, Robert's one-sided proposals as offsets from the near
//! bound, the inverse cdf, and past `μ ± σ·MAX` the exponential the law is there.

use super::SimdTruncatedNormal;
use super::TruncatedNormalSource;
use crate::traits::SimdFloatExt;
use crate::truncated::quadrature::ldexp;
use crate::truncated::quadrature::split;

/// Which of Robert (1995)'s proposals a one-sided standardised interval $[a, b]$ with $a \ge 0$ takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum TailProposal {
  /// The translated exponential of rate $\alpha^\*$ (Robert 1995, §2.1, Proposition 2.3).
  Exponential { alpha_star: f64 },
  /// The uniform on $[a, b]$ reweighted by $e^{(a^2 - z^2)/2}$ (Robert 1995, §2.2).
  Uniform,
}

/// The standardised one-sided interval $[a, a + w]$, $a \ge 0$, of a tail draw, negated back when `mirrored`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TailSetup {
  pub(super) a: f64,
  width: f64,
  pub(super) mirrored: bool,
  pub(super) proposal: TailProposal,
}

impl TailSetup {
  /// `w`, the width `(up − lo)/σ` where the law holds it (NaN else, `b − a`): bounds rounding to one double past `MAX/2`
  /// still leave the law its width.
  pub(super) fn new(a: f64, b: f64, w: f64, mirrored: bool) -> Self {
    let width = if w.is_nan() { b - a } else { w };
    let root = (a * a + 4.0).sqrt();
    // Past 1.34e154 `a²` overflows, where `√(a² + 4)` rounds to `a`.
    let root = if root.is_finite() { root } else { a };
    // Past `MAX/2` the sum overflows where the rate `a + 2/(a + √(a² + 4))` is `a` to its last double.
    let alpha_star = match 0.5 * (a + root) {
      rate if rate.is_finite() => rate,
      _ => a,
    };
    // Robert (1995), (2.1): the exponential proposal accepts more often from this width on; `(a² − a√(a² + 4))/4`
    // is `−a/(a + √(a² + 4))`, which overflows nowhere.
    let exponential_from = 2.0 * std::f64::consts::E.sqrt() / (a + root) * (-a / (a + root)).exp();
    let proposal = if width > exponential_from {
      TailProposal::Exponential { alpha_star }
    } else {
      TailProposal::Uniform
    };
    Self {
      a,
      width,
      mirrored,
      proposal,
    }
  }
}

impl<T: SimdFloatExt> SimdTruncatedNormal<T> {
  /// Rejection on the base normal while the interval holds over 5 % of its mass (1000 tries), else the exact scheme
  /// the interval's position allows.
  #[inline]
  pub(super) fn draw<S: TruncatedNormalSource<T>>(&self, src: &mut S) -> T {
    if self.norm_mass > 0.05 {
      for _ in 0..1000 {
        let x = src.base(&self.base);
        if x >= self.lower && x <= self.upper {
          return x;
        }
      }
    }
    match self.tail {
      Some(setup) => self.tail_sample(setup, src),
      None => T::from_f64_fast(self.inverse_cdf(src.uniform())),
    }
  }

  /// Robert (1995) accept-reject on the standardised one-sided interval, which forms no cdf: deep in a tail
  /// `F(upper) − F(lower)` loses its digits, and past about 8.3σ both round to 1 and inversion returns `+inf`.
  fn tail_sample<S: TruncatedNormalSource<T>>(&self, setup: TailSetup, src: &mut S) -> T {
    if setup.a == f64::INFINITY {
      return T::from_f64_fast(self.beyond_draw(setup, src.uniform()));
    }
    // The offset `y = z − a` from the near bound, which the law's own coordinate takes without `μ + σz`'s cancellation.
    let y = match setup.proposal {
      TailProposal::Exponential { alpha_star } => loop {
        // `1 - u` keeps the generator's `[0, 1)` off the `ln`'s zero.
        let y = -(1.0 - src.uniform()).ln() / alpha_star;
        if y > setup.width {
          continue;
        }
        let d = (setup.a + y) - alpha_star;
        if src.uniform() <= (-0.5 * d * d).exp() {
          break y;
        }
      },
      TailProposal::Uniform => loop {
        let y = src.uniform() * setup.width;
        // `(a² − z²)/2` as `−y(a + y/2)`: `z = a + y` rounds `y` away where `a·w` is still of order 1.
        if src.uniform() <= (-y * (setup.a + 0.5 * y)).exp() {
          break y;
        }
      },
    };
    let sigma = self.base.std_dev().to_f64().unwrap();
    T::from_f64_fast(if setup.mirrored {
      self.upper.to_f64().unwrap() - sigma * y
    } else {
      self.lower.to_f64().unwrap() + sigma * y
    })
  }

  /// $F^{-1}(F(\text{lower}) + u (F(\text{upper}) - F(\text{lower})))$: the sampler's inverse-cdf map.
  pub(super) fn inverse_cdf(&self, u: f64) -> f64 {
    self.unstandardise(crate::special::ndtri(
      self.f_lo + u * (self.f_up - self.f_lo),
    ))
  }

  /// Past `μ ± σ·MAX` the law is the exponential of rate `|near − μ|/σ²` from its near bound, cut at the far one: the
  /// curvature's share `σ²/(near − μ)²` there is under `1/MAX²`.
  fn beyond_draw(&self, setup: TailSetup, u: f64) -> f64 {
    let (mu, sigma) = (
      self.base.mean().to_f64().unwrap(),
      self.base.std_dev().to_f64().unwrap(),
    );
    let (lo, up) = (self.lower.to_f64().unwrap(), self.upper.to_f64().unwrap());
    let (near, far) = if setup.mirrored { (up, lo) } else { (lo, up) };
    // The scale `σ²/|near − μ|` as `m·2^p`, from halves where `near − μ` overflows.
    let (gap, halved) = match (near - mu).abs() {
      gap if gap.is_finite() => (gap, 0),
      _ => ((0.5 * near - 0.5 * mu).abs(), 1),
    };
    let ((ms, es), (mg, eg)) = (split(sigma), split(gap));
    let (m, p) = (ms * ms / mg, 2 * es - eg - halved);
    let reach = ldexp((far - near).abs() / m, -p);
    let e = if reach > 40.0 {
      -(1.0 - u).ln()
    } else {
      -(u * (-reach).exp_m1()).ln_1p()
    };
    let y = ldexp(e * m, p);
    if setup.mirrored { near - y } else { near + y }
  }

  /// The near bound of an interval past `μ ± σ·f64::MAX`, where the law is its local model.
  pub(super) fn beyond(&self) -> Option<f64> {
    let setup = self.tail?;
    let near = if setup.mirrored {
      self.upper
    } else {
      self.lower
    };
    (setup.a == f64::INFINITY).then(|| near.to_f64().unwrap())
  }
}
