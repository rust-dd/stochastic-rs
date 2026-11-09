//! `Law::quantile_gap`: two near quantiles' difference from their exact level gap, the mass by 16-node Gauss–Legendre
//! at exact offsets, in panels each within `e⁴` where the density changes by more across the gap.

use stochastic_rs_distributions::special::two_sum;

use crate::bivariate::integral::rule;
use crate::student_t::Law;

/// At most this many panels: past them the rest of a wide gap lies under `e⁻⁶⁴` of the first panel's density.
const PANELS: usize = 16;

impl Law {
  /// `T⁻¹(u) − T⁻¹(u − g)` (`x + y` at `g = u + v − 1`, `x − y` at `u − v`), `x̂ ≈ T⁻¹(u)`'s own error removed: Newton on
  /// `(T(x̂) − T(x̂ − d))/f(x̂) = (g + r)/f(x̂)`, `r = T(x̂) − u`, from `start`, less `r/f(x̂)`; in panels past `e⁴` in `f`.
  pub(crate) fn quantile_gap(&self, x: f64, u: f64, g: f64, start: f64, in_panels: bool) -> f64 {
    if g == 0.0 {
      return 0.0;
    }
    // Only `f(x)` carries the density's large exponent, which `precise_pdf` forms past double precision.
    let fx = self.inner.precise_pdf(x);
    let r = if x <= 0.0 {
      self.cdf_at(x) - u
    } else {
      (1.0 - u) - self.cdf_at(-x)
    };
    if in_panels {
      return self.wide_gap(x, g + r, start, r / fx);
    }
    self.solve(x, (g + r) / fx, start) - r / fx
  }

  /// `f(s + o)/f(s) = (1 + o(2s + o)/(ν + s²))^{−(ν + 1)/2}` at the exact offset `o`, not at a node rounded to `s`'s ulps.
  fn ratio(&self, s: f64) -> impl Fn(f64) -> f64 {
    let (nu, inv_root) = (self.nu, self.nu.sqrt().hypot(s).recip());
    move |o| {
      let q = (o * inv_root) * ((2.0 * s + o) * inv_root);
      (-0.5 * (nu + 1.0) * q.ln_1p()).exp()
    }
  }

  /// Newton on `(T(s) − T(s − d))/f(s) = target` from `start`.
  fn solve(&self, s: f64, target: f64, start: f64) -> f64 {
    let ratio = self.ratio(s);
    let mut d = start;
    for _ in 0..8 {
      let step = (target - mass(&ratio, d)) / ratio(-d);
      d += step;
      if step.abs() <= 1e-16 * d.abs() {
        break;
      }
    }
    d
  }

  /// `quantile_gap` past `e⁴`: `T(x) − T(ŷ)` to the double `ŷ = x − start` once in panels, each scaled by its own
  /// `precise_pdf` so no node carries the gap's whole exponent, then Newton from `ŷ`.
  fn wide_gap(&self, x: f64, level_gap: f64, start: f64, shift: f64) -> f64 {
    let y = x - start;
    let (hi, lo) = two_sum(x, -y);
    let span = if x * y >= 0.0 {
      self.span(x, y)
    } else {
      self.span(0.0, x) + self.span(0.0, y)
    };
    let fy = self.inner.precise_pdf(y);
    let offset = self.solve(y, (level_gap - hi.signum() * span) / fy, 0.0);
    hi + ((lo + offset) - shift)
  }

  /// `|T(a) − T(b)|`, `a, b` of one sign: panels from the denser end, each within `e⁴` by `ν + s'² = (ν + s²)e^{8/(ν + 1)}`,
  /// each width's rounding added at its far end's density.
  fn span(&self, a: f64, b: f64) -> f64 {
    let (mut s, end) = if a.abs() <= b.abs() { (a, b) } else { (b, a) };
    let grow = (8.0 / (self.nu + 1.0)).exp_m1();
    let mut fs = self.inner.precise_pdf(s);
    let mut total = 0.0;
    for _ in 0..PANELS {
      let reach = s.mul_add(s, (self.nu + s * s) * grow).sqrt();
      let next = if reach < end.abs() {
        reach.copysign(end)
      } else {
        end
      };
      let (w, lo) = two_sum(next, -s);
      let f_next = self.inner.precise_pdf(next);
      total += fs * mass(&self.ratio(s), -w).abs() + w.signum() * lo * f_next;
      if next == end {
        break;
      }
      (s, fs) = (next, f_next);
    }
    total
  }
}

/// `(T(s) − T(s − d))/f(s)` by 16-node Gauss–Legendre of `f/f(s)`.
fn mass(ratio: &impl Fn(f64) -> f64, d: f64) -> f64 {
  let half = 0.5 * d;
  half
    * rule()
      .iter()
      .map(|&(t, w)| w * ratio(half * (t - 1.0)))
      .sum::<f64>()
}
