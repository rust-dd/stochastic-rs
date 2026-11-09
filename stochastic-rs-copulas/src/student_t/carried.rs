//! `A` in two parts where `H`'s tail multiplies its rounding by `min(A², ν + 1)`: `c₀(sinh t_y − ρ sinh t)/cosh t` from
//! Taylor series about 0 whose leading terms are exact, and `ln P(T ≤ a)` at the double `a` with the low part's first-order term.

use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::special::ln_norm_cdf;
use stochastic_rs_distributions::special::two_prod;
use stochastic_rs_distributions::special::two_sum;

use crate::student_t::Conditional;
use crate::student_t::Law;
use crate::student_t::NORMAL_ABOVE;

/// To this `|t|` and `|t_y|` the series hold `1e-28` of their sums.
pub(crate) const CARRIED_TO: f64 = 1.0;

const TAILS: [(f64, f64); 14] = {
  let mut c = [(0.0, 0.0); 14];
  let (mut odd, mut even, mut k) = (1.0 / 6.0, 0.5, 0);
  while k < 14 {
    let n = (2 * k + 4) as f64;
    even /= (n - 1.0) * n;
    odd /= n * (n + 1.0);
    c[k] = (odd, even);
    k += 1;
  }
  c
};

/// `Σ_{k≥2} x^{2k}/(2k + 1)!` and `Σ_{k≥2} x^{2k}/(2k)!` from `x²`: the series past `x³/3!` and `x²/2!`.
fn tails(x2: f64) -> (f64, f64) {
  let (s, c) = TAILS
    .iter()
    .rev()
    .fold((0.0_f64, 0.0_f64), |(s, c), &(odd, even)| {
      (s.mul_add(x2, odd), c.mul_add(x2, even))
    });
  (s * x2 * x2, c * x2 * x2)
}

/// `sinh x` in two parts at `x ≥ 0` in two parts, `x ≤ 2·CARRIED_TO`: `x + x³/6` to `ε²`, the rest from `x⁵/120` on.
fn sinh_parts((x, x_lo): (f64, f64)) -> (f64, f64) {
  let (x2, x2_lo) = two_prod(x, x);
  let (x3, x3_err) = two_prod(x2, x);
  let sixth = x3 / 6.0;
  let sixth_lo = ((-sixth).mul_add(6.0, x3) + x3_err + x2_lo * x) / 6.0;
  let (rest, rest_c) = tails(x2);
  let (s, s_err) = two_sum(x, sixth);
  let (s, t_err) = two_sum(s, x * rest);
  (
    s,
    s_err + t_err + sixth_lo + x_lo * (1.0 + 0.5 * x2 + rest_c),
  )
}

/// `cosh x` in two parts at `x` in two parts, `|x| ≤ CARRIED_TO`: `1 + x²/2` to `ε²`, the rest from `x⁴/24` on.
fn cosh_parts((x, x_lo): (f64, f64)) -> (f64, f64) {
  let (x2, x2_lo) = two_prod(x, x);
  let (c, c_err) = two_sum(1.0, 0.5 * x2);
  let (rest_s, rest) = tails(x2);
  let (c, t_err) = two_sum(c, rest);
  (
    c,
    c_err + t_err + 0.5 * x2_lo + x * (1.0 + x2 / 6.0 + rest_s) * x_lo,
  )
}

/// `(a + a_lo)(b + b_lo)` to `ε²` of the product.
fn product((a, a_lo): (f64, f64), (b, b_lo): (f64, f64)) -> (f64, f64) {
  let (p, e) = two_prod(a, b);
  (p, e + (a * b_lo + a_lo * b))
}

impl Conditional {
  /// Whether `|t_y|` is within the series' reach.
  pub(crate) fn carried(&self) -> bool {
    self.abs_ty <= CARRIED_TO
  }

  /// `A` in two parts at `|t|` and the gap `g = |t_y| − |t|` in two parts, `|t|, |t_y| ≤ CARRIED_TO`: `c₀` from its square in
  /// two parts, and at the crossing `sinh|t_y| − |ρ| sinh|t| = 2cosh(|t| + g/2)·sinh(g/2) + (1 − |ρ|)sinh|t|` from the gap.
  pub(crate) fn argument_parts(
    &self,
    nu: f64,
    sign_t: f64,
    abs_t: (f64, f64),
    gap: (f64, f64),
  ) -> (f64, f64) {
    let (num, num_lo) = two_sum(nu, 1.0);
    let (r2, r2_lo) = two_prod(self.rho, self.rho);
    let (den, den_err) = two_sum(1.0, -r2);
    let den_lo = den_err - r2_lo;
    let q = num / den;
    let q_lo = ((-q).mul_add(den, num) + num_lo - q * den_lo) / den;
    let c = q.sqrt();
    let c0 = (c, ((-c).mul_add(c, q) + q_lo) / (2.0 * c));
    let sign_r = if sign_t == 0.0 {
      0.0
    } else {
      self.rho.signum() * sign_t
    };
    let (st, st_lo) = sinh_parts(abs_t);
    let rho = self.rho.abs();
    let (d, d_lo) = if self.sign_y != 0.0 && self.sign_y == sign_r {
      let half = (0.5 * gap.0, 0.5 * gap.1);
      let (sh, sh_lo) = sinh_parts((half.0.abs(), half.1 * half.0.signum()));
      let (m, m_err) = two_sum(abs_t.0, half.0);
      let (p, p_lo) = product(
        cosh_parts((m, m_err + abs_t.1 + half.1)),
        (sh * half.0.signum(), sh_lo * half.0.signum()),
      );
      let (w, w_err) = two_sum(1.0, -rho);
      let (r, r_lo) = product((w, w_err), (st, st_lo));
      let (sum, sum_err) = two_sum(2.0 * p, r);
      (
        self.sign_y * sum,
        self.sign_y * (sum_err + 2.0 * p_lo + r_lo),
      )
    } else {
      let (ty, ty_err) = two_sum(abs_t.0, gap.0);
      let (sy, sy_lo) = sinh_parts((ty.max(0.0), ty_err + abs_t.1 + gap.1));
      let (r, r_lo) = product((rho, 0.0), (st, st_lo));
      let (sum, sum_err) = two_sum(self.sign_y * sy, -sign_r * r);
      (sum, sum_err + self.sign_y * sy_lo - sign_r * r_lo)
    };
    let (p, p_lo) = product(c0, (d, d_lo));
    let (ch, ch_lo) = cosh_parts(abs_t);
    let a = p / ch;
    two_sum(a, ((-a).mul_add(ch, p) + p_lo - a * ch_lo) / ch)
  }
}

impl Law {
  /// `ln P(T ≤ a)` at a double `a < 0`, whose magnitude does not round through its logarithm, and `f(a)/P(T ≤ a)`.
  pub(crate) fn ln_lower_and_slope(&self, a: f64) -> (f64, f64) {
    let ln_lower = if self.nu >= NORMAL_ABOVE {
      ln_norm_cdf(a)
    } else if -a >= self.tail_start {
      self.ln_scale - self.nu * (-a).ln()
    } else {
      self.inner.cdf(a).expect("the Student-t law has a cdf").ln()
    };
    (ln_lower, (self.ln_pdf(a) - ln_lower).exp())
  }
}

#[cfg(test)]
mod tests;
