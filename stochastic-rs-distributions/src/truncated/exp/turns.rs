//! `t·x` modulo 2π for doubles `t` and `x`, exact however far the product lies past the doubles: the integer
//! mantissas multiply exactly, and only the bits of 1/(2π) that reach the fraction enter (Payne and Hanek's reduction).
//!
//! Reference: Payne, M. H. & Hanek, R. N. (1983), "Radian reduction for trigonometric functions", *ACM SIGNUM Newsletter* 18(1), 19-24, DOI 10.1145/1057600.1057602.
//! Reference: Cody, W. J. & Waite, W. (1980), *Software Manual for the Elementary Functions*, Prentice-Hall, ch. 8, ISBN 0-13-822064-6 (the reduction of a moderate argument by a split constant).

use std::f64::consts::TAU;

/// Bits 1 to 2304 of 1/(2π) after the binary point, 64 a word, most significant first (mpmath, 800 and 1000 digits):
/// a product of two doubles is below 2²⁰⁴⁸, and its fraction reads 256 bits past that.
const INV_TAU: [u64; 36] = [
  0x28BE60DB9391054A,
  0x7F09D5F47D4D3770,
  0x36D8A5664F10E410,
  0x7F9458EAF7AEF158,
  0x6DC91B8E909374B8,
  0x01924BBA82746487,
  0x3F877AC72C4A69CF,
  0xBA208D7D4BAED121,
  0x3A671C09AD17DF90,
  0x4E64758E60D4CE7D,
  0x272117E2EF7E4A0E,
  0xC7FE25FFF7816603,
  0xFBCBC462D6829B47,
  0xDB4D9FB3C9F2C26D,
  0xD3D18FD9A797FA8B,
  0x5D49EEB1FAF97C5E,
  0xCF41CE7DE294A4BA,
  0x9AFED7EC47E35742,
  0x1580CC11BF1EDAEA,
  0xFC33EF0826BD0D87,
  0x6A78E45857B986C2,
  0x19666157C5281A10,
  0x237FF620135CC9CC,
  0x41818555B29CEA32,
  0x58389EF0231AD1F1,
  0x0670D9F3773A024A,
  0xA0D6711DA2E58729,
  0xB76BD13455C6414F,
  0xA97FC1C14FDF8CFA,
  0x0CB0B793E60C9F6E,
  0xF0CF49BBDAC797BE,
  0x27CE87CD72BC9FC7,
  0x61FC48641F1F091A,
  0xBE9BB55DCB4C10CE,
  0xC571852D674670F0,
  0xB12B50534B174003,
];

/// Bits `j..j + 63` of 1/(2π), bit 1 the first after the binary point, and 0 before it.
fn window(j: i64) -> u64 {
  let k = j - 1;
  if k <= -64 {
    return 0;
  }
  if k < 0 {
    return INV_TAU[0] >> -k;
  }
  let (w, s) = ((k / 64) as usize, (k % 64) as u32);
  if s == 0 {
    INV_TAU[w]
  } else {
    (INV_TAU[w] << s) | (INV_TAU[w + 1] >> (64 - s))
  }
}

/// `|x| = m·2^e`, `m` an integer below 2⁵³.
fn split(x: f64) -> (u64, i64) {
  let bits = x.to_bits();
  let (exp, frac) = (((bits >> 52) & 0x7ff) as i64, bits & ((1 << 52) - 1));
  if exp == 0 {
    (frac, -1074)
  } else {
    (frac | (1 << 52), exp - 1075)
  }
}

/// `t·x/(2π)` modulo 1 in units of 2⁻¹²⁸, to within 2⁻¹²⁶, for finite `t` and `x`.
pub(super) fn turns(t: f64, x: f64) -> u128 {
  let ((mt, et), (mx, ex)) = (split(t), split(x));
  let n = u128::from(mt) * u128::from(mx);
  let e = et + ex;
  let (n1, n0) = ((n >> 64) as u64, n as u64);
  // `F = frac(2^e/(2π))` to 256 bits, so `n·F` modulo 1 keeps 128 bits past `n`'s 106.
  let f = [0, 1, 2, 3].map(|i| window(e + 1 + 64 * i));
  let p = |a: u64, b: u64| u128::from(a) * u128::from(b);
  let high = p(n1, f[1]).wrapping_add(p(n0, f[0])) as u64;
  let mid = p(n1, f[2]).wrapping_add(p(n0, f[1]));
  let (low, carry) = p(n1, f[3]).overflowing_add(p(n0, f[2]));
  let r = (u128::from(high) << 64)
    .wrapping_add(mid)
    .wrapping_add(low >> 64)
    .wrapping_add(u128::from(carry) << 64);
  if (t < 0.0) != (x < 0.0) {
    r.wrapping_neg()
  } else {
    r
  }
}

/// 2π in four parts (fdlibm's π/2 split, times 4), the first three with 31, 32 and 28 bits, so `k·Cᵢ` is exact for
/// `|k| < 2²¹`.
const TAU_PARTS: [f64; 4] = [
  6.2831853069365025,
  2.4308402025215864e-10,
  8.089064994844666e-21,
  3.39137106414756e-31,
];

/// `hi + lo` reduced to about `[−π, π]` by Cody and Waite's subtraction of `k` parts of 2π, for `|hi| < 2²⁰`.
pub(super) fn reduce(hi: f64, lo: f64) -> f64 {
  let k = (hi / TAU).round();
  let [c1, c2, c3, c3t] = TAU_PARTS;
  ((((hi - k * c1) - k * c2) + lo) - k * c3) - k * c3t
}

/// A fraction of a turn as an angle in `[−π, π)`.
pub(super) fn radians(r: u128) -> f64 {
  (r as i128) as f64 * (TAU / 2f64.powi(128))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Products past the doubles, to the table's deep end at `f64::MAX²`, against mpmath at 2600 and 2800 digits.
  #[test]
  fn turns_reduce_products_past_the_doubles() {
    for (t, x, want) in [
      (f64::MAX, f64::MAX, -0.5777223622114582),
      (1e300, 3e299, -2.0337336173754994),
      (-1e200, 7.5e150, 1.2946430995224043),
      (1.0, 5.0, -1.2831853071795865),
      (1e10, f64::MAX, 1.7123598064944354),
    ] {
      let got = radians(turns(t, x));
      assert!(
        (got - want).abs() <= 4.0 * f64::EPSILON,
        "{t:e}·{x:e}: {got} against {want}"
      );
    }
  }

  /// A moderate product reduced by parts agrees with its turns (mpmath: −0.77807382151921254371).
  #[test]
  fn parts_reduce_a_moderate_product() {
    let got = reduce(-545054.5371010363, 0.0);
    assert!(
      (got - -0.7780738215192125).abs() <= 4.0 * f64::EPSILON,
      "{got}"
    );
    assert!((got - radians(turns(-1.0, 545054.5371010363))).abs() <= 4.0 * f64::EPSILON);
  }
}
