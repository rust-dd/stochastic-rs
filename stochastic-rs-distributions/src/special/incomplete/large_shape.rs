//! The gamma family's tails from shape 100 on: Temme's uniform expansion for `0.3 ≤ x/a ≤ 2.35`, the series or fraction
//! scaled by Stirling's `Γ*` outside, and on both sides the exponent `a(λ − 1 − ln λ)` carried past double precision.
//! Reference: Temme, N.M. (1979), "The asymptotic expansion of the incomplete gamma functions", *SIAM Journal on Mathematical Analysis* 10(4), 757-766, DOI 10.1137/0510071 (DLMF 8.12.3–8.12.4, 8.12.7, 8.12.11–8.12.14).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 5.11.1 (Stirling's series for `ln Γ`), <https://dlmf.nist.gov/5.11.1>.
//! Reference: Shewchuk, J.R. (1997), "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", *Discrete & Computational Geometry* 18(3), 305-363, Theorem 7 (Knuth's two-sum).

use std::f64::consts::TAU;

use super::Tail;
use super::continued_fraction;
use super::series;
use crate::special::mills_ratio;

/// `x/a` where Temme's expansion is summed, `|η| ≤ 1.004`; outside it the series or the fraction converge within 30 terms.
const TEMME_BAND: (f64, f64) = (0.3, 2.35);

/// `d_{k,n}` of DLMF 8.12.12, exact rationals rounded, `c_k(η) = Σ_n d_{k,n} ηⁿ`, each cut where `|η| ≤ 1.004` and `a ≥ 100`
/// leave less than `2e-18`.
const TEMME: [&[f64]; 8] = [
  &[
    -0.3333333333333333,
    0.08333333333333333,
    -0.014814814814814815,
    0.0011574074074074073,
    0.0003527336860670194,
    -0.0001787551440329218,
    3.919263178522438e-5,
    -2.185448510679992e-6,
    -1.85406221071516e-6,
    8.296711340953087e-7,
    -1.7665952736826078e-7,
    6.707853543401498e-9,
    1.0261809784240309e-8,
    -4.382036018453353e-9,
    9.14769958223679e-10,
    -2.5514193994946248e-11,
    -5.830772132550426e-11,
    2.4361948020667415e-11,
    -5.0276692801141755e-12,
    1.1004392031956135e-13,
    3.371763262400985e-13,
    -1.392388722418162e-13,
    2.8534893807047445e-14,
    -5.139111834242572e-16,
    -1.9752288294349442e-15,
    8.099521156704561e-16,
    -1.6522531216398162e-16,
    2.5305430097478883e-18,
    1.1686939738559576e-17,
    -4.770037049820485e-18,
  ],
  &[
    -0.001851851851851852,
    -0.003472222222222222,
    0.0026455026455026454,
    -0.0009902263374485596,
    0.00020576131687242798,
    -4.018775720164609e-7,
    -1.8098550334489977e-5,
    7.64916091608111e-6,
    -1.6120900894563446e-6,
    4.647127802807434e-9,
    1.378633446915721e-7,
    -5.752545603517705e-8,
    1.1951628599778148e-8,
    -1.7543241719747647e-11,
    -1.0091543710600413e-9,
    4.162792991842583e-10,
    -8.56390702649298e-11,
    6.067215101604758e-14,
    7.1624989648114856e-12,
    -2.933186643771437e-12,
    5.996696365683689e-13,
    -2.1671786527323313e-16,
    -4.978339972369262e-14,
    2.0291628823713425e-14,
    -4.13125571381061e-15,
    8.286516239883097e-19,
    3.4100308869333327e-16,
  ],
  &[
    0.004133597883597883,
    -0.0026813271604938273,
    0.0007716049382716049,
    2.0093878600823047e-6,
    -0.0001073665322636516,
    5.2923448829120125e-5,
    -1.2760635188618728e-5,
    3.423578734096138e-8,
    1.3721957309062934e-6,
    -6.298992138380055e-7,
    1.4280614206064242e-7,
    -2.0477098421990866e-10,
    -1.409252991086752e-8,
    6.228974084922022e-9,
    -1.3670488396617114e-9,
    9.428356159014678e-13,
    1.2872252400089318e-10,
    -5.5645956134363323e-11,
    1.197593554636698e-11,
    -4.1689782251838634e-15,
    -1.0940640427884595e-12,
    4.662239946390136e-13,
    -9.905105763906907e-14,
  ],
  &[
    0.0006494341563786008,
    0.00022947209362139917,
    -0.0004691894943952557,
    0.00026772063206283885,
    -7.561801671883977e-5,
    -2.396505113867297e-7,
    1.1082654115347302e-5,
    -5.6749528269915965e-6,
    1.4230900732435883e-6,
    -2.7861080291528143e-11,
    -1.6958404091930278e-7,
    8.099464905388083e-8,
    -1.9111168485973655e-8,
    2.3928620439808118e-12,
    2.0620131815488797e-9,
    -9.460496661855133e-10,
    2.1541049775774907e-10,
    -1.388823336813903e-14,
    -2.1894761681963938e-11,
    9.790998951171684e-12,
    -2.178219188018096e-12,
  ],
  &[
    -0.0008618882909167117,
    0.0007840392217200666,
    -0.0002990724803031902,
    -1.4638452578843418e-6,
    6.641498215465122e-5,
    -3.968365047179435e-5,
    1.1375726970678419e-5,
    2.507497226237533e-10,
    -1.6954149536558305e-6,
    8.907507532205309e-7,
    -2.292934834000805e-7,
    2.956794137544049e-11,
    2.8865829742708783e-8,
    -1.4189739437803219e-8,
    3.4463580499464896e-9,
    -2.3024517174528067e-13,
    -3.9409233028046403e-10,
    1.86023389685045e-10,
  ],
  &[
    -0.00033679855336635813,
    -6.972813758365857e-5,
    0.0002772753244959392,
    -0.00019932570516188847,
    6.797780477937208e-5,
    1.419062920643967e-7,
    -1.3594048189768693e-5,
    8.018470256334202e-6,
    -2.291481176508095e-6,
    -3.252473551298454e-10,
    3.4652846491085265e-7,
    -1.8447187191171344e-7,
    4.8240967037894184e-8,
  ],
  &[
    0.0005313079364639922,
    -0.0005921664373536939,
    0.0002708782096718045,
    7.902353232660328e-7,
    -8.153969367561969e-5,
    5.61168275310625e-5,
    -1.8329116582843375e-5,
    -3.0796134506033047e-9,
    3.465155368803609e-6,
    -2.0291327396058603e-6,
  ],
  &[
    0.00034436760689237765,
    5.171790908260592e-5,
    -0.00033493161081142234,
    0.0002812695154763237,
  ],
];

/// Temme's `Q = ½ erfc(η√(a/2)) + S` (DLMF 8.12.4) in the band, as `e^{−aη²/2}/√(2π)` times Mills' ratio at `|η|√a` plus
/// `S`'s `c_k` sum over `√a`, `P` alike below `x = a`; outside, `√(a/2π) e^{−a(λ−1−ln λ)}/Γ*(a)` times the series or fraction.
#[inline(never)]
pub(super) fn tail(a: f64, x: f64) -> Tail {
  let lambda = x / a;
  if (TEMME_BAND.0..=TEMME_BAND.1).contains(&lambda) {
    let (y, y_lo) = settle(exponent(a, x));
    let upper = x >= a;
    let eta = (2.0 * (y / a)).sqrt().copysign(x - a);
    let sum = TEMME.iter().rev().fold(0.0, |total, d| {
      total / a + d.iter().rev().fold(0.0, |acc, &c| acc * eta + c)
    });
    let mills = mills_ratio((2.0 * y).sqrt());
    let bracket = if upper {
      mills + sum / a.sqrt()
    } else {
      mills - sum / a.sqrt()
    };
    return Tail {
      upper,
      factor: bracket / TAU.sqrt() * (-y_lo).exp(),
      ln_scale: -y,
    };
  }
  let (l, l_lo) = ln_ratio(x, a);
  let (d, d_lo) = two_sum(x, -a);
  let p = a * l;
  let (ln_scale, ln_scale_lo) = if p.is_finite() {
    let (s, s_err) = two_sum(p, -d);
    settle((s, s_err + (a.mul_add(l, -p) + a * l_lo) - d_lo))
  } else {
    (p, 0.0)
  };
  let scale = (a / TAU).sqrt() * (ln_scale_lo - ln_gamma_star(a)).exp();
  if x < a {
    Tail {
      upper: false,
      factor: scale * series(a, x),
      ln_scale,
    }
  } else {
    Tail {
      upper: true,
      factor: scale * continued_fraction(a, x),
      ln_scale,
    }
  }
}

/// `a(λ − 1 − ln λ) = D·T − 2aT³ Σ_j T^{2j}/(2j + 3)`, `D = x − a`, `T = D/(x + a)`, as `y + y_lo`: two-sums and fused
/// multiply-adds recover the roundings of `D`, `T`, `D·T`, `T³`, the sum's first term and the cubic term's product.
fn exponent(a: f64, x: f64) -> (f64, f64) {
  const THIRD_LO: f64 = 1.0 / 3.0 / 18_014_398_509_481_984.0;
  let (d, d_lo) = two_sum(x, -a);
  let (m, m_lo) = two_sum(0.5 * x, 0.5 * a);
  let t = 0.5 * d / m;
  let t_lo = ((-t).mul_add(m, 0.5 * d) + 0.5 * d_lo - t * m_lo) / m;
  let p = d * t;
  let p_lo = d.mul_add(t, -p) + d * t_lo + d_lo * t;
  let t2 = t * t;
  let t2_lo = t.mul_add(t, -t2) + 2.0 * t * t_lo;
  let t3 = t2 * t;
  let t3_lo = t2.mul_add(t, -t3) + t2 * t_lo + t2_lo * t;
  let mut term = t2;
  let mut rest = 0.0;
  for j in 2..64 {
    term *= t2;
    let add = term / f64::from(2 * j + 3);
    rest += add;
    if add <= 1e-19 {
      break;
    }
  }
  let fifth = t2 / 5.0;
  let fifth_lo = ((-fifth).mul_add(5.0, t2) + t2_lo) / 5.0;
  let (v, v_err) = two_sum(fifth, rest);
  let (u, u_err) = two_sum(1.0 / 3.0, v);
  let u_lo = u_err + v_err + fifth_lo + THIRD_LO;
  let q = t3 * u;
  let q_lo = t3.mul_add(u, -q) + t3 * u_lo + t3_lo * u;
  let e = a * (2.0 * q);
  let e_lo = a.mul_add(2.0 * q, -e) + a * (2.0 * q_lo);
  let (y, y_err) = two_sum(p, -e);
  (y, y_err + p_lo - e_lo)
}

/// `ln(x/a)` as `(hi, lo)`: the quotient's remainder from a fused multiply-add, then `x/a = 2^k m` with
/// `m ∈ [1/√2, √2)`, `k ln 2` in two parts and `ln m = 2 atanh((m − 1)/(m + 1))` with its leading term compensated.
fn ln_ratio(x: f64, a: f64) -> (f64, f64) {
  const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
  const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
  let lambda = x / a;
  if lambda < f64::MIN_POSITIVE {
    return (x.ln() - a.ln(), 0.0);
  }
  let lambda_lo = (-lambda).mul_add(a, x) / a;
  let bits = lambda.to_bits();
  let mut k = ((bits >> 52) & 0x7ff) as i32 - 1023;
  let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
  if m > std::f64::consts::SQRT_2 {
    m *= 0.5;
    k += 1;
  }
  let num = m - 1.0;
  let (den, den_lo) = two_sum(m, 1.0);
  let s = num / den;
  let s_lo = ((-s).mul_add(den, num) - s * den_lo) / den;
  let s2 = s * s;
  let mut term = 1.0;
  let mut w = 1.0 / 3.0;
  for j in 1..16 {
    term *= s2;
    let add = term / f64::from(2 * j + 3);
    w += add;
    if add <= 1e-19 * w {
      break;
    }
  }
  let kf = f64::from(k);
  let (h, h_lo) = two_sum(kf * LN2_HI, 2.0 * s);
  two_sum(
    h,
    h_lo + (kf * LN2_LO + 2.0 * s_lo + 2.0 * s * s2 * w + lambda_lo / lambda),
  )
}

/// `(hi, lo)` with a low part past 1 folded into `hi`, where it is within a few of `hi`'s ulps: only a small `lo` is exponentiated.
fn settle((hi, lo): (f64, f64)) -> (f64, f64) {
  if lo.abs() > 1.0 {
    (hi + lo, 0.0)
  } else {
    (hi, lo)
  }
}

/// `s + e = a + b` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
  let s = a + b;
  let z = s - a;
  (s, (a - (s - z)) + (b - z))
}

/// `ln Γ*(a) = ln Γ(a) − (a − ½) ln a + a − ½ ln 2π` by Stirling's series, whose fourth term is below `1e-17` from `a = 100`.
fn ln_gamma_star(a: f64) -> f64 {
  let r = 1.0 / a;
  let r2 = r * r;
  r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 / 1260.0))
}
