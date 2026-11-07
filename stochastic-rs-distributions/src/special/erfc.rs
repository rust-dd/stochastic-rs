//! `erfc` as the `libm` crate transcribes fdlibm's `s_erf.c`, operation for operation, so it inlines into `norm_cdf`'s
//! callers where the crate's out-of-line copy cannot; every value keeps that crate's bits.
//! Reference: Sun Microsystems (1993), fdlibm `s_erf.c` (the rational approximations of `erfc` on `[0, 0.84375]`, `[0.84375, 1.25]`, `[1.25, 1/0.35]` and `[1/0.35, 28]`).
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice is preserved.

const ERX: f64 = 0.8450629115104675;

const PP0: f64 = 0.12837916709551256;

const PP1: f64 = -0.3250421072470015;

const PP2: f64 = -0.02848174957559851;

const PP3: f64 = -0.005770270296489442;

const PP4: f64 = -2.3763016656650163e-5;

const QQ1: f64 = 0.39791722395915535;

const QQ2: f64 = 0.0650222499887673;

const QQ3: f64 = 0.005081306281875766;

const QQ4: f64 = 0.00013249473800432164;

const QQ5: f64 = -3.960228278775368e-6;

const PA0: f64 = -0.0023621185607526594;

const PA1: f64 = 0.41485611868374833;

const PA2: f64 = -0.3722078760357013;

const PA3: f64 = 0.31834661990116175;

const PA4: f64 = -0.11089469428239668;

const PA5: f64 = 0.035478304325618236;

const PA6: f64 = -0.002166375594868791;

const QA1: f64 = 0.10642088040084423;

const QA2: f64 = 0.540397917702171;

const QA3: f64 = 0.07182865441419627;

const QA4: f64 = 0.12617121980876164;

const QA5: f64 = 0.01363708391202905;

const QA6: f64 = 0.011984499846799107;

const RA0: f64 = -0.009864944034847148;

const RA1: f64 = -0.6938585727071818;

const RA2: f64 = -10.558626225323291;

const RA3: f64 = -62.375332450326006;

const RA4: f64 = -162.39666946257347;

const RA5: f64 = -184.60509290671104;

const RA6: f64 = -81.2874355063066;

const RA7: f64 = -9.814329344169145;

const SA1: f64 = 19.651271667439257;

const SA2: f64 = 137.65775414351904;

const SA3: f64 = 434.56587747522923;

const SA4: f64 = 645.3872717332679;

const SA5: f64 = 429.00814002756783;

const SA6: f64 = 108.63500554177944;

const SA7: f64 = 6.570249770319282;

const SA8: f64 = -0.0604244152148581;

const RB0: f64 = -0.0098649429247001;

const RB1: f64 = -0.799283237680523;

const RB2: f64 = -17.757954917754752;

const RB3: f64 = -160.63638485582192;

const RB4: f64 = -637.5664433683896;

const RB5: f64 = -1025.0951316110772;

const RB6: f64 = -483.5191916086514;

const SB1: f64 = 30.33806074348246;

const SB2: f64 = 325.7925129965739;

const SB3: f64 = 1536.729586084437;

const SB4: f64 = 3199.8582195085955;

const SB5: f64 = 2553.0504064331644;

const SB6: f64 = 474.52854120695537;

const SB7: f64 = -22.44095244658582;

#[inline]
pub(super) fn erfc(x: f64) -> f64 {
  let mut ix = (x.to_bits() >> 32) as u32;
  let sign = ix >> 31 != 0;
  ix &= 0x7fff_ffff;
  if ix >= 0x7ff0_0000 {
    return 2.0 * f64::from(u8::from(sign)) + 1.0 / x;
  }
  if ix < 0x3feb_0000 {
    if ix < 0x3c70_0000 {
      return 1.0 - x;
    }
    let z = x * x;
    let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
    let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
    let y = r / s;
    if sign || ix < 0x3fd0_0000 {
      return 1.0 - (x + x * y);
    }
    return 0.5 - (x - 0.5 + x * y);
  }
  if ix < 0x403c_0000 {
    return if sign {
      2.0 - erfc_from_084(ix, x)
    } else {
      erfc_from_084(ix, x)
    };
  }
  let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000);
  if sign {
    2.0 - x1p_1022
  } else {
    x1p_1022 * x1p_1022
  }
}

#[inline]
fn erfc_to_125(x: f64) -> f64 {
  let s = x.abs() - 1.0;
  let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
  let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
  1.0 - ERX - p / q
}

#[inline]
fn erfc_from_084(ix: u32, x: f64) -> f64 {
  if ix < 0x3ff4_0000 {
    return erfc_to_125(x);
  }
  let x = x.abs();
  let s = 1.0 / (x * x);
  let (r, big_s) = if ix < 0x4006_db6d {
    (
      RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
      1.0
        + s
          * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
    )
  } else {
    (
      RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
      1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
    )
  };
  let z = f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000);
  libm::exp(-z * z - 0.5625) * libm::exp((z - x) * (z + x) + r / big_s) / x
}

#[cfg(test)]
mod tests {
  /// Bit for bit with the `libm` crate on a million xorshift cases over every double and over `[−32, 32]`, and at the ends
  /// of each approximation's interval.
  #[test]
  fn erfc_keeps_the_libm_bits() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    for _ in 0..500_000 {
      let bits = next();
      for x in [
        f64::from_bits(bits),
        (bits >> 11) as f64 / (1u64 << 53) as f64 * 64.0 - 32.0,
      ] {
        if !x.is_nan() {
          assert_eq!(
            super::erfc(x).to_bits(),
            libm::erfc(x).to_bits(),
            "x = {x:e}"
          );
        }
      }
    }
    for x in [
      0.0,
      -0.0,
      5e-324,
      2.0_f64.powi(-56),
      0.25,
      0.84375,
      1.25,
      1.0 / 0.35,
      6.0,
      28.0,
      -28.0,
      f64::INFINITY,
      f64::NEG_INFINITY,
    ] {
      for x in [x, -x, x.next_up(), x.next_down()] {
        assert_eq!(
          super::erfc(x).to_bits(),
          libm::erfc(x).to_bits(),
          "x = {x:e}"
        );
      }
    }
    assert!(super::erfc(f64::NAN).is_nan());
  }
}
