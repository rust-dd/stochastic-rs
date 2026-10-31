//! 200 random laws, every parameter log-uniform over the admissible range: every analytic and 10 draws answer, inside
//! its range, and NaN only where documented, so a hang fails CI by its timeout.

use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

/// SplitMix64: the laws are the same on every run.
struct Gen(u64);

impl Gen {
  fn unit(&mut self) -> f64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 * 2f64.powi(-53)
  }

  fn exp10(&mut self, lo: f64, hi: f64) -> f64 {
    10f64.powf(lo + (hi - lo) * self.unit())
  }
}

const LEVELS: [f64; 8] = [
  5e-324,
  1e-300,
  1e-10,
  0.1,
  0.5,
  0.9,
  1.0 - 1e-10,
  1.0 - f64::EPSILON / 2.0,
];

/// A law, its support, the mgf/cf arguments, and which documented NaNs it may answer: the mode of a U-shaped beta on a
/// tie, and the spread and shape of a law narrower than any offset resolves (variance 0).
fn sweep(
  name: &str,
  law: &(impl DistributionExt + Distribution<f64>),
  (lo, up): (f64, f64),
  ts: &[f64],
  tie: bool,
) {
  let inside = |what: &str, v: f64| {
    assert!(!v.is_nan() && v >= lo && v <= up, "{name}: {what} = {v:e}");
  };
  let var = law.variance().unwrap();
  assert!(var >= 0.0, "{name}: variance = {var:e}");
  for (what, v) in [
    ("skewness", law.skewness()),
    ("kurtosis", law.kurtosis()),
    ("entropy", law.entropy()),
  ] {
    assert!(!v.unwrap().is_nan() || var == 0.0, "{name}: {what} NaN");
  }
  inside("mean", law.mean().unwrap());
  inside("median", law.median().unwrap());
  let mode = law.mode().unwrap();
  if !(tie && mode.is_nan()) {
    inside("mode", mode);
  }
  // An open interval's midpoint is NaN, and dropped below.
  let mut points = vec![lo, up, lo + 0.5 * (up - lo)];
  for p in LEVELS {
    let q = law.quantile(p).unwrap();
    inside(&format!("quantile({p:e})"), q);
    points.push(q);
  }
  for x in points.into_iter().filter(|x| !x.is_nan()) {
    let (c, d) = (law.cdf(x).unwrap(), law.pdf(x).unwrap());
    assert!((0.0..=1.0).contains(&c), "{name}: cdf({x:e}) = {c:e}");
    assert!(d >= 0.0, "{name}: pdf({x:e}) = {d:e}");
  }
  for &t in ts {
    let m = law.moment_generating_function(t).unwrap();
    assert!(m >= 0.0, "{name}: mgf({t:e}) = {m:e}");
  }
  let mut rng = SimdRng::from_seed(11);
  for i in 0..10 {
    inside(&format!("draw {i}"), law.sample(&mut rng));
  }
}

#[test]
fn random_laws_answer_inside_their_ranges() {
  let mut g = Gen(0x5eed_2026);
  let inf = f64::INFINITY;
  let mut made = 0;
  while made < 200 {
    let sign = if g.unit() < 0.5 { 1.0 } else { -1.0 };
    match made % 4 {
      0 => {
        let (mu, sigma) = (sign * g.exp10(-300.0, 300.0), g.exp10(-300.0, 300.0));
        let z = |g: &mut Gen| {
          let s = if g.unit() < 0.5 { 1.0 } else { -1.0 };
          if g.unit() < 0.5 {
            s * g.exp10(-300.0, 300.0)
          } else {
            10.0 * s * g.unit()
          }
        };
        let (a, b) = (mu + sigma * z(&mut g), mu + sigma * z(&mut g));
        let lo = if g.unit() < 0.2 { -inf } else { a.min(b) };
        let up = if g.unit() < 0.2 { inf } else { a.max(b) };
        if lo >= up || lo == inf || up == -inf {
          continue;
        }
        let law = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up);
        let ts = [1.0, -1.0, 1.0 / sigma, -2.0 / sigma, 1e10];
        sweep(
          &format!("TN({mu:e}, {sigma:e}, [{lo:e}, {up:e}])"),
          &law,
          (lo, up),
          &ts,
          false,
        );
      }
      1 => {
        let (k, theta) = (g.exp10(-300.0, 300.0), g.exp10(-300.0, 300.0));
        let lo = if g.unit() < 0.3 {
          0.0
        } else {
          theta * g.exp10(-300.0, 300.0)
        };
        let up = if g.unit() < 0.25 {
          inf
        } else {
          lo + theta * g.exp10(-300.0, 300.0)
        };
        if lo >= up || !lo.is_finite() {
          continue;
        }
        let law = SimdTruncatedGamma::<f64>::new(k, theta, lo, up);
        sweep(
          &format!("TG({k:e}, {theta:e}, [{lo:e}, {up:e}])"),
          &law,
          (lo, up),
          &[],
          false,
        );
      }
      2 => {
        let (a, b) = (g.exp10(-300.0, 300.0), g.exp10(-300.0, 300.0));
        let lo = if g.unit() < 0.3 {
          0.0
        } else {
          g.exp10(-300.0, 0.0)
        };
        let up = match (3.0 * g.unit()) as u32 {
          0 => 1.0,
          1 => 1.0 - g.exp10(-16.5, 0.0),
          _ => lo + (1.0 - lo) * g.exp10(-300.0, 0.0),
        };
        if lo >= up || up > 1.0 {
          continue;
        }
        let law = SimdTruncatedBeta::<f64>::new(a, b, lo, up);
        let tie = (a < 1.0 && b < 1.0) || (a == 1.0 && b == 1.0);
        sweep(
          &format!("TB({a:e}, {b:e}, [{lo:e}, {up:e}])"),
          &law,
          (lo, up),
          &[],
          tie,
        );
      }
      _ => {
        let lam = g.exp10(-300.0, 300.0);
        let lo = if g.unit() < 0.3 {
          0.0
        } else {
          g.exp10(-300.0, 300.0)
        };
        let up = if g.unit() < 0.25 {
          inf
        } else {
          lo + g.exp10(-300.0, 300.0)
        };
        if lo >= up || !lo.is_finite() {
          continue;
        }
        let law = SimdTruncatedExp::<f64>::new(lam, lo, up);
        let name = format!("TE({lam:e}, [{lo:e}, {up:e}])");
        let ts = [1.0, -1.0, 0.5 * lam, -2.0 * lam];
        sweep(&name, &law, (lo, up), &ts, false);
        for t in ts {
          let cf = law.characteristic_function(t).unwrap();
          // The phase `e^{it·lower}` keeps no digit where `t·lower` overflows.
          assert!(
            !cf.re.is_nan() || !(t * lo).is_finite(),
            "{name}: cf({t:e}) NaN"
          );
          assert!(
            cf.norm().is_nan() || cf.norm() <= 1.0 + 1e-12,
            "{name}: |cf({t:e})| = {}",
            cf.norm()
          );
        }
      }
    }
    made += 1;
  }
}
