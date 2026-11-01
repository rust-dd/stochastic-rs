//! The strata the log-uniform draw misses: unit and ordinary shapes, betas with a bound at 0 or 1, shapes below 1 with a
//! bound between the singular end's rungs, normals far from `μ` or past `μ ± σ·MAX`, every family's extremes, and the
//! normal's mgf where its terms pass the doubles.

use crate::survey::Case;
use crate::survey::Gen;
use crate::survey::Law;
use crate::survey::Stratum;

const MAX: f64 = f64::MAX;

const INF: f64 = f64::INFINITY;

/// Shapes beside 1 at the three distances the plain form's switches sit near.
const BESIDE_ONE: [f64; 6] = [
  1.0 + 1e-12,
  1.0 - 1e-12,
  1.0 + 1e-9,
  1.0 - 1e-9,
  1.0 + 1e-6,
  1.0 - 1e-6,
];

/// The normals whose tail setup overflowed past `MAX/2` (part 2's hang laws).
const HANG_LAWS: [(f64, f64, f64, f64); 7] = [
  (0.0, 1.0, 1e308, MAX),
  (0.0, 1.0, 9e307, INF),
  (0.0, 1e-10, 1e298, INF),
  (-MAX, 1.0, 1.0, 2.0),
  (-MAX / 2.0, 1.0, MAX / 2.0, MAX),
  (-MAX, 1.5, 0.0, INF),
  (MAX, 1.0, -MAX, -1.0),
];

/// `per_stratum` cases of `stratum` from its own stream of `seed`; the extremes begin with the hang laws.
pub fn draw(stratum: Stratum, seed: u64, per_stratum: usize) -> Vec<Case> {
  let mut g = Gen(seed ^ (0xA076_1D64_78BD_642F_u64.wrapping_mul(stratum as u64 + 1)));
  let mut out = Vec::new();
  if stratum == Stratum::Extreme {
    for (mu, sigma, lo, up) in HANG_LAWS.into_iter().take(per_stratum) {
      out.push(case(stratum, Law::N { mu, sigma, lo, up }, Vec::new()));
    }
  }
  let mut tries = 0;
  while out.len() < per_stratum && tries < 100 * per_stratum {
    tries += 1;
    let made = match stratum {
      Stratum::Unit => unit(&mut g).map(|l| (l, Vec::new())),
      Stratum::Ordinary => ordinary(&mut g).map(|l| (l, Vec::new())),
      Stratum::Bound => bound(&mut g).map(|l| (l, Vec::new())),
      Stratum::Rungs => rungs(&mut g).map(|l| (l, Vec::new())),
      Stratum::Far => far(&mut g).map(|l| (l, Vec::new())),
      Stratum::Extreme => extreme(&mut g).map(|l| (l, Vec::new())),
      Stratum::Mgf => mgf(&mut g),
      Stratum::Draw => None,
    };
    if let Some((law, ts)) = made.filter(|(law, _)| law.admissible()) {
      out.push(case(stratum, law, ts));
    }
  }
  out
}

fn case(stratum: Stratum, law: Law, ts: Vec<f64>) -> Case {
  Case { stratum, law, ts }
}

fn unit_shape(g: &mut Gen) -> f64 {
  if g.coin(0.5) {
    1.0
  } else {
    g.pick(&BESIDE_ONE)
  }
}

fn sorted(x: f64, y: f64) -> (f64, f64) {
  (x.min(y), x.max(y))
}

/// The beta's mode where it has one inside, else its mean, and its standard deviation.
fn beta_place(a: f64, b: f64) -> (f64, f64) {
  let m = if a > 1.0 && b > 1.0 {
    (a - 1.0) / (a + b - 2.0)
  } else {
    a / (a + b)
  };
  let s = a + b;
  (m, (a * b / (s * s * (s + 1.0))).sqrt())
}

fn beta_interval(g: &mut Gen) -> (f64, f64) {
  match g.next() % 4 {
    0 => (0.0, 1.0),
    1 => (0.0, g.uniform(0.05, 0.95)),
    2 => (g.uniform(0.05, 0.95), 1.0),
    _ => sorted(g.uniform(0.01, 0.99), g.uniform(0.01, 0.99)),
  }
}

/// The gamma's interval within a few scales of its mode: a bound at 0, ordinary bounds, an open end.
fn gamma_interval(g: &mut Gen, k: f64, theta: f64) -> (f64, f64) {
  let (m, sd) = ((k - 1.0).max(0.0) * theta, k.sqrt() * theta);
  let lo = if g.coin(0.3) {
    0.0
  } else {
    (m + sd * g.uniform(-3.0, 2.0)).max(0.0)
  };
  let up = if g.coin(0.3) {
    INF
  } else {
    lo + sd * g.uniform(0.05, 6.0)
  };
  (lo, up)
}

/// Shapes exactly 1 (α, β, both, k) and 1 ± 1e-12, 1e-9, 1e-6.
fn unit(g: &mut Gen) -> Option<Law> {
  if g.coin(0.6) {
    let ordinary = |g: &mut Gen| g.log_uniform(0.1, 100.0);
    let (a, b) = match g.next() % 3 {
      0 => (unit_shape(g), ordinary(g)),
      1 => (ordinary(g), unit_shape(g)),
      _ => (unit_shape(g), unit_shape(g)),
    };
    let (lo, up) = beta_interval(g);
    Some(Law::B { a, b, lo, up })
  } else {
    let (k, theta) = (unit_shape(g), g.log_uniform(1e-3, 1e3));
    let (lo, up) = gamma_interval(g, k, theta);
    Some(Law::G { k, theta, lo, up })
  }
}

/// Shapes in `[0.1, 100]`, bounds within a few standard deviations of the mode: the parents keep their digits here.
fn ordinary(g: &mut Gen) -> Option<Law> {
  match g.next() % 4 {
    0 => {
      let (mu, sigma) = (g.uniform(-10.0, 10.0), g.log_uniform(0.1, 10.0));
      let lo = if g.coin(0.15) {
        -INF
      } else {
        mu + sigma * g.uniform(-4.0, 3.0)
      };
      let up = if g.coin(0.15) {
        INF
      } else if lo.is_finite() {
        lo + sigma * g.uniform(0.05, 6.0)
      } else {
        mu + sigma * g.uniform(-3.0, 4.0)
      };
      Some(Law::N { mu, sigma, lo, up })
    }
    1 => {
      let (k, theta) = (g.log_uniform(0.1, 100.0), g.log_uniform(1e-3, 1e3));
      let (lo, up) = gamma_interval(g, k, theta);
      Some(Law::G { k, theta, lo, up })
    }
    2 => {
      let (a, b) = (g.log_uniform(0.1, 100.0), g.log_uniform(0.1, 100.0));
      let (m, sd) = beta_place(a, b);
      let lo = (m + sd * g.uniform(-4.0, 1.0)).clamp(0.0, 1.0);
      let up = (lo + sd * g.uniform(0.05, 6.0)).clamp(0.0, 1.0);
      Some(Law::B { a, b, lo, up })
    }
    _ => {
      let lam = g.log_uniform(0.1, 10.0);
      let lo = g.uniform(0.0, 3.0) / lam;
      let up = if g.coin(0.3) {
        INF
      } else {
        lo + g.uniform(0.05, 8.0) / lam
      };
      Some(Law::E { lam, lo, up })
    }
  }
}

/// Betas with a bound at 0 or 1 and the other bound ordinary, shapes ordinary or beside 1.
fn bound(g: &mut Gen) -> Option<Law> {
  let shape = |g: &mut Gen| {
    if g.coin(0.2) {
      unit_shape(g)
    } else {
      g.log_uniform(0.1, 100.0)
    }
  };
  let (a, b) = (shape(g), shape(g));
  let (m, sd) = beta_place(a, b);
  let (lo, up) = if g.coin(0.5) {
    let up = if g.coin(0.5) {
      (m + sd * g.uniform(-2.0, 4.0)).clamp(1e-3, 0.999)
    } else {
      g.uniform(0.02, 0.98)
    };
    (0.0, up)
  } else {
    let lo = if g.coin(0.5) {
      (m + sd * g.uniform(-4.0, 2.0)).clamp(1e-3, 0.999)
    } else {
      g.uniform(0.02, 0.98)
    };
    (lo, 1.0)
  };
  Some(Law::B { a, b, lo, up })
}

/// A shape below 1 whose bound lies between its singular end's first rungs (a tenth and a hundredth of the width).
fn rungs(g: &mut Gen) -> Option<Law> {
  if g.coin(0.5) {
    let (k, theta) = (g.log_uniform(0.02, 0.98), g.log_uniform(1e-3, 1e3));
    let lo = theta * g.log_uniform(1e-3, 5.0);
    let up = if g.coin(0.6) {
      INF
    } else {
      lo + theta * g.log_uniform(1.0, 60.0)
    };
    Some(Law::G { k, theta, lo, up })
  } else {
    let (small, other) = (g.log_uniform(0.02, 0.98), g.log_uniform(0.5, 50.0));
    let reach = if g.coin(0.3) {
      1.0
    } else {
      g.uniform(0.2, 1.0)
    };
    let near = reach * g.log_uniform(1e-4, 0.3);
    if g.coin(0.5) {
      Some(Law::B {
        a: small,
        b: other,
        lo: near,
        up: reach,
      })
    } else {
      Some(Law::B {
        a: other,
        b: small,
        lo: 1.0 - reach,
        up: 1.0 - near,
      })
    }
  }
}

/// `n` ulps above `x`.
fn ulps_above(x: f64, n: u32) -> f64 {
  (0..n).fold(x, |y, _| y.next_up())
}

/// Narrow intervals far from `μ`, open far tails, and laws past `μ ± σ·MAX`.
fn far(g: &mut Gen) -> Option<Law> {
  let (mu, sigma) = (g.uniform(-5.0, 5.0), g.log_uniform(0.1, 10.0));
  match g.next() % 3 {
    0 => {
      let lo = mu + g.sign() * sigma * g.exp10(1.0, 15.0);
      let up = if g.coin(0.5) {
        ulps_above(lo, g.pick(&[1.0, 2.0, 8.0, 256.0, 4096.0]) as u32)
      } else {
        (lo + sigma * g.exp10(-12.0, -3.0)).max(lo.next_up())
      };
      Some(Law::N { mu, sigma, lo, up })
    }
    1 => {
      let d = sigma * g.exp10(1.0, 300.0);
      if g.coin(0.5) {
        Some(Law::N {
          mu,
          sigma,
          lo: mu + d,
          up: INF,
        })
      } else {
        Some(Law::N {
          mu,
          sigma,
          lo: -INF,
          up: mu - d,
        })
      }
    }
    _ => {
      let side = g.sign();
      let mu = -side * MAX * g.uniform(0.5, 1.0);
      let sigma = g.uniform(0.05, 0.45);
      let (lo, up) = match g.next() % 6 {
        0 => (1e-310, 1.0),
        1 => (0.0, 1.0),
        2 => (1e-310, 2e-310),
        3 => (0.0, INF),
        4 => (-1.0, 2.0),
        _ => (1e-300, 1e-299),
      };
      let (lo, up) = if side > 0.0 { (lo, up) } else { (-up, -lo) };
      Some(Law::N { mu, sigma, lo, up })
    }
  }
}

/// A magnitude past 1e300, a subnormal, or an ordinary one.
fn extreme_magnitude(g: &mut Gen) -> f64 {
  match g.next() % 3 {
    0 => g.log_uniform(1e300, MAX),
    1 => f64::from_bits(1 + g.next() % (1u64 << 52)),
    _ => g.log_uniform(1e-3, 1e3),
  }
}

/// Bounds past 1e300, subnormal widths and scales, in all four families.
fn extreme(g: &mut Gen) -> Option<Law> {
  let width = |g: &mut Gen, lo: f64| match g.next() % 3 {
    0 => ulps_above(lo, (1 + g.next() % 16) as u32),
    1 => lo + f64::from_bits(1 + g.next() % (1u64 << 40)),
    _ => lo + g.log_uniform(1e300, MAX).min(MAX - lo.abs()),
  };
  match g.next() % 4 {
    0 => {
      let mu = if g.coin(0.2) {
        0.0
      } else {
        g.sign() * extreme_magnitude(g)
      };
      let sigma = extreme_magnitude(g);
      let lo = match g.next() % 3 {
        0 => g.sign() * g.log_uniform(1e300, MAX),
        1 => mu + g.sign() * sigma * g.exp10(0.0, 2.0),
        _ => g.sign() * f64::from_bits(g.next() % (1u64 << 52)),
      };
      let up = if g.coin(0.2) { INF } else { width(g, lo) };
      Some(Law::N { mu, sigma, lo, up })
    }
    1 => {
      let (k, theta) = (extreme_magnitude(g), extreme_magnitude(g));
      let lo = match g.next() % 3 {
        0 => 0.0,
        1 => g.log_uniform(1e300, MAX),
        _ => f64::from_bits(g.next() % (1u64 << 52)),
      };
      let up = if g.coin(0.2) { INF } else { width(g, lo) };
      Some(Law::G { k, theta, lo, up })
    }
    2 => {
      let (a, b) = (extreme_magnitude(g), extreme_magnitude(g));
      let (lo, up) = match g.next() % 4 {
        0 => (0.0, f64::from_bits(1 + g.next() % (1u64 << 52))),
        1 => (1.0 - f64::EPSILON * g.pick(&[0.5, 1.0, 4.0]), 1.0),
        2 => {
          let lo = g.uniform(0.0, 1.0);
          (lo, ulps_above(lo, (1 + g.next() % 16) as u32).min(1.0))
        }
        _ => (0.0, 1.0),
      };
      Some(Law::B { a, b, lo, up })
    }
    _ => {
      let lam = extreme_magnitude(g);
      let lo = match g.next() % 3 {
        0 => 0.0,
        1 => g.log_uniform(1e300, MAX),
        _ => f64::from_bits(g.next() % (1u64 << 52)),
      };
      let up = match g.next() % 3 {
        0 => INF,
        1 => MAX,
        _ => width(g, lo),
      };
      Some(Law::E { lam, lo, up })
    }
  }
}

/// I-4's family (`σt` past `MAX` with `μ + σ²t/2 < 0`), New Breakage 1's exactly cancelling exponents and their
/// neighbours an ulp off, and `|μ|` near `MAX` with `σ|t|` near it: each law with the `t` it is drawn for.
fn mgf(g: &mut Gen) -> Option<(Law, Vec<f64>)> {
  let mirror = g.coin(0.5);
  let (law, ts) = match g.next() % 3 {
    0 => {
      let sigma = g.uniform(1.0, 1.9);
      let mu = -g.uniform(1.0, 1.79) * 1e308;
      let t = (MAX / sigma * g.uniform(1.0, 1.5)).min(MAX);
      let lo = mu + mu.abs() * g.uniform(0.0, 1.5);
      (
        Law::N {
          mu,
          sigma,
          lo,
          up: INF,
        },
        vec![t, 0.5 * t, t.next_down()],
      )
    }
    1 => {
      let sigma = g.pick(&[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]);
      let t = (2.0 * (g.next() % 128) as f64 + 1.0) * 2f64.powi(10 + (g.next() % 1012) as i32);
      let exact = -0.5 * sigma * sigma * t;
      let mu = match g.next() % 3 {
        0 => exact,
        1 => exact.next_up(),
        _ => exact.next_down(),
      };
      let lo = mu + sigma * g.pick(&[0.0, 0.0, 1.0, 2.0]);
      (
        Law::N {
          mu,
          sigma,
          lo,
          up: INF,
        },
        vec![t, t.next_up(), t.next_down()],
      )
    }
    _ => {
      let mu = g.sign() * MAX * g.uniform(0.5, 1.0);
      let sigma = g.log_uniform(1e-3, 1e3);
      let t = g.sign() * (MAX / sigma * g.uniform(0.25, 1.0)).min(MAX);
      let lo = mu - sigma * g.uniform(0.0, 4.0);
      let up = if g.coin(0.5) {
        INF
      } else {
        (lo + sigma * g.uniform(0.5, 8.0)).min(MAX)
      };
      (Law::N { mu, sigma, lo, up }, vec![t, -t, 1.0 / sigma])
    }
  };
  if !law.admissible() || ts.iter().any(|t| !t.is_finite()) {
    return None;
  }
  let Law::N { mu, sigma, lo, up } = law else {
    return None;
  };
  if mirror {
    let ts = ts.into_iter().map(|t| -t).collect();
    Some((
      Law::N {
        mu: -mu,
        sigma,
        lo: -up,
        up: -lo,
      },
      ts,
    ))
  } else {
    Some((law, ts))
  }
}
