//! Metal coefficient program results observed through the first Volterra step.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::fs::OpenOptions;
use std::io::Write;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Expr;
use stochastic_rs_stochastic::device::Metal;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volterra::kernel::ExponentialKernel;
use stochastic_rs_stochastic::volterra::sve::VolterraSde;

enum Observation {
  Finite,
  NaN,
  Infinity(bool),
}

struct Case {
  name: String,
  drift: Expr,
  diffusion: Expr,
  points: usize,
  observation: Observation,
}

impl Case {
  fn new(name: impl Into<String>, drift: Expr) -> Self {
    Self {
      name: name.into(),
      drift,
      diffusion: Expr::lit(0.0),
      points: 2,
      observation: Observation::Finite,
    }
  }

  fn pair(name: &str, drift: Expr, diffusion: Expr) -> Self {
    Self {
      diffusion,
      ..Self::new(name, drift)
    }
  }

  fn observes(mut self, observation: Observation) -> Self {
    self.observation = observation;
    self
  }

  fn sample(&self) -> Array1<f32> {
    let process = VolterraSde::<f32, _, _>::new(
      ExponentialKernel::new(1.0, 1.0),
      self.drift.compile().unwrap(),
      self.diffusion.compile().unwrap(),
      self.points,
      Some(-0.75),
      Some(1.0),
      Deterministic::new(233),
    )
    .on::<Metal>();
    assert!(
      process.device_ready(),
      "{} must execute on Metal",
      self.name
    );
    let path = process.try_sample().unwrap();
    let value = path[1];
    match self.observation {
      Observation::Finite => assert!(value.is_finite(), "{}: {value}", self.name),
      Observation::NaN => assert!(value.is_nan(), "{}: {value}", self.name),
      Observation::Infinity(negative) => {
        assert!(value.is_infinite(), "{}: {value}", self.name);
        assert_eq!(value.is_sign_negative(), negative, "{}", self.name);
      }
    }
    path
  }
}

fn binary(code: u32, left: Expr, right: Expr) -> Expr {
  match code {
    3 => left + right,
    4 => left - right,
    5 => left * right,
    6 => left / right,
    7 => left.powf(right),
    8 => left.max(right),
    9 => left.min(right),
    _ => unreachable!(),
  }
}

fn unary(code: u32, value: Expr) -> Expr {
  match code {
    10 => -value,
    11 => value.sqrt(),
    12 => value.exp(),
    13 => value.ln(),
    14 => value.abs(),
    15 => value.tanh(),
    _ => unreachable!(),
  }
}

fn negations(mut value: Expr, count: usize) -> Expr {
  for _ in 0..count {
    value = -value;
  }
  value
}

fn cases() -> Vec<Case> {
  let mut cases = Vec::new();
  for code in 0..16 {
    let expression = match code {
      0 => Expr::t(),
      1 => Expr::x(),
      2 => Expr::lit(2.25),
      3..=9 => binary(code, Expr::lit(2.25), Expr::lit(0.75)),
      _ => unary(code, Expr::lit(0.75)),
    };
    cases.push(Case::new(format!("opcode-{code}"), expression));
  }
  let depth_eight = (1..8)
    .rev()
    .fold(Expr::lit(8.0), |sum, value| Expr::lit(value as f64) + sum);
  assert_eq!(depth_eight.compile().unwrap().depth(), 8);
  let max_length = negations(Expr::lit(0.5), 61);
  assert_eq!(max_length.compile().unwrap().len(), 62);
  let both_limits = negations(depth_eight.clone(), 47);
  let program = both_limits.compile().unwrap();
  assert_eq!((program.len(), program.depth()), (62, 8));
  cases.push(Case::new("depth-eight", depth_eight));
  cases.push(Case::new("length-sixty-two", max_length.clone()));
  cases.push(Case::new(
    "depth-eight-length-sixty-two",
    both_limits.clone(),
  ));
  cases.push(Case::pair(
    "two-short-first",
    Expr::t(),
    both_limits.clone(),
  ));
  cases.push(Case::pair(
    "two-short-second",
    both_limits.clone(),
    Expr::x(),
  ));
  cases.push(Case::pair("two-maximum-lengths", max_length, both_limits));
  let mut time_and_state = Case::pair(
    "time-and-state",
    Expr::t() * Expr::x() + 0.5,
    -Expr::x() - Expr::t(),
  );
  time_and_state.points = 3;
  cases.push(time_and_state);
  for code in [8, 9] {
    for (name, left, right, observation) in [
      ("nan-left", f64::NAN, 1.0, Observation::Finite),
      ("nan-right", 1.0, f64::NAN, Observation::Finite),
      ("nan-both", f64::NAN, f64::NAN, Observation::NaN),
    ] {
      cases.push(
        Case::new(
          format!("opcode-{code}-{name}"),
          binary(code, Expr::lit(left), Expr::lit(right)),
        )
        .observes(observation),
      );
    }
    for (left, right) in [(0.0_f64, -0.0_f64), (-0.0, 0.0), (-0.0, -0.0), (0.0, 0.0)] {
      // Taking the reciprocal inside the program preserves zero signs across the process sum.
      let expression = 1.0 / binary(code, Expr::lit(left), Expr::lit(right));
      cases.push(
        Case::new(
          format!(
            "opcode-{code}-zeros-{:016x}-{:016x}",
            left.to_bits(),
            right.to_bits()
          ),
          expression,
        )
        .observes(Observation::Infinity(right.is_sign_negative())),
      );
    }
  }
  for code in [10, 11, 14, 15] {
    for value in [0.0_f64, -0.0] {
      let negative = match code {
        10 => !value.is_sign_negative(),
        11 => value.is_sign_negative(),
        _ => false,
      };
      cases.push(
        Case::new(
          format!("opcode-{code}-zero-{:016x}", value.to_bits()),
          1.0 / unary(code, Expr::lit(value)),
        )
        .observes(Observation::Infinity(negative)),
      );
    }
  }
  for (name, expression, observation) in [
    (
      "divide-zero",
      Expr::lit(1.0) / 0.0,
      Observation::Infinity(false),
    ),
    (
      "divide-negative-zero",
      Expr::lit(1.0) / -0.0,
      Observation::Infinity(true),
    ),
    ("divide-zero-zero", Expr::lit(0.0) / 0.0, Observation::NaN),
    (
      "pow-negative-fractional",
      Expr::lit(-2.0).powf(0.5),
      Observation::NaN,
    ),
    (
      "pow-nan-zero",
      Expr::lit(f64::NAN).powf(0.0),
      Observation::Finite,
    ),
    ("sqrt-negative", Expr::lit(-1.0).sqrt(), Observation::NaN),
    (
      "exp-infinity",
      Expr::lit(f64::INFINITY).exp(),
      Observation::Infinity(false),
    ),
    (
      "exp-negative-infinity",
      1.0 / Expr::lit(f64::NEG_INFINITY).exp(),
      Observation::Infinity(false),
    ),
    ("log-negative", Expr::lit(-1.0).ln(), Observation::NaN),
    ("log-zero", Expr::lit(0.0).ln(), Observation::Infinity(true)),
    (
      "tanh-infinity",
      Expr::lit(f64::INFINITY).tanh(),
      Observation::NaN,
    ),
  ] {
    cases.push(Case::new(name, expression).observes(observation));
  }
  cases
}

#[test]
#[ignore = "Run expression edge regressions explicitly on Metal hardware"]
fn program_paths_match_recording() {
  let mut pins = include_str!("fixtures/metal_euler_program.txt")
    .lines()
    .filter(|line| !line.starts_with('#') && !line.is_empty());
  for case in cases() {
    for (index, actual) in case.sample().into_iter().enumerate() {
      let mut fields = pins.next().expect("missing pin").split_whitespace();
      assert_eq!(fields.next().unwrap(), case.name);
      assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), index);
      let expected = u32::from_str_radix(fields.next().unwrap(), 16).unwrap();
      assert!(fields.next().is_none());
      let actual = actual.to_bits();
      assert_eq!(
        actual, expected,
        "{}[{index}]: {actual:08x} != {expected:08x}",
        case.name
      );
    }
  }
  assert!(pins.next().is_none(), "unused pins");
}

#[test]
#[ignore = "Record the Metal expression baseline explicitly"]
fn record_program_paths() {
  let path =
    std::env::var_os("STOCHASTIC_RS_PROGRAM_FIXTURE").expect("set a fresh path for the fixture");
  let label = std::env::var("STOCHASTIC_RS_PROGRAM_BASELINE_LABEL")
    .expect("set revision, GPU, OS, Xcode/SDK and math mode");
  assert!(!label.contains(['\n', '\r']));
  let mut file = OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(path)
    .unwrap();
  writeln!(file, "# {label}\n# case point f32_bits_hex").unwrap();
  for case in cases() {
    for (index, value) in case.sample().into_iter().enumerate() {
      writeln!(file, "{} {index} {:08x}", case.name, value.to_bits()).unwrap();
    }
  }
}
