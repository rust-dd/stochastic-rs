//! The stable jump law as the GPU draws it: its sine and cosine near their zeros, and its jump
//! at the smallest exponential the uniform map gives and where its sine vanishes.

/// The 4096 largest `f32` below `π/2` and their negatives, a grid over `(−π/2, π/2)`, and the
/// values either side of `π/4`, where the cosine changes its polynomial.
#[cfg(metal_backend)]
fn cos_cases() -> Vec<f32> {
  use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

  let top = FRAC_PI_2.to_bits() - 1;
  let mut values = (0..4096u32)
    .map(|k| f32::from_bits(top - k))
    .flat_map(|x| [x, -x])
    .collect::<Vec<_>>();
  values.extend((-2047..=2047).map(|k| k as f32 * (FRAC_PI_2 / 2048.0)));
  values.extend((0..64u32).flat_map(|k| {
    [
      f32::from_bits(FRAC_PI_4.to_bits() + k),
      f32::from_bits(FRAC_PI_4.to_bits() - k),
    ]
  }));
  values
}

/// The 4096 largest `f32` below `π` and their negatives, values near 0, a grid over `(−π, π)`, and
/// the values either side of `π/4` and `3π/4`, where the sine changes its polynomial.
#[cfg(metal_backend)]
fn sin_cases() -> Vec<f32> {
  use std::f32::consts::{FRAC_PI_4, PI};

  let top = PI.to_bits() - 1;
  let mut values = (0..4096u32)
    .map(|k| f32::from_bits(top - k))
    .chain([
      2f32.powi(-100),
      2f32.powi(-60),
      2f32.powi(-30),
      f32::EPSILON,
    ])
    .flat_map(|x| [x, -x])
    .collect::<Vec<_>>();
  values.extend(
    (-4095..=4095)
      .filter(|&k| k != 0)
      .map(|k| k as f32 * (PI / 4096.0)),
  );
  for edge in [FRAC_PI_4, 3.0 * FRAC_PI_4] {
    values.extend((0..64u32).flat_map(|k| {
      [
        f32::from_bits(edge.to_bits() + k),
        f32::from_bits(edge.to_bits() - k),
      ]
    }));
  }
  values
}

/// `(words, α, count)`: the words put `V` at both ends of `(−π/2, π/2)` and across it, and the
/// exponential at its smallest (`u32::MAX`), at `ln 2` and in its refined zero cell.
#[cfg(metal_backend)]
fn stable_cases() -> Vec<([u32; 4], f32, f32)> {
  let angles = [
    0,
    1 << 9,
    2 << 9,
    0x2000_0000,
    0x4000_0000,
    0x6000_0000,
    0x8000_0000,
    0xa000_0000,
    0xc000_0000,
    0xe000_0000,
    u32::MAX - (1 << 9),
    u32::MAX,
  ];
  let mut cases = Vec::new();
  for first in angles {
    for third in [u32::MAX, 0x8000_0000, 0x1ff] {
      for alpha in [0.5f32, 0.9, 1.0, 1.7, 2.0] {
        for count in [0.0f32, 1.0, 4.0] {
          cases.push(([first, 0x9e37_79b9, third, 0x7f4a_7c15], alpha, count));
        }
      }
    }
  }
  cases
}

/// `(words, α, count)` where the law's sine vanishes: `α` near 2 and `V` within a few ulp of ±π/2,
/// the exponential at `ln 2`, at `ln 4` and in its refined zero cell.
#[cfg(metal_backend)]
fn vanishing_sine_cases() -> Vec<([u32; 4], f32, f32)> {
  let angles = [
    0,
    1 << 9,
    2 << 9,
    7 << 9,
    u32::MAX - (6 << 9),
    u32::MAX - (1 << 9),
    u32::MAX,
  ];
  let mut cases = Vec::new();
  for first in angles {
    for third in [0x8000_0000, 0x4000_0000, 0x1ff] {
      for alpha in [1.99f32, 1.999, 2.0] {
        for count in [1.0f32, 4.0] {
          cases.push(([first, 0x9e37_79b9, third, 0x7f4a_7c15], alpha, count));
        }
      }
    }
  }
  cases
}

/// The Chambers–Mallows–Stuck jump in `f64` from the kernel's `f32` angle, its two `f32` products
/// and its exponential.
#[cfg(metal_backend)]
fn reference(words: [u32; 4], alpha: f32, count: f32) -> f64 {
  use crate::euler::philox::u01_f32;
  use crate::euler::tests::philox::exp1_f32;

  let angle = (u01_f32(words[0]) - 0.5) * std::f32::consts::PI;
  let (sine, cosine) = ((alpha * angle) as f64, (angle - alpha * angle) as f64);
  let inverse = 1.0 / alpha as f64;
  let ratio = cosine.cos() / exp1_f32(words[2], words[3]) as f64;
  (count as f64).powf(inverse) * sine.sin() / (angle as f64).cos().powf(inverse)
    * ratio.powf((1.0 - alpha as f64) * inverse)
}

/// The law's jump for each case, from the kernel entry `stable_kat` at scale 1.
#[cfg(metal_backend)]
fn stable_jumps(cases: &[([u32; 4], f32, f32)]) -> Vec<f32> {
  use metal_oxide::{Argument, DynamicLaunchConfig, Launch, Operation, Pipeline};

  let n = cases.len();
  let words = cases.iter().flat_map(|c| c.0).collect::<Vec<_>>();
  let laws = cases
    .iter()
    .flat_map(|&(_, alpha, count)| [alpha, 1.0, count])
    .collect::<Vec<_>>();
  let context = crate::device::metal::context(0).expect("a Metal device");
  let words_in = context.device.buffer_from_slice(&words).unwrap();
  let laws_in = context.device.buffer_from_slice(&laws).unwrap();
  let mut jumps = context.device.buffer_zeroed::<f32>(n).unwrap();
  let pipeline = Pipeline::new(&context.device, &context.module, "stable_kat").unwrap();
  let config = DynamicLaunchConfig::for_elements(n as u32, 64).unwrap();
  // SAFETY: the buffers hold `n` cases of four, three and one values; nothing else holds them.
  unsafe {
    Launch::new(
      Ok(&pipeline),
      config,
      (&words_in, &laws_in, &mut jumps, n as u32),
      |(words, laws, jumps, cases)| {
        Ok([
          Argument::read(words),
          Argument::read(laws),
          Argument::write(jumps),
          Argument::value(*cases)?,
        ])
      },
    )
    .wait()
  }
  .unwrap();
  jumps.read().wait().to_vec()
}

/// Within `π/2` of zero the law's cosine keeps four ulp of its digits, and so its sign, where a
/// fast cosine's absolute error of 2⁻¹³ may outweigh a result near ±π/2.
#[cfg(metal_backend)]
#[test]
fn metal_stable_cosine_keeps_its_digits_near_a_right_angle() {
  let values = cos_cases();
  for (&x, &got) in values
    .iter()
    .zip(&super::vocabulary::unary_kat("cos_kat", &values))
  {
    let want = (x as f64).cos();
    assert!(
      (got as f64 - want).abs() <= 4.0 * f32::EPSILON as f64 * want,
      "cos {x:e}: {got:e} against {want:e}"
    );
  }
}

/// Within `π` of zero the law's sine keeps four ulp of its digits, and so its sign, where a fast
/// sine's absolute error of 2⁻¹³ may outweigh a result near 0 or ±π.
#[cfg(metal_backend)]
#[test]
fn metal_stable_sine_keeps_its_digits_near_zero_and_pi() {
  let values = sin_cases();
  for (&x, &got) in values
    .iter()
    .zip(&super::vocabulary::unary_kat("sin_kat", &values))
  {
    let want = (x as f64).sin();
    assert!(
      (got as f64 - want).abs() <= 4.0 * f32::EPSILON as f64 * want.abs(),
      "sin {x:e}: {got:e} against {want:e}"
    );
  }
}

/// At the smallest exponential the jump is finite for every α, zero without arrivals, and the
/// `f64` draw to 1e-3, a bound the fast powers keep with room.
#[cfg(metal_backend)]
#[test]
fn metal_stable_jump_stays_finite_at_the_smallest_exponential() {
  let cases = stable_cases();
  for (&(words, alpha, count), &got) in cases.iter().zip(&stable_jumps(&cases)) {
    let case = format!("words {words:08x?}, alpha {alpha}, count {count}");
    assert!(got.is_finite(), "{case}: {got}");
    if count == 0.0 {
      assert_eq!(got, 0.0, "{case}");
    } else {
      let want = reference(words, alpha, count);
      assert!(
        (got as f64 - want).abs() <= 1e-3 * want.abs(),
        "{case}: {got:e} against {want:e}"
      );
    }
  }
}

/// Near α = 2 and V = ±π/2 the jump is a sine near zero over a cosine near zero: it keeps the `f64`
/// draw to 1e-3 where a fast sine's 2⁻¹³ would be hundreds of times the sine.
#[cfg(metal_backend)]
#[test]
fn metal_stable_jump_keeps_its_digits_where_its_sine_vanishes() {
  let cases = vanishing_sine_cases();
  for (&(words, alpha, count), &got) in cases.iter().zip(&stable_jumps(&cases)) {
    let want = reference(words, alpha, count);
    assert!(
      (got as f64 - want).abs() <= 1e-3 * want.abs(),
      "words {words:08x?}, alpha {alpha}, count {count}: {got:e} against {want:e}"
    );
  }
}
