//! The stable jump law as the GPU draws it: its cosine near a right angle, and its jump at the
//! smallest exponential the uniform map gives.

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

/// The Chambers–Mallows–Stuck jump in `f64` from the kernel's `f32` angle and exponential, or
/// `None` where `sin(αV)` is within 0.1 of zero and a fast sine's absolute error is not small.
#[cfg(metal_backend)]
fn reference(words: [u32; 4], alpha: f32, count: f32) -> Option<f64> {
  use crate::euler::philox::u01_f32;
  use crate::euler::tests::philox::exp1_f32;

  let angle = ((u01_f32(words[0]) - 0.5) * std::f32::consts::PI) as f64;
  let (alpha, count) = (alpha as f64, count as f64);
  let sine = (alpha * angle).sin();
  (sine.abs() >= 0.1).then(|| {
    let ratio = ((1.0 - alpha) * angle).cos() / exp1_f32(words[2], words[3]) as f64;
    count.powf(1.0 / alpha) * sine / angle.cos().powf(1.0 / alpha)
      * ratio.powf((1.0 - alpha) / alpha)
  })
}

/// Within `π/2` of zero the law's cosine keeps four ulp of its digits, and so its sign, where a
/// fast cosine's absolute error of 2⁻¹³ may outweigh a result near ±π/2.
#[cfg(metal_backend)]
#[test]
fn metal_stable_cosine_keeps_its_digits_near_a_right_angle() {
  use metal_oxide::{Argument, DynamicLaunchConfig, Launch, Operation, Pipeline};

  let values = cos_cases();
  let n = values.len();
  let context = crate::device::metal::context(0).expect("a Metal device");
  let input = context.device.buffer_from_slice(&values).unwrap();
  let mut cosines = context.device.buffer_zeroed::<f32>(n).unwrap();
  let pipeline = Pipeline::new(&context.device, &context.module, "cos_kat").unwrap();
  let config = DynamicLaunchConfig::for_elements(n as u32, 64).unwrap();
  // SAFETY: both buffers hold `n` values; nothing else holds them.
  unsafe {
    Launch::new(
      Ok(&pipeline),
      config,
      (&input, &mut cosines, n as u32),
      |(input, cosines, cases)| {
        Ok([
          Argument::read(input),
          Argument::write(cosines),
          Argument::value(*cases)?,
        ])
      },
    )
    .wait()
  }
  .unwrap();
  let cosines = cosines.read().wait().to_vec();
  for (&x, &got) in values.iter().zip(&cosines) {
    let want = (x as f64).cos();
    assert!(
      (got as f64 - want).abs() <= 4.0 * f32::EPSILON as f64 * want,
      "cos {x:e}: {got:e} against {want:e}"
    );
  }
}

/// At the smallest exponential the jump is finite for every α, zero without arrivals, and the
/// `f64` draw to 2e-3 wherever a fast sine's 2⁻¹³ stays that small against `sin(αV)`.
#[cfg(metal_backend)]
#[test]
fn metal_stable_jump_stays_finite_at_the_smallest_exponential() {
  use metal_oxide::{Argument, DynamicLaunchConfig, Launch, Operation, Pipeline};

  let cases = stable_cases();
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
  let jumps = jumps.read().wait().to_vec();
  for (&(words, alpha, count), &got) in cases.iter().zip(&jumps) {
    let case = format!("words {words:08x?}, alpha {alpha}, count {count}");
    assert!(got.is_finite(), "{case}: {got}");
    if count == 0.0 {
      assert_eq!(got, 0.0, "{case}");
    } else if let Some(want) = reference(words, alpha, count) {
      assert!(
        (got as f64 - want).abs() <= 2e-3 * want.abs(),
        "{case}: {got:e} against {want:e}"
      );
    }
  }
}
