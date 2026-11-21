//! The family vocabulary as the GPU computes it.

/// The kernel entry `name` applied to each value: one `f32` in, one out.
#[cfg(metal_backend)]
pub(super) fn unary_kat(name: &str, values: &[f32]) -> Vec<f32> {
  use metal_oxide::{Argument, DynamicLaunchConfig, Launch, Operation, Pipeline};

  let n = values.len();
  let context = crate::device::metal::context(0).expect("a Metal device");
  let input = context.device.buffer_from_slice(values).unwrap();
  let mut output = context.device.buffer_zeroed::<f32>(n).unwrap();
  let pipeline = Pipeline::new(&context.device, &context.module, name).unwrap();
  let config = DynamicLaunchConfig::for_elements(n as u32, 64).unwrap();
  // SAFETY: both buffers hold `n` values; nothing else holds them.
  unsafe {
    Launch::new(
      Ok(&pipeline),
      config,
      (&input, &mut output, n as u32),
      |(input, output, cases)| {
        Ok([
          Argument::read(input),
          Argument::write(output),
          Argument::value(*cases)?,
        ])
      },
    )
    .wait()
  }
  .unwrap();
  output.read().wait().to_vec()
}

/// Every top uniform `1 − (2k + 1)·2⁻²⁴`, the values just above 1, and a grid over `(0, 2]`.
#[cfg(metal_backend)]
fn log_cases() -> Vec<f32> {
  let mut values = (0..4096u32)
    .map(|k| 1.0 - (2 * k + 1) as f32 * (f32::EPSILON / 2.0))
    .collect::<Vec<_>>();
  values.extend((1..4096u32).map(|k| 1.0 + k as f32 * f32::EPSILON));
  values.extend((1..=2048u32).map(|k| k as f32 / 1024.0));
  values
}

/// Near 1 the family log keeps its sign and its relative digits, so `−ln u` of a uniform is
/// positive; elsewhere it is the fast log, within that log's absolute error.
#[cfg(metal_backend)]
#[test]
fn metal_family_log_keeps_its_sign_and_digits_near_one() {
  let values = log_cases();
  let logs = unary_kat("log_kat", &values);
  for (&x, &got) in values.iter().zip(&logs) {
    let want = (x as f64).ln();
    let got = got as f64;
    assert_eq!(got < 0.0, x < 1.0, "ln {x:e} = {got:e} has the wrong sign");
    if (x - 1.0).abs() < 0.031_25 {
      assert!(
        (got - want).abs() <= 4.0 * f32::EPSILON as f64 * want.abs(),
        "ln {x:e}: {got:e} against {want:e}"
      );
    } else {
      assert!(
        (got - want).abs() <= 1e-6 + 8.0 * f32::EPSILON as f64 * want.abs(),
        "ln {x:e}: {got:e} against {want:e}"
      );
    }
  }
}
