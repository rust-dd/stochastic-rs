/// One law from one seed: the constructor's seed budget, the first eight raw draws and ten step hashes.
#[derive(Debug)]
pub struct Snapshot {
  pub seed: u64,
  pub seeds_consumed: u64,
  pub first8: [u64; 8],
  pub hashes: [u64; 10],
}

#[derive(Clone, Copy)]
pub enum ScalarKind {
  F64,
  F32,
  Int,
}

impl ScalarKind {
  /// libm and FMA move the last bits across platforms; a stream change moves every bit.
  pub fn close(self, got: u64, want: u64) -> bool {
    match self {
      ScalarKind::Int => got == want,
      ScalarKind::F64 => within(f64::from_bits(got), f64::from_bits(want), 1e-9),
      ScalarKind::F32 => within(
        f64::from(f32::from_bits(got as u32)),
        f64::from(f32::from_bits(want as u32)),
        1e-5,
      ),
    }
  }
}

fn within(got: f64, want: f64, rel: f64) -> bool {
  (got - want).abs() <= rel * want.abs().max(f64::MIN_POSITIVE)
}
