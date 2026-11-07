/// One law from one seed: seed budget, first eight draws, heads of steps 2–9 (one per fill of step 3) and of the worker chunks, ten hashes.
#[derive(Debug)]
pub struct Snapshot {
  pub seed: u64,
  pub seeds_consumed: u64,
  pub first8: [u64; 8],
  pub step_heads: [u64; 9],
  pub chunk_heads: [u64; 8],
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
  (got - want).abs() <= rel * (1.0 + want.abs())
}
