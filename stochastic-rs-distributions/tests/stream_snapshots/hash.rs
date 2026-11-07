pub fn fnv1a(bits: impl IntoIterator<Item = u64>) -> u64 {
  bits
    .into_iter()
    .fold(0xcbf2_9ce4_8422_2325_u64, |mut h, b| {
      for byte in b.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
      }
      h
    })
}

pub trait Bits {
  fn bits(self) -> u64;
}

impl Bits for f64 {
  fn bits(self) -> u64 {
    self.to_bits()
  }
}

impl Bits for f32 {
  fn bits(self) -> u64 {
    self.to_bits() as u64
  }
}

impl Bits for u64 {
  fn bits(self) -> u64 {
    self
  }
}

impl Bits for u32 {
  fn bits(self) -> u64 {
    self as u64
  }
}

impl Bits for i64 {
  fn bits(self) -> u64 {
    self as u64
  }
}
