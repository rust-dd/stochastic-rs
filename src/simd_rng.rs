use rand::RngCore;
use wide::{i32x8, u32x8, u64x4};

#[inline(always)]
fn rotl_u64x4(x: u64x4, k: u32) -> u64x4 {
  (x << k) | (x >> (64 - k))
}

#[inline(always)]
fn rotl_u32x8(x: u32x8, k: u32) -> u32x8 {
  (x << k) | (x >> (32 - k))
}

struct Xoshiro256PP4 {
  s0: u64x4,
  s1: u64x4,
  s2: u64x4,
  s3: u64x4,
}

impl Xoshiro256PP4 {
  fn new_from_rng(rng: &mut impl RngCore) -> Self {
    let mut seed = [0u8; 128];
    rng.fill_bytes(&mut seed);
    let u = unsafe { core::mem::transmute::<[u8; 128], [u64; 16]>(seed) };
    Self {
      s0: u64x4::new([u[0], u[1], u[2], u[3]]),
      s1: u64x4::new([u[4], u[5], u[6], u[7]]),
      s2: u64x4::new([u[8], u[9], u[10], u[11]]),
      s3: u64x4::new([u[12], u[13], u[14], u[15]]),
    }
  }

  #[inline(always)]
  fn next(&mut self) -> u64x4 {
    let result = rotl_u64x4(self.s0 + self.s3, 23) + self.s0;
    let t = self.s1 << 17u32;
    self.s2 = self.s2 ^ self.s0;
    self.s3 = self.s3 ^ self.s1;
    self.s1 = self.s1 ^ self.s2;
    self.s0 = self.s0 ^ self.s3;
    self.s2 = self.s2 ^ t;
    self.s3 = rotl_u64x4(self.s3, 45);
    result
  }
}

struct Xoshiro128PP8 {
  s0: u32x8,
  s1: u32x8,
  s2: u32x8,
  s3: u32x8,
}

impl Xoshiro128PP8 {
  fn new_from_rng(rng: &mut impl RngCore) -> Self {
    let mut seed = [0u8; 128];
    rng.fill_bytes(&mut seed);
    let u = unsafe { core::mem::transmute::<[u8; 128], [u32; 32]>(seed) };
    Self {
      s0: u32x8::new([u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]]),
      s1: u32x8::new([u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]]),
      s2: u32x8::new([u[16], u[17], u[18], u[19], u[20], u[21], u[22], u[23]]),
      s3: u32x8::new([u[24], u[25], u[26], u[27], u[28], u[29], u[30], u[31]]),
    }
  }

  #[inline(always)]
  fn next(&mut self) -> u32x8 {
    let result = rotl_u32x8(self.s0 + self.s3, 7) + self.s0;
    let t = self.s1 << 9u32;
    self.s2 = self.s2 ^ self.s0;
    self.s3 = self.s3 ^ self.s1;
    self.s1 = self.s1 ^ self.s2;
    self.s0 = self.s0 ^ self.s3;
    self.s2 = self.s2 ^ t;
    self.s3 = rotl_u32x8(self.s3, 11);
    result
  }
}

const F64_SCALE: f64 = 1.0 / (1u64 << 53) as f64;
const F32_SCALE: f32 = 1.0 / (1u32 << 24) as f32;

pub struct SimdRng {
  f64_engine: Xoshiro256PP4,
  f32_engine: Xoshiro128PP8,
  u64_buf: [u64; 4],
  u64_idx: usize,
}

impl SimdRng {
  pub fn new() -> Self {
    let mut rng = rand::rng();
    Self {
      f64_engine: Xoshiro256PP4::new_from_rng(&mut rng),
      f32_engine: Xoshiro128PP8::new_from_rng(&mut rng),
      u64_buf: [0; 4],
      u64_idx: 4,
    }
  }

  #[inline(always)]
  pub fn next_i32x8(&mut self) -> i32x8 {
    let raw = self.f32_engine.next().to_array();
    i32x8::new(unsafe { core::mem::transmute::<[u32; 8], [i32; 8]>(raw) })
  }

  #[inline(always)]
  pub fn next_f64_array(&mut self) -> [f64; 8] {
    let a = self.f64_engine.next().to_array();
    let b = self.f64_engine.next().to_array();
    [
      (a[0] >> 11) as f64 * F64_SCALE,
      (a[1] >> 11) as f64 * F64_SCALE,
      (a[2] >> 11) as f64 * F64_SCALE,
      (a[3] >> 11) as f64 * F64_SCALE,
      (b[0] >> 11) as f64 * F64_SCALE,
      (b[1] >> 11) as f64 * F64_SCALE,
      (b[2] >> 11) as f64 * F64_SCALE,
      (b[3] >> 11) as f64 * F64_SCALE,
    ]
  }

  #[inline(always)]
  pub fn next_f32_array(&mut self) -> [f32; 8] {
    let a = self.f32_engine.next().to_array();
    [
      (a[0] >> 8) as f32 * F32_SCALE,
      (a[1] >> 8) as f32 * F32_SCALE,
      (a[2] >> 8) as f32 * F32_SCALE,
      (a[3] >> 8) as f32 * F32_SCALE,
      (a[4] >> 8) as f32 * F32_SCALE,
      (a[5] >> 8) as f32 * F32_SCALE,
      (a[6] >> 8) as f32 * F32_SCALE,
      (a[7] >> 8) as f32 * F32_SCALE,
    ]
  }
}

impl RngCore for SimdRng {
  #[inline(always)]
  fn next_u32(&mut self) -> u32 {
    self.next_u64() as u32
  }

  #[inline(always)]
  fn next_u64(&mut self) -> u64 {
    if self.u64_idx >= 4 {
      self.u64_buf = self.f64_engine.next().to_array();
      self.u64_idx = 0;
    }
    let val = self.u64_buf[self.u64_idx];
    self.u64_idx += 1;
    val
  }

  fn fill_bytes(&mut self, dest: &mut [u8]) {
    let mut chunks = dest.chunks_exact_mut(8);
    for chunk in &mut chunks {
      chunk.copy_from_slice(&self.next_u64().to_le_bytes());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let bytes = self.next_u64().to_le_bytes();
      rem.copy_from_slice(&bytes[..rem.len()]);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn f64_in_range() {
    let mut rng = SimdRng::new();
    for _ in 0..1000 {
      let vals = rng.next_f64_array();
      for v in vals {
        assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
      }
    }
  }

  #[test]
  fn f32_in_range() {
    let mut rng = SimdRng::new();
    for _ in 0..1000 {
      let vals = rng.next_f32_array();
      for v in vals {
        assert!((0.0..1.0).contains(&v), "f32 out of range: {v}");
      }
    }
  }

  #[test]
  fn rng_core_works() {
    let mut rng = SimdRng::new();
    let a = rng.next_u64();
    let b = rng.next_u64();
    assert_ne!(a, b);
  }
}
