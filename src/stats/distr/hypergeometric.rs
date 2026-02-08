use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

pub struct SimdHypergeometric {
  n_total: u32,
  k_success: u32,
  n_draws: u32,
  buffer: UnsafeCell<[u32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdHypergeometric {
  pub fn new(n_total: u32, k_success: u32, n_draws: u32) -> Self {
    Self {
      n_total,
      k_success,
      n_draws,
      buffer: UnsafeCell::new([0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [u32]) {
    for x in out.iter_mut() {
      let mut count = 0;
      let mut rem_succ = self.k_success;
      let mut rem_tot = self.n_total;
      let mut draws = self.n_draws;
      while draws > 0 {
        let u: f32 = rng.random_range(0.0..1.0);
        if u < (rem_succ as f32) / (rem_tot as f32) {
          count += 1;
          rem_succ -= 1;
        }
        rem_tot -= 1;
        draws -= 1;
      }
      *x = count;
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<u32> for SimdHypergeometric {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}
