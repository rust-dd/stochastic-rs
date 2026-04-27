//! `FloatExt` implementations for `f64` and `f32`, plus thread-local scratch
//! buffers used by the Fgn circulant embedding.

use std::cell::RefCell;

use ndarray::Array1;
use num_complex::Complex;

use crate::normal::SimdNormal;
use crate::traits::FloatExt;

thread_local! {
  static STANDARD_NORMAL_F64: RefCell<Option<Box<SimdNormal<f64, 64>>>> = const { RefCell::new(None) };
  static STANDARD_NORMAL_F32: RefCell<Option<Box<SimdNormal<f32, 64>>>> = const { RefCell::new(None) };
  static FGN_SCRATCH_F64: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
  static FGN_SCRATCH_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
}

impl FloatExt for f64 {
  fn from_usize_(n: usize) -> Self {
    n as f64
  }

  fn fill_standard_normal_slice(out: &mut [Self]) {
    if out.is_empty() {
      return;
    }
    STANDARD_NORMAL_F64.with(|cell| {
      let mut slot = cell.borrow_mut();
      let dist = slot.get_or_insert_with(|| Box::new(SimdNormal::new(0.0, 1.0)));
      dist.fill_standard_fast(out);
    });
  }

  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R {
    FGN_SCRATCH_F64.with(|scratch| {
      let mut scratch = scratch.borrow_mut();
      if scratch.len() < len {
        scratch.resize(len, Complex::new(0.0, 0.0));
      }
      f(&mut scratch[..len])
    })
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    assert!(std_dev > 0.0, "std_dev must be positive");
    let mut out = Array1::<f64>::zeros(n);
    if n == 0 {
      return out;
    }
    let out_slice = out
      .as_slice_mut()
      .expect("normal_array output must be contiguous");
    Self::fill_standard_normal_slice(out_slice);
    for x in out_slice.iter_mut() {
      *x = mean + std_dev * *x;
    }
    out
  }
}

impl FloatExt for f32 {
  fn from_usize_(n: usize) -> Self {
    n as f32
  }

  fn fill_standard_normal_slice(out: &mut [Self]) {
    if out.is_empty() {
      return;
    }
    STANDARD_NORMAL_F32.with(|cell| {
      let mut slot = cell.borrow_mut();
      let dist = slot.get_or_insert_with(|| Box::new(SimdNormal::new(0.0, 1.0)));
      dist.fill_standard_fast(out);
    });
  }

  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R {
    FGN_SCRATCH_F32.with(|scratch| {
      let mut scratch = scratch.borrow_mut();
      if scratch.len() < len {
        scratch.resize(len, Complex::new(0.0, 0.0));
      }
      f(&mut scratch[..len])
    })
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    assert!(std_dev > 0.0, "std_dev must be positive");
    let mut out = Array1::<f32>::zeros(n);
    if n == 0 {
      return out;
    }
    let out_slice = out
      .as_slice_mut()
      .expect("normal_array output must be contiguous");
    Self::fill_standard_normal_slice(out_slice);
    for x in out_slice.iter_mut() {
      *x = mean + std_dev * *x;
    }
    out
  }
}
