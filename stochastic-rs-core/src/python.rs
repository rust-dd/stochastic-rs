//! Type-conversion shims used by the workspace-wide PyO3 wrapper macros.
//!
//! `IntoF32` / `IntoF64` allow a single macro body to dispatch ctor parameters
//! to either an `f32` or `f64` inner sampler, transparently lifting `f64` /
//! `Option<f64>` / `Vec<f64>` from Python into the right ndarray storage.

use ndarray::Array1;

pub trait IntoF32 {
  type Target;
  fn into_f32(self) -> Self::Target;
}

pub trait IntoF64 {
  type Target;
  fn into_f64(self) -> Self::Target;
}

impl IntoF32 for f64 {
  type Target = f32;
  fn into_f32(self) -> f32 {
    self as f32
  }
}

impl IntoF64 for f64 {
  type Target = f64;
  fn into_f64(self) -> f64 {
    self
  }
}

impl IntoF32 for Option<f64> {
  type Target = Option<f32>;
  fn into_f32(self) -> Option<f32> {
    self.map(|v| v as f32)
  }
}

impl IntoF64 for Option<f64> {
  type Target = Option<f64>;
  fn into_f64(self) -> Option<f64> {
    self
  }
}

impl IntoF32 for usize {
  type Target = usize;
  fn into_f32(self) -> usize {
    self
  }
}

impl IntoF64 for usize {
  type Target = usize;
  fn into_f64(self) -> usize {
    self
  }
}

impl IntoF32 for Option<usize> {
  type Target = Option<usize>;
  fn into_f32(self) -> Option<usize> {
    self
  }
}

impl IntoF64 for Option<usize> {
  type Target = Option<usize>;
  fn into_f64(self) -> Option<usize> {
    self
  }
}

impl IntoF32 for Option<bool> {
  type Target = Option<bool>;
  fn into_f32(self) -> Option<bool> {
    self
  }
}

impl IntoF64 for Option<bool> {
  type Target = Option<bool>;
  fn into_f64(self) -> Option<bool> {
    self
  }
}

impl IntoF32 for Vec<f64> {
  type Target = Array1<f32>;
  fn into_f32(self) -> Array1<f32> {
    Array1::from_vec(self).mapv(|v| v as f32)
  }
}

impl IntoF64 for Vec<f64> {
  type Target = Array1<f64>;
  fn into_f64(self) -> Array1<f64> {
    Array1::from_vec(self)
  }
}

impl IntoF32 for Option<Vec<f64>> {
  type Target = Option<Array1<f32>>;
  fn into_f32(self) -> Option<Array1<f32>> {
    self.map(|v| Array1::from_vec(v).mapv(|x| x as f32))
  }
}

impl IntoF64 for Option<Vec<f64>> {
  type Target = Option<Array1<f64>>;
  fn into_f64(self) -> Option<Array1<f64>> {
    self.map(Array1::from_vec)
  }
}

impl IntoF32 for u32 {
  type Target = u32;
  fn into_f32(self) -> u32 {
    self
  }
}

impl IntoF64 for u32 {
  type Target = u32;
  fn into_f64(self) -> u32 {
    self
  }
}

impl IntoF32 for String {
  type Target = String;
  fn into_f32(self) -> String {
    self
  }
}

impl IntoF64 for String {
  type Target = String;
  fn into_f64(self) -> String {
    self
  }
}
