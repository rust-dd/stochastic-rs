use std::any::TypeId;

use ndarray::Array2;

use crate::traits::FloatExt;

fn array2_from_flat<T: FloatExt, U: Copy + Into<f64>>(
  host: &[U],
  m: usize,
  cols: usize,
) -> Array2<T> {
  let mut out = Array2::<T>::zeros((m, cols));
  for i in 0..m {
    for j in 0..cols {
      out[[i, j]] = T::from_f64_fast(host[i * cols + j].into());
    }
  }
  out
}

pub(super) fn array2_from_vec_f32<T: FloatExt>(v: Vec<f32>, m: usize, cols: usize) -> Array2<T> {
  if TypeId::of::<T>() == TypeId::of::<f32>() {
    let out = Array2::<f32>::from_shape_vec((m, cols), v).expect("shape must be valid");
    unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) }
  } else {
    array2_from_flat::<T, f32>(&v, m, cols)
  }
}

pub(super) fn array2_from_vec_f64<T: FloatExt>(v: Vec<f64>, m: usize, cols: usize) -> Array2<T> {
  if TypeId::of::<T>() == TypeId::of::<f64>() {
    let out = Array2::<f64>::from_shape_vec((m, cols), v).expect("shape must be valid");
    unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(out) }
  } else {
    array2_from_flat::<T, f64>(&v, m, cols)
  }
}
