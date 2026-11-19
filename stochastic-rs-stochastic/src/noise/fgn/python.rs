//! The Python class of `Fgn`, in `f32` or `f64` by its `dtype` argument.
use super::Fgn;

py_process_1d!(PyFgn, Fgn,
  sig: (hurst, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, n: usize, t: Option<f64>),
  device
);
