use ndarray::ArrayView2;

use super::*;

/// The device primitive of the Euler engine: one launch under one seed. A device handle that
/// implements it gets [`EulerBackend`] from `kernel_euler_backend!`; the host handles do not.
#[doc(hidden)]
pub trait EulerKernel<T: FloatExt>: Backend {
  /// Paths `first .. first + m` of the launch stream seeded by `seed`, column 0 the initial value;
  /// every draw's counter carries the global path, so chunks under one seed equal one launch.
  fn euler_kernel<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>, DeviceError>;

  /// [`euler_kernel`](Self::euler_kernel) with the launch lent to `f` as an `m × n` view valid
  /// only for the call. The default copies; a unified-memory device overrides it to lend in place.
  fn euler_kernel_lend<P: EulerCoefficients<T>, O>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
    f: impl FnOnce(ArrayView2<T>) -> O,
  ) -> Result<O, DeviceError> {
    let chunk = self.euler_kernel(process, first, m, seed)?;
    Ok(f(chunk.view()))
  }

  /// Paths `first .. first + m` of a system's launch stream, as a
  /// `components × m × n` array.
  fn euler_system_kernel<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array3<T>, DeviceError>;

  /// Bytes of path data one launch may hold.
  fn batch_budget(&self) -> usize;

  /// `rows` cut until a launch of `spec` over `steps` points, with the `fgn` pipeline first, stays
  /// in the kernels' index type; never below one, so a lone oversized path is refused, not wrapped.
  fn index_rows(
    &self,
    rows: usize,
    spec: EulerSpec<T>,
    steps: usize,
    fgn: Option<FgnSpec<'_, T>>,
  ) -> usize;

  /// The whole system batch under `seed`, chunked to the budget.
  fn euler_system_batch<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array3<T>, DeviceError> {
    check_path_index(m)?;
    let n = process.grid_points();
    let (planes, rows) = system_rows(self, process);
    if m <= rows {
      return self.euler_system_kernel(process, 0, m, seed);
    }
    let mut out = Array3::<T>::zeros((planes, m, n));
    crate::device::over_chunks(m, rows, |first, len| {
      let chunk = self.euler_system_kernel(process, first, len, seed)?;
      out
        .slice_mut(ndarray::s![.., first..first + len, ..])
        .assign(&chunk);
      Ok(())
    })?;
    Ok(out)
  }

  /// The whole batch under `seed`, chunked to the budget. A device may
  /// override it to pipeline the chunks; the result must stay bit-identical.
  fn euler_kernel_batch<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>, DeviceError> {
    check_path_index(m)?;
    let n = process.grid_points();
    let rows = launch_rows(self, n, process.euler_spec(), n, process.fgn_spec());
    if m <= rows {
      return self.euler_kernel(process, 0, m, seed);
    }
    let mut out = Array2::<T>::zeros((m, n));
    crate::device::over_chunks(m, rows, |first, len| {
      let chunk = self.euler_kernel(process, first, len, seed)?;
      out
        .slice_mut(ndarray::s![first..first + len, ..])
        .assign(&chunk);
      Ok(())
    })?;
    Ok(out)
  }
}

/// `m` paths, or the launch's error before anything is allocated: past `u32::MAX` the kernels'
/// `unsigned int` path index cannot reach them all, whatever the chunks.
pub(crate) fn check_path_index(m: usize) -> Result<(), DeviceError> {
  if m > u32::MAX as usize {
    return Err(DeviceError::Launch(format!(
      "{m} paths exceed the kernels' 32-bit path index"
    )));
  }
  Ok(())
}

/// The body's `steps`, `paths`, `first_path` as `unsigned int`s, or the launch's error; `first + m`
/// and `m · increments` are checked too: past 32 bits two paths would share noise or increments.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn launch_scalars(
  n: usize,
  first: usize,
  m: usize,
  increments: u32,
) -> Result<(u32, u32, u32), DeviceError> {
  use crate::device::launch_len;
  let steps = launch_len::<u32>(n, "steps")?;
  let paths = launch_len::<u32>(m, "paths")?;
  let first_path = launch_len::<u32>(first, "first_path")?;
  first_path.checked_add(paths).ok_or_else(|| {
    DeviceError::Launch("first_path + paths exceeds the kernel index range".into())
  })?;
  paths.checked_mul(increments).ok_or_else(|| {
    DeviceError::Launch("paths * increments exceeds the kernel index range".into())
  })?;
  Ok((steps, paths, first_path))
}
