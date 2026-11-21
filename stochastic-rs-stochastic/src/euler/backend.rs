use ndarray::ArrayView1;

use super::*;
use crate::traits::process::sample_map_chunked;
use crate::traits::process::sample_par_chunked;

/// How a backend handle produces Euler paths: a CPU handle runs the process's own sampler, a device
/// handle its [`EulerKernel`]; the `try_*` methods return a [`DeviceError`] where the rest panic.
pub trait EulerBackend<T: FloatExt>: Backend {
  /// One path.
  fn try_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Result<Array1<T>, DeviceError>;

  /// `m` paths.
  fn try_euler_paths<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// `f` over `m` paths, mapped as they are produced, so the batch never has
  /// to fit in memory at once.
  fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError>;

  /// [`try_euler_paths_map`](Self::try_euler_paths_map) with a borrowed row: a device chunk is
  /// one `m × n` block in host memory, so a view reads it once where the owning form copies rows.
  fn try_euler_paths_map_view<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(ArrayView1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError>;

  /// The batch as one `m × n` matrix; on a device the launch buffer itself.
  fn try_euler_matrix<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Array2<T>, DeviceError>;

  /// [`try_sample`](Self::try_sample), panicking with the device's error.
  fn euler_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Array1<T> {
    self
      .try_sample(process)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths`](Self::try_euler_paths), panicking with the device's
  /// error; [`Backend::probe`] first turns that failure into a `Result`.
  fn euler_paths<P: EulerCoefficients<T>>(&self, process: &P, m: usize) -> Vec<Array1<T>> {
    self
      .try_euler_paths(process, m)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths_map`](Self::try_euler_paths_map), panicking with the
  /// device's error.
  fn euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_euler_paths_map(process, m, f)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths_map_view`](Self::try_euler_paths_map_view), panicking
  /// with the device's error.
  fn euler_paths_map_view<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_euler_paths_map_view(process, m, f)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// One draw of a multi-component system, its `D` rows built into the process's output.
  fn try_system_sample<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
  ) -> Result<P::Output, DeviceError>;

  /// `m` draws of a multi-component system.
  fn try_system_paths<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Vec<P::Output>, DeviceError>;

  /// `f` over `m` draws of a system, mapped as they are produced.
  fn try_system_paths_map<const D: usize, P: EulerSystem<T, D>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&P::Output) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError>;

  /// [`try_system_sample`](Self::try_system_sample), panicking with the
  /// device's error.
  fn system_sample<const D: usize, P: EulerSystem<T, D>>(&self, process: &P) -> P::Output {
    self
      .try_system_sample(process)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_system_paths`](Self::try_system_paths), panicking with the
  /// device's error.
  fn system_paths<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    m: usize,
  ) -> Vec<P::Output> {
    self
      .try_system_paths(process, m)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_system_paths_map`](Self::try_system_paths_map), panicking with the
  /// device's error.
  fn system_paths_map<const D: usize, P: EulerSystem<T, D>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&P::Output) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_system_paths_map(process, m, f)
      .unwrap_or_else(crate::device::device_panic)
  }
}

/// A host handle samples through the process's own sampler, chunked the way
/// `ProcessExt` chunks, so its streams are those of the process.
macro_rules! host_euler_backend {
  ($handle:ty) => {
    impl<T: FloatExt> EulerBackend<T> for $handle {
      fn try_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Result<Array1<T>, DeviceError> {
        Ok(process.host_sample())
      }

      fn try_euler_paths<P: EulerCoefficients<T>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Vec<Array1<T>>, DeviceError> {
        Ok(sample_par_chunked(process, m))
      }

      fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
        &self,
        process: &P,
        m: usize,
        f: impl Fn(&Array1<T>) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(sample_map_chunked(process, m, f))
      }

      fn try_euler_paths_map_view<P: EulerCoefficients<T>, R: Send>(
        &self,
        process: &P,
        m: usize,
        f: impl Fn(ArrayView1<T>) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(sample_map_chunked(process, m, |path| f(path.view())))
      }

      fn try_euler_matrix<P: EulerCoefficients<T>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Array2<T>, DeviceError> {
        let rows = sample_par_chunked(process, m);
        let n = rows.first().map_or(process.grid_points(), |r| r.len());
        let mut out = Array2::<T>::zeros((m, n));
        for (i, row) in rows.iter().enumerate() {
          out.row_mut(i).assign(row);
        }
        Ok(out)
      }

      fn try_system_sample<const D: usize, P: EulerSystem<T, D>>(
        &self,
        process: &P,
      ) -> Result<P::Output, DeviceError> {
        Ok(process.host_sample())
      }

      fn try_system_paths<const D: usize, P: EulerSystem<T, D>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Vec<P::Output>, DeviceError> {
        Ok(sample_par_chunked(process, m))
      }

      fn try_system_paths_map<const D: usize, P: EulerSystem<T, D>, R: Send>(
        &self,
        process: &P,
        m: usize,
        f: impl Fn(&P::Output) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(sample_map_chunked(process, m, f))
      }
    }
  };
}

host_euler_backend!(Cpu);
#[cfg(accelerate_backend)]
host_euler_backend!(crate::device::Accelerate);

/// A device kernel is an Euler backend: one seed per call, chunks to the handle's budget, the map
/// applied per chunk in parallel. One impl per handle: coherence forbids a blanket one here.
#[cfg(device_backend)]
macro_rules! kernel_euler_backend {
  ($handle:ty, [$($gen:tt)*] $scalar:ty) => {
    impl<$($gen)*> EulerBackend<$scalar> for $handle {
    fn try_sample<P: EulerCoefficients<$scalar>>(&self, process: &P) -> Result<Array1<$scalar>, DeviceError> {
      let seed = process.device_seed();
      Ok(<Self as EulerKernel<$scalar>>::euler_kernel(self, process, 0, 1, seed)?.row(0).to_owned())
    }

    fn try_euler_paths<P: EulerCoefficients<$scalar>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Vec<Array1<$scalar>>, DeviceError> {
      let seed = process.device_seed();
      Ok(
        <Self as EulerKernel<$scalar>>::euler_kernel_batch(self, process, m, seed)?
          .outer_iter()
          .map(|row| row.to_owned())
          .collect(),
      )
    }

    fn try_euler_paths_map<P: EulerCoefficients<$scalar>, R: Send>(
      &self,
      process: &P,
      m: usize,
      f: impl Fn(&Array1<$scalar>) -> R + Sync,
    ) -> Result<Vec<R>, DeviceError> {
      use rayon::prelude::*;
      let seed = process.device_seed();
      check_path_index(m)?;
      let steps = process.grid_points();
      let rows = launch_rows(self, steps, process.euler_spec(), steps, process.fgn_spec());
      let mut out = Vec::with_capacity(m);
      // `over_chunks`, so a chunk too large for the device is retried at half
      // the size rather than failing the batch.
      crate::device::over_chunks(m, rows, |first, len| {
        // One buffer per rayon worker, not one allocation per path (about 68 ns each, more than
        // the kernel on short paths); `try_euler_paths_map_view` is the same walk without the copy.
        let chunk = <Self as EulerKernel<$scalar>>::euler_kernel_lend(
          self,
          process,
          first,
          len,
          seed,
          |view| {
            let n = view.ncols();
            match view.as_slice() {
              // The lent batch is contiguous, so rayon splits one slice into rows where it lies
              // instead of being fed a vector of views built first.
              Some(flat) => flat
                .par_chunks(n.max(1))
                .map_init(
                  || Array1::<$scalar>::zeros(n),
                  |buffer, row| {
                    buffer.assign(&ArrayView1::from(row));
                    f(buffer)
                  },
                )
                .collect::<Vec<R>>(),
              None => view
                .outer_iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map_init(
                  || Array1::<$scalar>::zeros(n),
                  |buffer, row| {
                    buffer.assign(&row);
                    f(buffer)
                  },
                )
                .collect::<Vec<R>>(),
            }
          },
        )?;
        out.extend(chunk);
        Ok(())
      })?;
      Ok(out)
    }

    fn try_euler_paths_map_view<P: EulerCoefficients<$scalar>, R: Send>(
      &self,
      process: &P,
      m: usize,
      f: impl Fn(ArrayView1<$scalar>) -> R + Sync,
    ) -> Result<Vec<R>, DeviceError> {
      use rayon::prelude::*;
      let seed = process.device_seed();
      check_path_index(m)?;
      let steps = process.grid_points();
      let rows = launch_rows(self, steps, process.euler_spec(), steps, process.fgn_spec());
      let mut out = Vec::with_capacity(m);
      // `over_chunks`, so a chunk too large for the device is retried at half
      // the size rather than failing the batch.
      crate::device::over_chunks(m, rows, |first, len| {
        // The chunk is the launch as the device holds it: on unified memory, the buffer the
        // kernel wrote, read once where it lies with no copy or allocation per path.
        let chunk = <Self as EulerKernel<$scalar>>::euler_kernel_lend(
          self,
          process,
          first,
          len,
          seed,
          |view| {
            let n = view.ncols();
            match view.as_slice() {
              // As above: the rows of a contiguous batch are chunks of one
              // slice, and rayon splits a slice without materialising it.
              Some(flat) => flat
                .par_chunks(n.max(1))
                .map(|row| f(ArrayView1::from(row)))
                .collect::<Vec<R>>(),
              None => view
                .outer_iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(&f)
                .collect::<Vec<R>>(),
            }
          },
        )?;
        out.extend(chunk);
        Ok(())
      })?;
      Ok(out)
    }

    fn try_euler_matrix<P: EulerCoefficients<$scalar>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Array2<$scalar>, DeviceError> {
      <Self as EulerKernel<$scalar>>::euler_kernel_batch(self, process, m, process.device_seed())
    }

    fn try_system_sample<const D: usize, P: EulerSystem<$scalar, D>>(
      &self,
      process: &P,
    ) -> Result<P::Output, DeviceError> {
      let seed = process.device_seed();
      let planes = <Self as EulerKernel<$scalar>>::euler_system_kernel(self, process, 0, 1, seed)?;
      Ok(P::Output::from(system_row::<$scalar, D>(&planes, 0)))
    }

    fn try_system_paths<const D: usize, P: EulerSystem<$scalar, D>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Vec<P::Output>, DeviceError> {
      let seed = process.device_seed();
      let planes = <Self as EulerKernel<$scalar>>::euler_system_batch(self, process, m, seed)?;
      Ok(
        (0..m)
          .map(|row| P::Output::from(system_row::<$scalar, D>(&planes, row)))
          .collect(),
      )
    }

    fn try_system_paths_map<const D: usize, P: EulerSystem<$scalar, D>, R: Send>(
      &self,
      process: &P,
      m: usize,
      f: impl Fn(&P::Output) -> R + Sync,
    ) -> Result<Vec<R>, DeviceError> {
      use rayon::prelude::*;
      let seed = process.device_seed();
      check_path_index(m)?;
      let (_, rows) = system_rows(self, process);
      let mut out = Vec::with_capacity(m);
      // `over_chunks`, so a chunk too large for the device is retried at half
      // the size rather than failing the batch.
      crate::device::over_chunks(m, rows, |first, len| {
        let planes =
          <Self as EulerKernel<$scalar>>::euler_system_kernel(self, process, first, len, seed)?;
        let chunk: Vec<P::Output> = (0..len)
          .map(|row| P::Output::from(system_row::<$scalar, D>(&planes, row)))
          .collect();
        out.extend(chunk.into_par_iter().map(|o| f(&o)).collect::<Vec<R>>());
        Ok(())
      })?;
      Ok(out)
    }
    }
  };
}

#[cfg(metal_backend)]
kernel_euler_backend!(crate::device::Metal, [] f32);
#[cfg(feature = "cuda")]
kernel_euler_backend!(crate::device::Cuda, [T: FloatExt] T);
