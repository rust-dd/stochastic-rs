use std::any::TypeId;
use std::sync::Arc;

use cudarc::driver::*;
use ndarray::Array2;
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;

use super::super::Fgn;
use super::convert::array2_from_vec_f32;
use super::convert::array2_from_vec_f64;
use super::state::CUFFT_FORWARD;
use super::state::GpuKernels;
use super::state::PinnedHost;
use super::state::SIZED_F32;
use super::state::SIZED_F64;
use super::state::SizedCtxF32;
use super::state::SizedCtxF64;
use super::state::get_or_init_gpu;
use super::state::staging;
use crate::device::DeviceError;
use crate::device::launch_len;
use crate::traits::FloatExt;

mod double;
mod single;

use double::sample_f64;
pub(crate) use double::sample_f64_device;
use single::sample_f32;
pub(crate) use single::sample_f32_device;

type Result<T> = std::result::Result<T, DeviceError>;

/// Outputs of this byte size or more take the pinned-staging, parallel-copy path; smaller ones
/// go direct through `clone_dtoh`, where staging and rayon cost more than they save.
const STAGING_MIN_BYTES: usize = 32 << 20;

/// `rows` cut until a launch of `streams` paths a row keeps every index in the kernels' `int`: a
/// path is `2n` complex work values and the kernel writes `data[2 * tid + 1]`, so it costs `4n`.
pub(crate) fn index_rows(rows: usize, streams: usize, n: usize) -> usize {
  crate::device::rows_within_index_limit(rows, streams * 4 * n, i32::MAX as usize)
}

/// One launch's integer kernel and plan arguments, each checked against the
/// type the kernel or cuFFT takes it in rather than cast into it.
struct LaunchArgs {
  /// The transform length `2n`: the generate kernel's period, the read-out's
  /// stride and the plan's length.
  traj: i32,
  /// Rows of the launch, the plan's batch.
  batch: i32,
  /// Complex values the generate kernel writes, and the grid covering them.
  total: i32,
  grid: u32,
  /// The read-out's row length, its element count and its grid.
  out_size: i32,
  total_out: i32,
  grid_out: u32,
}

impl LaunchArgs {
  /// The arguments of a launch of `m` rows of an `n`-point embedding, each
  /// `out_size` values long once read out.
  fn new(n: usize, m: usize, out_size: usize) -> Result<Self> {
    let traj_size = 2 * n;
    // The generate kernel writes `data[2 * tid + 1]` for every `tid` below
    // `total`, so the work buffer's whole length must fit the `int` too.
    launch_len::<i32>(2 * m * traj_size, "data")?;
    Ok(Self {
      traj: launch_len(traj_size, "traj_size")?,
      batch: launch_len(m, "batch")?,
      total: launch_len(m * traj_size, "total")?,
      grid: launch_len(m * traj_size, "gen_scale grid")?,
      out_size: launch_len(out_size, "out_size")?,
      total_out: launch_len(m * out_size, "total")?,
      grid_out: launch_len(m * out_size, "extract grid")?,
    })
  }
}

/// An output `Vec<T>` with its pages pre-faulted in parallel. Call it right after the kernels
/// launch: faulting then overlaps device compute, and the fault cost dominates a large transfer.
fn alloc_prefaulted<T: ValidAsZeroBits + Copy>(len: usize) -> Vec<T> {
  let mut v = Vec::<T>::with_capacity(len);
  let spare = v.spare_capacity_mut();
  let bytes = std::mem::size_of_val(spare);
  // SAFETY: the slice covers exactly the reserved capacity, and a
  // `MaybeUninit<u8>` asks nothing of the bytes behind it.
  let raw = unsafe {
    std::slice::from_raw_parts_mut(spare.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>, bytes)
  };
  const CHUNK: usize = 1 << 22;
  raw
    .par_chunks_mut(CHUNK)
    .for_each(|c| c.fill(std::mem::MaybeUninit::new(0)));
  // SAFETY: every element up to the capacity is now zero bytes, which
  // `ValidAsZeroBits` says is a value of `T`.
  unsafe { v.set_len(len) };
  v
}

/// DMAs a device buffer through the cached page-locked staging buffer, then copies into `host`
/// in parallel; `host` must be pre-faulted (see [`alloc_prefaulted`]) so the copy takes no faults.
fn drain_into<T>(
  stream: &Arc<CudaStream>,
  d_out: &CudaSlice<T>,
  staging: &PinnedHost<T>,
  host: &mut [T],
) -> Result<()>
where
  T: Copy + Send + Sync + DeviceRepr,
{
  let len = host.len();
  debug_assert_eq!(staging.len, len, "staging buffer sized to the output");
  let dst = unsafe { std::slice::from_raw_parts_mut(staging.ptr, len) };
  stream
    .memcpy_dtoh(d_out, dst)
    .map_err(|e| DeviceError::Launch(format!("dtoh: {e}")))?;
  stream
    .synchronize()
    .map_err(|e| DeviceError::Launch(format!("sync dtoh: {e}")))?;

  let src = unsafe { std::slice::from_raw_parts(staging.ptr, len) };
  const CHUNK: usize = 1 << 18;
  host
    .par_chunks_mut(CHUNK)
    .zip(src.par_chunks(CHUNK))
    .for_each(|(o, p)| o.copy_from_slice(p));
  Ok(())
}

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// `m` paths on the selected CUDA device, in chunks that fit the batch budget: one seed and a
  /// running element offset, so the result is the same whatever the budget.
  pub(crate) fn sample_cuda_impl<S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Cuda,
  ) -> Result<Array2<T>> {
    let n = self.padded_n;
    let offset = self.offset;
    let out_size = n - offset;
    let hurst = self.hurst().to_f64().unwrap();
    let t = self.t().unwrap_or(T::one()).to_f64().unwrap();
    let seed = seed_src.next_seed();
    // Per path: 2 * traj_size complex scalars of work buffer plus the output row.
    let rows = index_rows(
      crate::device::chunk_rows(
        device.batch_budget,
        4 * n + out_size,
        std::mem::size_of::<T>(),
      ),
      1,
      n,
    );
    let mut out = Array2::<T>::zeros((m, out_size));
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = if TypeId::of::<T>() == TypeId::of::<f32>() {
        let eigs: Vec<f32> = self
          .sqrt_eigenvalues()
          .iter()
          .map(|x| x.to_f32().unwrap())
          .collect();
        sample_f32::<T>(&eigs, n, len, offset, hurst, t, seed, first, device.ordinal)?
      } else {
        // `f64` is what the type says, so a failing double-precision launch is
        // reported, never quietly replaced by the `f32` kernel.
        let eigs: Vec<f64> = self
          .sqrt_eigenvalues()
          .iter()
          .map(|x| x.to_f64().unwrap())
          .collect();
        sample_f64::<T>(&eigs, n, len, offset, hurst, t, seed, first, device.ordinal)?
      };
      out
        .slice_mut(ndarray::s![first..first + len, ..])
        .assign(&chunk);
      first += len;
    }
    Ok(out)
  }
}
