use std::any::TypeId;
use std::sync::Arc;

use cudarc::cufft;
use cudarc::driver::*;
use ndarray::Array2;
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;

use super::super::Fgn;
use super::convert::array2_from_vec_f32;
use super::convert::array2_from_vec_f64;
use super::state::CUFFT_FORWARD;
use super::state::GPU;
use super::state::PinnedHost;
use super::state::SIZED_F32;
use super::state::SIZED_F64;
use super::state::SizedCtxF32;
use super::state::SizedCtxF64;
use super::state::get_or_init_gpu;
use crate::device::DeviceError;
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

/// Output transfers at or above this byte size use the pinned-staging +
/// parallel-copy path; smaller ones go direct via `clone_dtoh`, where the
/// staging round-trip and rayon overhead aren't worth it.
const STAGING_MIN_BYTES: usize = 32 << 20;

/// Philox counter base for one launch, a pure function of the launch's seed.
///
/// The kernel keys Philox with the low 32 bits of `seed` and counts from
/// `tid + seq`; deriving `seq` from the seed (SplitMix64 finaliser) makes
/// every draw a function of the seed alone, so two `Fgn`s built from the
/// same `Deterministic` seed produce the same paths no matter what other
/// device draws the process made in between. A process-global counter used
/// to sit here and made CUDA fGN paths depend on that history.
fn counter_offset(seed: u64) -> u64 {
  let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^ (z >> 31)
}

/// Allocates an output `Vec<T>` and pre-faults its pages in parallel.
///
/// Call this right after the GPU kernels are launched: they run asynchronously,
/// so faulting the fresh allocation here overlaps device compute instead of
/// serialising as part of the post-transfer copy. The fault cost (not the PCIe
/// transfer) is what dominates materialising a multi-hundred-MB result on the
/// host, so hiding it under compute is the main lever left.
fn alloc_prefaulted<T: Copy>(len: usize) -> Vec<T> {
  let mut v = Vec::<T>::with_capacity(len);
  #[allow(clippy::uninit_vec)]
  unsafe {
    v.set_len(len)
  };
  let bytes = len * std::mem::size_of::<T>();
  let raw = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, bytes) };
  const CHUNK: usize = 1 << 22;
  raw.par_chunks_mut(CHUNK).for_each(|c| c.fill(0));
  v
}

/// DMAs a device buffer through the cached page-locked staging buffer at full
/// PCIe bandwidth, then copies into `host` in parallel. `host` must already be
/// allocated and pre-faulted (see [`alloc_prefaulted`]) so the copy is pure
/// bandwidth with no first-touch faults.
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

fn sample_f32<T: FloatExt>(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u64,
  first: usize,
) -> Result<Array2<T>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);

  get_or_init_gpu()?;
  // Clone the handles out of the global lock so another size can launch
  // concurrently; the per-size state below keeps its own lock for its buffers.
  let (stream, gen_scale, extract) = {
    let g = GPU.lock();
    let k = g.as_ref().unwrap();
    (
      k.stream.clone(),
      k.gen_scale_f32.clone(),
      k.extract_f32.clone(),
    )
  };
  let mut sized = SIZED_F32.lock();
  let s = crate::device::lru_slot(
    &mut sized,
    |s| {
      s.n == n && s.m == m && s.offset == offset && s.hurst_bits == hurst_bits && s.t_bits == t_bits
    },
    || {
      let plan =
        cufft::result::plan_1d(traj_size as i32, cufft::sys::cufftType::CUFFT_C2C, m as i32)
          .map_err(|e| DeviceError::Launch(format!("cuFFT plan: {e}")))?;
      unsafe {
        cufft::result::set_stream(plan, stream.cu_stream() as _)
          .map_err(|e| DeviceError::Launch(format!("cuFFT set_stream: {e}")))?;
      }
      Ok(SizedCtxF32 {
        fft_plan: plan,
        d_eigs: stream
          .clone_htod(sqrt_eigs)
          .map_err(|e| DeviceError::Launch(format!("htod eigs: {e}")))?,
        d_data: stream
          .alloc_zeros::<f32>(2 * m * traj_size)
          .map_err(|e| DeviceError::Launch(format!("alloc data: {e}")))?,
        d_out: stream
          .alloc_zeros::<f32>(m * out_size)
          .map_err(|e| DeviceError::Launch(format!("alloc out: {e}")))?,
        host_pinned: PinnedHost::<f32>::alloc(m * out_size)?,
        n,
        m,
        offset,
        hurst_bits,
        t_bits,
      })
    },
  )?;
  let profile = std::env::var("STOCHASTIC_RS_CUDA_PROFILE").is_ok();
  let tstart = std::time::Instant::now();

  // 1. Fused generate normals + scale by eigenvalues
  let total_complex = (m * traj_size) as i32;
  let traj_i32 = traj_size as i32;
  // One seed per batch; chunks continue the element count, so a batch
  // produced in chunks equals one launch element for element.
  let seq = counter_offset(seed) + (first * traj_size) as u64;
  unsafe {
    stream
      .launch_builder(&gen_scale)
      .arg(&mut s.d_data)
      .arg(&s.d_eigs)
      .arg(&traj_i32)
      .arg(&total_complex)
      .arg(&seed)
      .arg(&seq)
      .launch(LaunchConfig::for_num_elems(total_complex as u32))
      .map_err(|e| DeviceError::Launch(format!("gen_scale: {e}")))?;
  }

  // 2. Batched FFT
  {
    let (ptr, _g) = s.d_data.device_ptr_mut(&stream);
    unsafe {
      cufft::result::exec_c2c(s.fft_plan, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
        .map_err(|e| DeviceError::Launch(format!("cuFFT: {e}")))?;
    }
  }

  // 3. Extract real parts + scale
  let total_out = (m * out_size) as i32;
  let out_i32 = out_size as i32;
  let stride_i32 = traj_size as i32;
  unsafe {
    stream
      .launch_builder(&extract)
      .arg(&s.d_data)
      .arg(&mut s.d_out)
      .arg(&out_i32)
      .arg(&stride_i32)
      .arg(&scale)
      .arg(&total_out)
      .launch(LaunchConfig::for_num_elems(total_out as u32))
      .map_err(|e| DeviceError::Launch(format!("extract: {e}")))?;
  }
  // 4. DtoH. For large outputs, allocate + pre-fault the host buffer now while
  // the async kernels above are still running, so the page faults overlap device
  // compute; then DMA through the cached pinned staging and copy in parallel.
  // Small outputs go direct.
  let len = m * out_size;
  let prefaulted =
    (len * std::mem::size_of::<f32>() >= STAGING_MIN_BYTES).then(|| alloc_prefaulted::<f32>(len));

  let t_compute = if profile {
    stream.synchronize().ok();
    tstart.elapsed()
  } else {
    std::time::Duration::ZERO
  };

  let host = match prefaulted {
    Some(mut host) => {
      drain_into(&stream, &s.d_out, &s.host_pinned, &mut host)?;
      host
    }
    None => stream
      .clone_dtoh(&s.d_out)
      .map_err(|e| DeviceError::Launch(format!("dtoh: {e}")))?,
  };
  let t_dtoh = if profile {
    tstart.elapsed()
  } else {
    std::time::Duration::ZERO
  };
  drop(sized);

  let fgn = array2_from_vec_f32::<T>(host, m, out_size);
  if profile {
    eprintln!(
      "CUDAPROF f32 n={n} m={m} compute={:.2?} dtoh={:.2?} total={:.2?}",
      t_compute,
      t_dtoh.saturating_sub(t_compute),
      tstart.elapsed()
    );
  }
  Ok(fgn)
}

fn sample_f64<T: FloatExt>(
  sqrt_eigs: &[f64],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u64,
  first: usize,
) -> Result<Array2<T>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f64).powf(-hurst) * t.powf(hurst);

  get_or_init_gpu()?;
  // Clone the handles out of the global lock so another size can launch
  // concurrently; the per-size state below keeps its own lock for its buffers.
  let (stream, gen_scale, extract) = {
    let g = GPU.lock();
    let k = g.as_ref().unwrap();
    (
      k.stream.clone(),
      k.gen_scale_f64.clone(),
      k.extract_f64.clone(),
    )
  };
  let mut sized = SIZED_F64.lock();
  let s = crate::device::lru_slot(
    &mut sized,
    |s| {
      s.n == n && s.m == m && s.offset == offset && s.hurst_bits == hurst_bits && s.t_bits == t_bits
    },
    || {
      let plan =
        cufft::result::plan_1d(traj_size as i32, cufft::sys::cufftType::CUFFT_Z2Z, m as i32)
          .map_err(|e| DeviceError::Launch(format!("cuFFT plan: {e}")))?;
      unsafe {
        cufft::result::set_stream(plan, stream.cu_stream() as _)
          .map_err(|e| DeviceError::Launch(format!("cuFFT set_stream: {e}")))?;
      }
      Ok(SizedCtxF64 {
        fft_plan: plan,
        d_eigs: stream
          .clone_htod(sqrt_eigs)
          .map_err(|e| DeviceError::Launch(format!("htod eigs: {e}")))?,
        d_data: stream
          .alloc_zeros::<f64>(2 * m * traj_size)
          .map_err(|e| DeviceError::Launch(format!("alloc data: {e}")))?,
        d_out: stream
          .alloc_zeros::<f64>(m * out_size)
          .map_err(|e| DeviceError::Launch(format!("alloc out: {e}")))?,
        host_pinned: PinnedHost::<f64>::alloc(m * out_size)?,
        n,
        m,
        offset,
        hurst_bits,
        t_bits,
      })
    },
  )?;

  // 1. Fused generate + scale
  let total_complex = (m * traj_size) as i32;
  let traj_i32 = traj_size as i32;
  // One seed per batch; chunks continue the element count, so a batch
  // produced in chunks equals one launch element for element.
  let seq = counter_offset(seed) + (first * traj_size) as u64;
  unsafe {
    stream
      .launch_builder(&gen_scale)
      .arg(&mut s.d_data)
      .arg(&s.d_eigs)
      .arg(&traj_i32)
      .arg(&total_complex)
      .arg(&seed)
      .arg(&seq)
      .launch(LaunchConfig::for_num_elems(total_complex as u32))
      .map_err(|e| DeviceError::Launch(format!("gen_scale: {e}")))?;
  }

  // 2. Batched FFT
  {
    let (ptr, _g) = s.d_data.device_ptr_mut(&stream);
    unsafe {
      cufft::result::exec_z2z(s.fft_plan, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
        .map_err(|e| DeviceError::Launch(format!("cuFFT: {e}")))?;
    }
  }

  // 3. Extract + scale
  let total_out = (m * out_size) as i32;
  let out_i32 = out_size as i32;
  let stride_i32 = traj_size as i32;
  unsafe {
    stream
      .launch_builder(&extract)
      .arg(&s.d_data)
      .arg(&mut s.d_out)
      .arg(&out_i32)
      .arg(&stride_i32)
      .arg(&scale)
      .arg(&total_out)
      .launch(LaunchConfig::for_num_elems(total_out as u32))
      .map_err(|e| DeviceError::Launch(format!("extract: {e}")))?;
  }

  // 4. DtoH. Pre-fault the host buffer (large outputs) while the async kernels
  // run, then DMA through the cached pinned staging and copy in parallel; small
  // outputs go direct.
  let len = m * out_size;
  let host = match (len * std::mem::size_of::<f64>() >= STAGING_MIN_BYTES)
    .then(|| alloc_prefaulted::<f64>(len))
  {
    Some(mut host) => {
      drain_into(&stream, &s.d_out, &s.host_pinned, &mut host)?;
      host
    }
    None => stream
      .clone_dtoh(&s.d_out)
      .map_err(|e| DeviceError::Launch(format!("dtoh: {e}")))?,
  };
  drop(sized);

  let fgn = array2_from_vec_f64::<T>(host, m, out_size);
  Ok(fgn)
}

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// `m` paths on the selected CUDA device, in chunks that fit the batch
  /// budget: one seed for the whole batch and a running element offset, so
  /// the result is the same whatever the budget.
  pub(crate) fn sample_cuda_native_impl(&self, m: usize) -> Result<Array2<T>> {
    let n = self.n;
    let offset = self.offset;
    let out_size = n - offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let seed: u64 = rand::Rng::random(&mut self.seed.rng());
    // Per path: 2 * traj_size complex scalars of work buffer plus the output row.
    let rows = crate::device::chunk_rows(4 * n + out_size, std::mem::size_of::<T>());
    let mut out = Array2::<T>::zeros((m, out_size));
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = if TypeId::of::<T>() == TypeId::of::<f32>() {
        let eigs: Vec<f32> = self
          .sqrt_eigenvalues
          .iter()
          .map(|x| x.to_f32().unwrap())
          .collect();
        sample_f32::<T>(&eigs, n, len, offset, hurst, t, seed, first)?
      } else {
        // `f64` is what the type says, so a failing double-precision launch is
        // reported, never quietly replaced by the `f32` kernel.
        let eigs: Vec<f64> = self
          .sqrt_eigenvalues
          .iter()
          .map(|x| x.to_f64().unwrap())
          .collect();
        sample_f64::<T>(&eigs, n, len, offset, hurst, t, seed, first)?
      };
      out
        .slice_mut(ndarray::s![first..first + len, ..])
        .assign(&chunk);
      first += len;
    }
    Ok(out)
  }
}
