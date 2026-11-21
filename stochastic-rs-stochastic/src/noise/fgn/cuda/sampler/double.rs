use cudarc::cufft;

use super::*;

/// The double-precision pipeline with the increments left on the device, as
/// in [`sample_f32_device`].
pub(crate) fn sample_f64_device(
  sqrt_eigs: &[f64],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u64,
  first: usize,
  ordinal: usize,
) -> Result<CudaSlice<f64>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f64).powf(-hurst) * t.powf(hurst);
  let args = LaunchArgs::new(n, m, out_size)?;

  let GpuKernels {
    stream,
    gen_scale_f64: gen_scale,
    extract_f64: extract,
    ..
  } = get_or_init_gpu(ordinal)?;
  let mut sized = SIZED_F64.lock();
  let s = crate::device::lru_slot(
    &mut sized,
    |s| {
      Arc::ptr_eq(s.d_out.stream(), &stream)
        && s.n == n
        && s.m == m
        && s.offset == offset
        && s.hurst_bits == hurst_bits
        && s.t_bits == t_bits
    },
    || {
      let plan = cufft::result::plan_1d(args.traj, cufft::sys::cufftType::CUFFT_Z2Z, args.batch)
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
        host_pinned: None,
        n,
        m,
        offset,
        hurst_bits,
        t_bits,
      })
    },
  )?;

  // 1. Fused generate + scale. One seed per batch; chunks continue the element count, so a
  // chunked batch equals one launch element for element.
  let [seed_lo, seed_hi] = crate::euler::philox::key(seed);
  let first_cell = (first * traj_size) as u64;
  unsafe {
    stream
      .launch_builder(&gen_scale)
      .arg(&mut s.d_data)
      .arg(&s.d_eigs)
      .arg(&args.traj)
      .arg(&args.total)
      .arg(&seed_lo)
      .arg(&seed_hi)
      .arg(&first_cell)
      .launch(LaunchConfig::for_num_elems(args.grid))
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
  unsafe {
    stream
      .launch_builder(&extract)
      .arg(&s.d_data)
      .arg(&mut s.d_out)
      .arg(&args.out_size)
      .arg(&args.traj)
      .arg(&scale)
      .arg(&args.total_out)
      .launch(LaunchConfig::for_num_elems(args.grid_out))
      .map_err(|e| DeviceError::Launch(format!("extract: {e}")))?;
  }

  // 4. Hand the output over where it lies, as in the single-precision path.
  let out = stream
    .clone_dtod(&s.d_out)
    .map_err(|e| DeviceError::Launch(format!("dtod: {e}")))?;
  drop(sized);
  Ok(out)
}

pub(super) fn sample_f64<T: FloatExt>(
  sqrt_eigs: &[f64],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u64,
  first: usize,
  ordinal: usize,
) -> Result<Array2<T>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f64).powf(-hurst) * t.powf(hurst);
  let args = LaunchArgs::new(n, m, out_size)?;

  let GpuKernels {
    stream,
    gen_scale_f64: gen_scale,
    extract_f64: extract,
    ..
  } = get_or_init_gpu(ordinal)?;
  let mut sized = SIZED_F64.lock();
  let s = crate::device::lru_slot(
    &mut sized,
    |s| {
      Arc::ptr_eq(s.d_out.stream(), &stream)
        && s.n == n
        && s.m == m
        && s.offset == offset
        && s.hurst_bits == hurst_bits
        && s.t_bits == t_bits
    },
    || {
      let plan = cufft::result::plan_1d(args.traj, cufft::sys::cufftType::CUFFT_Z2Z, args.batch)
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
        host_pinned: None,
        n,
        m,
        offset,
        hurst_bits,
        t_bits,
      })
    },
  )?;

  // 1. Fused generate + scale. One seed per batch; chunks continue the element count, so a
  // chunked batch equals one launch element for element.
  let [seed_lo, seed_hi] = crate::euler::philox::key(seed);
  let first_cell = (first * traj_size) as u64;
  unsafe {
    stream
      .launch_builder(&gen_scale)
      .arg(&mut s.d_data)
      .arg(&s.d_eigs)
      .arg(&args.traj)
      .arg(&args.total)
      .arg(&seed_lo)
      .arg(&seed_hi)
      .arg(&first_cell)
      .launch(LaunchConfig::for_num_elems(args.grid))
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
  unsafe {
    stream
      .launch_builder(&extract)
      .arg(&s.d_data)
      .arg(&mut s.d_out)
      .arg(&args.out_size)
      .arg(&args.traj)
      .arg(&scale)
      .arg(&args.total_out)
      .launch(LaunchConfig::for_num_elems(args.grid_out))
      .map_err(|e| DeviceError::Launch(format!("extract: {e}")))?;
  }

  // 4. DtoH. Large outputs pre-fault the host buffer while the kernels run, then DMA through the
  // cached pinned staging and copy in parallel; small outputs go direct.
  let len = m * out_size;
  let host = match (len * std::mem::size_of::<f64>() >= STAGING_MIN_BYTES)
    .then(|| alloc_prefaulted::<f64>(len))
  {
    Some(mut host) => {
      drain_into(
        &stream,
        &s.d_out,
        staging(&mut s.host_pinned, len)?,
        &mut host,
      )?;
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
