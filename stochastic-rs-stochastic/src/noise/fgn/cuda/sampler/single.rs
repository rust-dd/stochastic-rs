use cudarc::cufft;

use super::*;

/// The pipeline with the increments left on the device, where the Euler engine reads them without
/// a host round trip; the returned slice is a device copy, so the per-size cache stays reusable.
pub(crate) fn sample_f32_device(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u64,
  first: usize,
  ordinal: usize,
) -> Result<CudaSlice<f32>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let args = LaunchArgs::new(n, m, out_size)?;

  let GpuKernels {
    stream,
    gen_scale_f32: gen_scale,
    extract_f32: extract,
    ..
  } = get_or_init_gpu(ordinal)?;
  let mut sized = SIZED_F32.lock();
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
      let plan = cufft::result::plan_1d(args.traj, cufft::sys::cufftType::CUFFT_C2C, args.batch)
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
        host_pinned: None,
        n,
        m,
        offset,
        hurst_bits,
        t_bits,
      })
    },
  )?;

  // 1. Fused generate normals + scale by eigenvalues. One seed per batch; chunks continue the
  // element count, so a chunked batch equals one launch element for element.
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
      cufft::result::exec_c2c(s.fft_plan, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
        .map_err(|e| DeviceError::Launch(format!("cuFFT: {e}")))?;
    }
  }

  // 3. Extract real parts + scale
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
  // 4. Hand the output over where it lies. A device-to-device copy keeps the
  // per-size cache reusable and costs a fraction of a host round trip.
  let out = stream
    .clone_dtod(&s.d_out)
    .map_err(|e| DeviceError::Launch(format!("dtod: {e}")))?;
  drop(sized);
  Ok(out)
}

pub(super) fn sample_f32<T: FloatExt>(
  sqrt_eigs: &[f32],
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
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let args = LaunchArgs::new(n, m, out_size)?;

  let GpuKernels {
    stream,
    gen_scale_f32: gen_scale,
    extract_f32: extract,
    ..
  } = get_or_init_gpu(ordinal)?;
  let mut sized = SIZED_F32.lock();
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
      let plan = cufft::result::plan_1d(args.traj, cufft::sys::cufftType::CUFFT_C2C, args.batch)
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
        host_pinned: None,
        n,
        m,
        offset,
        hurst_bits,
        t_bits,
      })
    },
  )?;
  let profile = log::log_enabled!(log::Level::Trace);
  let tstart = std::time::Instant::now();

  // 1. Fused generate normals + scale by eigenvalues. One seed per batch; chunks continue the
  // element count, so a chunked batch equals one launch element for element.
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
      cufft::result::exec_c2c(s.fft_plan, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
        .map_err(|e| DeviceError::Launch(format!("cuFFT: {e}")))?;
    }
  }

  // 3. Extract real parts + scale
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
  let t_dtoh = if profile {
    tstart.elapsed()
  } else {
    std::time::Duration::ZERO
  };
  drop(sized);

  let fgn = array2_from_vec_f32::<T>(host, m, out_size);
  if profile {
    log::trace!(
      "CUDAPROF f32 n={n} m={m} compute={:.2?} dtoh={:.2?} total={:.2?}",
      t_compute,
      t_dtoh.saturating_sub(t_compute),
      tstart.elapsed()
    );
  }
  Ok(fgn)
}
