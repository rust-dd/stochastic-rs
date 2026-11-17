use std::any::TypeId;

use super::*;

/// One chunk's launch without the copy back: the caller owns the stream.
#[allow(clippy::too_many_arguments)]
fn launch_chunk<R>(
  stream: &Arc<CudaStream>,
  func: &CudaFunction,
  params: [R; crate::euler::PARAM_SLOTS],
  x0: [R; 4],
  dt: R,
  family: u32,
  components: u32,
  noises: u32,
  key: [u32; 2],
  first: usize,
  n: usize,
  m: usize,
  increments: Option<(&CudaSlice<R>, u32)>,
  curve: &[R],
  n_curves: u32,
  jump_lambda: R,
  use_jumps: u32,
  jump_law: u32,
  jump_a: R,
  jump_b: R,
  jump_c: R,
  step_first: u32,
  gamma_law: u32,
  g1_shape: R,
  g1_scale: R,
  g1_per: R,
  g2_shape: R,
  g2_scale: R,
  g2_per: R,
  lift: [&[R]; 3],
  has_lift: u32,
  lift_n: u32,
  lift_db: R,
  lift_fb: R,
  lift_x0: R,
  hist_slot: u32,
  series_n: u32,
  series_live: u32,
  table_n: u32,
  table_u0: R,
  program: &[R],
  program_n: u32,
  d_out: &mut CudaSlice<R>,
) -> Result<()>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + CachedOut,
{
  let use_incs = increments.map_or(0, |(_, streams)| streams);
  let (steps, paths, first_path) = super::launch_scalars(n, first, m, use_incs)?;
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| driver_error("htod params", e))?;
  let sqrt_dt = dt.sqrt();
  // The kernel always binds the increment pointer; an unused slot gets one
  // element rather than a null. A supplied slice was written on this device by
  // the fGN pipeline and is bound where it lies.
  let owned;
  let d_incs = match increments {
    Some((slice, _)) => slice,
    None => {
      owned = stream
        .alloc_zeros::<R>(1)
        .map_err(|e| driver_error("alloc incs", e))?;
      &owned
    }
  };
  // The kernel always binds the curve pointer; an unused slot gets one
  // element rather than a null.
  let use_curve = n_curves;
  // The three lift tables; an unused table gets one element rather than a
  // null, as the increment and curve pointers do.
  let mut d_lift = Vec::with_capacity(3);
  for table in lift {
    d_lift.push(if table.is_empty() {
      stream
        .alloc_zeros::<R>(1)
        .map_err(|e| driver_error("alloc lift", e))?
    } else {
      stream
        .clone_htod(table)
        .map_err(|e| driver_error("htod lift", e))?
    });
  }
  let d_curve = if curve.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| driver_error("alloc curve", e))?
  } else {
    stream
      .clone_htod(curve)
      .map_err(|e| driver_error("htod curve", e))?
  };
  let d_program = stream
    .clone_htod(program)
    .map_err(|e| driver_error("htod program", e))?;
  unsafe {
    stream
      .launch_builder(func)
      .arg(&mut *d_out)
      .arg(&d_params)
      .arg(&family)
      .arg(&components)
      .arg(&noises)
      .arg(&x0[0])
      .arg(&x0[1])
      .arg(&x0[2])
      .arg(&x0[3])
      .arg(&dt)
      .arg(&sqrt_dt)
      .arg(&key[0])
      .arg(&key[1])
      .arg(&steps)
      .arg(&paths)
      .arg(&first_path)
      .arg(d_incs)
      .arg(&use_incs)
      .arg(&d_curve)
      .arg(&use_curve)
      .arg(&jump_lambda)
      .arg(&use_jumps)
      .arg(&jump_law)
      .arg(&jump_a)
      .arg(&jump_b)
      .arg(&jump_c)
      .arg(&step_first)
      .arg(&gamma_law)
      .arg(&g1_shape)
      .arg(&g1_scale)
      .arg(&g1_per)
      .arg(&g2_shape)
      .arg(&g2_scale)
      .arg(&g2_per)
      .arg(&d_lift[0])
      .arg(&d_lift[1])
      .arg(&d_lift[2])
      .arg(&has_lift)
      .arg(&lift_n)
      .arg(&lift_db)
      .arg(&lift_fb)
      .arg(&lift_x0)
      .arg(&hist_slot)
      .arg(&series_n)
      .arg(&series_live)
      .arg(&table_n)
      .arg(&table_u0)
      .arg(&d_program)
      .arg(&program_n)
      .launch(paths_config(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  Ok(())
}

/// The whole batch through the two-stream pipeline, `rows` paths per chunk.
#[allow(clippy::too_many_arguments)]
fn pipelined<R>(
  ordinal: usize,
  real: &'static str,
  params: [R; crate::euler::PARAM_SLOTS],
  x0: [R; 4],
  dt: R,
  family: u32,
  components: u32,
  noises: u32,
  key: [u32; 2],
  n: usize,
  m: usize,
  rows: usize,
  curve: &[R],
  n_curves: u32,
  jump_lambda: R,
  use_jumps: u32,
  jump_law: u32,
  jump_a: R,
  jump_b: R,
  jump_c: R,
  step_first: u32,
  gamma_law: u32,
  g1_shape: R,
  g1_scale: R,
  g1_per: R,
  g2_shape: R,
  g2_scale: R,
  g2_per: R,
  hist_slot: u32,
  series_n: u32,
  series_live: u32,
  table_n: u32,
  table_u0: R,
  program: &[R],
  program_n: u32,
) -> Result<Vec<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + Send + Sync + CachedOut,
{
  let shape = Shape::new(
    super::families::Family::from_code(family).expect("a declared family"),
    use_jumps != 0 || jump_law != 0,
    gamma_law != 0,
  );
  let Launcher {
    context,
    streams,
    func,
  } = ensure_kernels(ordinal, shape, real)?;
  let planes = components as usize;
  let mut host = vec![R::zero(); planes * m * n];
  /// Synchronises both streams before the page-locked buffers can be freed.
  ///
  /// A chunk's copy back is enqueued and left in flight while the next chunk
  /// computes, so any `?` in the loop below unwinds with a device-to-host
  /// transfer still running. Freeing its destination then hands the driver a
  /// page it no longer owns; cudarc's own pinned slice synchronises in its
  /// `Drop` for the same reason. Declared *after* the buffers so it drops
  /// *before* them.
  struct DrainOnDrop([Arc<CudaStream>; 2]);

  impl Drop for DrainOnDrop {
    fn drop(&mut self) {
      for stream in &self.0 {
        let _ = stream.synchronize();
      }
    }
  }

  // Both staging buffers come from the cache: pinning pages is a syscall
  // that scales with the allocation, and at the default budget these are a
  // gigabyte each — paid on every pipelined batch before this.
  let staging = {
    let mut guard = KERNELS.lock();
    let kernels = lender(&mut guard, &context)?;
    [
      take_staging::<R>(kernels, 0, planes * rows * n)?,
      take_staging::<R>(kernels, 1, planes * rows * n)?,
    ]
  };
  let _drain = DrainOnDrop([streams[0].clone(), streams[1].clone()]);
  // Per slot: the device buffer kept alive until its copy has landed, and
  // the `(first, len)` rows the staging buffer holds.
  let mut in_flight: [Option<(usize, usize)>; 2] = [None, None];
  let drain =
    |slot: usize, in_flight: &mut [Option<(usize, usize)>; 2], host: &mut [R]| -> Result<()> {
      if let Some((f0, l0)) = in_flight[slot].take() {
        streams[slot]
          .synchronize()
          .map_err(|e| DeviceError::Launch(format!("sync chunk: {e}")))?;
        let src = unsafe { std::slice::from_raw_parts(staging[slot].ptr, planes * l0 * n) };
        // A chunk holds its own planes back to back; the batch holds each
        // plane whole, so the rows land one plane at a time.
        for c in 0..planes {
          let to = (c * m + f0) * n;
          host[to..to + l0 * n].copy_from_slice(&src[c * l0 * n..(c + 1) * l0 * n]);
        }
      }
      Ok(())
    };
  // The two output buffers the pipeline alternates between, sized for the
  // largest chunk and kept across calls: allocating them per chunk is what
  // made a Metal batch three times slower than its kernel.
  let mut buffers = {
    let mut guard = KERNELS.lock();
    let kernels = lender(&mut guard, &context)?;
    let need = planes * rows * n;
    [
      take_output::<R>(kernels, &streams[0], 0, need)?,
      take_output::<R>(kernels, &streams[1], 1, need)?,
    ]
  };
  let mut first = 0;
  let mut k = 0;
  while first < m {
    let len = rows.min(m - first);
    let slot = k % 2;
    drain(slot, &mut in_flight, &mut host)?;
    launch_chunk(
      &streams[slot],
      &func,
      params,
      x0,
      dt,
      family,
      components,
      noises,
      key,
      first,
      n,
      len,
      // The pipelined batch is Gaussian only: `euler_kernel_batch` routes a
      // fractional process around it, chunk by chunk through `euler_kernel`,
      // so no increment buffer ever meets this launch.
      None,
      curve,
      n_curves,
      jump_lambda,
      use_jumps,
      jump_law,
      jump_a,
      jump_b,
      jump_c,
      step_first,
      gamma_law,
      g1_shape,
      g1_scale,
      g1_per,
      g2_shape,
      g2_scale,
      g2_per,
      [&[], &[], &[]],
      0,
      0,
      R::zero(),
      R::zero(),
      R::zero(),
      hist_slot,
      series_n,
      series_live,
      table_n,
      table_u0,
      program,
      program_n,
      &mut buffers[slot],
    )?;
    let dst = unsafe { std::slice::from_raw_parts_mut(staging[slot].ptr, planes * len * n) };
    streams[slot]
      .memcpy_dtoh(&buffers[slot].slice(0..planes * len * n), dst)
      .map_err(|e| driver_error("dtoh chunk", e))?;
    in_flight[slot] = Some((first, len));
    first += len;
    k += 1;
  }
  drain(0, &mut in_flight, &mut host)?;
  drain(1, &mut in_flight, &mut host)?;
  if let Ok(kernels) = lender(&mut KERNELS.lock(), &context) {
    let [b0, b1] = buffers;
    return_output(kernels, 0, b0);
    return_output(kernels, 1, b1);
    let [s0, s1] = staging;
    return_staging(kernels, 0, s0);
    return_staging(kernels, 1, s1);
  }
  Ok(host)
}

/// The pipelined batch for an explicit specification, in the precision of `T`.
#[allow(clippy::too_many_arguments)]
pub(super) fn pipelined_paths<T: FloatExt>(
  ordinal: usize,
  spec: EulerSpec<T>,
  x0: [T; 4],
  n: usize,
  dt: T,
  m: usize,
  rows: usize,
  seed: u64,
  curves: Option<Vec<Vec<T>>>,
  jump_lambda: Option<T>,
  sizes: Option<crate::euler::JumpSizes<T>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<T>>,
  lift: Option<crate::euler::LiftSpec<'_, T>>,
  series: Option<u32>,
  table: Option<crate::euler::TableSpec<T>>,
  program: Option<crate::euler::ProgramSpec<'_>>,
) -> Result<Array3<T>> {
  let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
  let (program_t, program_n) = crate::euler::encode_programs::<T>(program.as_ref());
  let program64: Vec<f64> = program_t
    .iter()
    .map(|v| v.to_f64().unwrap_or(0.0))
    .collect();
  let program32: Vec<f32> = program64.iter().map(|v| *v as f32).collect();
  debug_assert!(lift.is_none(), "the pipelined batch carries no lift");
  let (family, params) = spec.encode();
  let hist_slot = crate::euler::history_slot(family, n);
  let series_n = crate::euler::series_terms(family, n, series);
  let series_live = crate::euler::series_live(family);
  let (table_n, table_u0) = crate::euler::table_terms(family, table);
  let arity = super::families::Family::from_code(family).expect("a declared family");
  let use_jumps = u32::from(jump_lambda.is_some());
  let lambda64 = jump_lambda.map_or(0.0, |v| v.to_f64().unwrap_or(0.0));
  let (jump_law, ja, jb, jc) = sizes.map_or((0, 0.0, 0.0, 0.0), |s| {
    let (law, a, b, c) = s.encode();
    (
      law,
      a.to_f64().unwrap_or(0.0),
      b.to_f64().unwrap_or(0.0),
      c.to_f64().unwrap_or(0.0),
    )
  });
  let step_first = u32::from(step_first);
  let (gamma_law, gs1, gc1, gp1, gs2, gc2, gp2) =
    gammas.map_or((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |g| {
      let (law, s1, c1, p1, s2, c2, p2) = g.encode();
      let f = |v: T| v.to_f64().unwrap_or(0.0);
      (law, f(s1), f(c1), f(p1), f(s2), f(c2), f(p2))
    });
  let (components, noises) = (arity.components() as u32, arity.noises() as u32);
  let planes = components as usize;
  let dt = dt.to_f64().unwrap_or(0.0);
  let key = super::philox::key(seed);
  let p64: [f64; crate::euler::PARAM_SLOTS] =
    std::array::from_fn(|i| params[i].to_f64().unwrap_or(0.0));
  if TypeId::of::<T>() == TypeId::of::<f64>() {
    let curve64: Vec<f64> = curve.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let data = pipelined::<f64>(
      ordinal,
      "double",
      p64,
      std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0)),
      dt,
      family,
      components,
      noises,
      key,
      n,
      m,
      rows,
      &curve64,
      n_curves,
      lambda64,
      use_jumps,
      jump_law,
      ja,
      jb,
      jc,
      step_first,
      gamma_law,
      gs1,
      gc1,
      gp1,
      gs2,
      gc2,
      gp2,
      hist_slot,
      series_n,
      series_live,
      table_n,
      table_u0.to_f64().unwrap_or(0.0),
      &program64,
      program_n,
    )?;
    let out = Array3::<f64>::from_shape_vec((planes, m, n), data)
      .expect("the kernel returns components * m * n values");
    return Ok(unsafe { std::mem::transmute::<Array3<f64>, Array3<T>>(out) });
  }
  assert!(
    TypeId::of::<T>() == TypeId::of::<f32>(),
    "FloatExt is implemented for f32 and f64 only"
  );
  let p32: [f32; crate::euler::PARAM_SLOTS] = std::array::from_fn(|i| p64[i] as f32);
  let curve32: Vec<f32> = curve.iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
  let data = pipelined::<f32>(
    ordinal,
    "float",
    p32,
    std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0) as f32),
    dt as f32,
    family,
    components,
    noises,
    key,
    n,
    m,
    rows,
    &curve32,
    n_curves,
    lambda64 as f32,
    use_jumps,
    jump_law,
    ja as f32,
    jb as f32,
    jc as f32,
    step_first,
    gamma_law,
    gs1 as f32,
    gc1 as f32,
    gp1 as f32,
    gs2 as f32,
    gc2 as f32,
    gp2 as f32,
    hist_slot,
    series_n,
    series_live,
    table_n,
    table_u0.to_f64().unwrap_or(0.0) as f32,
    &program32,
    program_n,
  )?;
  let out = Array3::<f32>::from_shape_vec((planes, m, n), data)
    .expect("the kernel returns components * m * n values");
  Ok(unsafe { std::mem::transmute::<Array3<f32>, Array3<T>>(out) })
}
