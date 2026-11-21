use std::any::TypeId;

use super::*;

/// One launch, with `finish` run over the values it wrote: the page-locked landing buffer is
/// lent for the call, so a caller that only reads the batch never pays a copy of it.
fn run<R, O>(
  ordinal: usize,
  real: &'static str,
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
  finish: impl FnOnce(&[R]) -> O,
) -> Result<O>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + CachedOut,
{
  let shape = Shape::new(
    super::families::Family::from_code(family).expect("a declared family"),
    use_jumps != 0 || jump_law != 0,
    gamma_law != 0,
  );
  let use_incs = increments.map_or(0, |(_, streams)| streams);
  let (steps, paths, first_path) = super::launch_scalars(n, first, m, use_incs)?;
  let Launcher {
    context,
    streams: [stream, _],
    func,
  } = ensure_kernels(ordinal, shape, real)?;
  let out_len = components as usize * m * n;
  let mut d_out = {
    let mut guard = KERNELS.lock();
    take_output::<R>(lender(&mut guard, &context)?, &stream, 0, out_len)?
  };
  let stream = &stream;
  let func = &func;
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| driver_error("htod params", e))?;
  let sqrt_dt = dt.sqrt();
  // The kernel always binds the increment pointer, so an unused slot gets one element, not a
  // null. A supplied slice was written on this device by the fGN pipeline and is bound in place.
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
      .arg(&mut d_out)
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
  // Only this launch's rows: an earlier launch may have grown the buffer. The copy goes through
  // page-locked memory; `clone_dtoh` would allocate per launch and copy into pageable pages.
  let mut staging = {
    let mut guard = KERNELS.lock();
    take_staging::<R>(lender(&mut guard, &context)?, 0, out_len)?
  };
  stream
    .memcpy_dtoh(&d_out.slice(0..out_len), unsafe {
      staging.as_mut_slice(out_len)
    })
    .map_err(|e| driver_error("dtoh", e))?;
  // A copy into page-locked memory is asynchronous, so the buffer is only the
  // launch's output once the stream says so.
  stream
    .synchronize()
    .map_err(|e| driver_error("dtoh sync", e))?;
  let out = finish(unsafe { staging.as_slice(out_len) });
  // Only the cache that lent the buffers takes them back; after a release or rebind they drop here.
  if let Ok(kernels) = lender(&mut KERNELS.lock(), &context) {
    return_output(kernels, 0, d_out);
    return_staging(kernels, 0, staging);
  }
  Ok(out)
}

/// The kernel launch for an explicit specification, with `finish` run over
/// the `components × m × n` values it wrote, in the caller's precision.
pub(super) fn device_paths<T: FloatExt, O>(
  ordinal: usize,
  spec: EulerSpec<T>,
  x0: [T; 4],
  n: usize,
  dt: T,
  first: usize,
  m: usize,
  seed: u64,
  fgn: Option<crate::euler::FgnSpec<'_, T>>,
  curves: Option<Vec<Vec<T>>>,
  jump_lambda: Option<T>,
  sizes: Option<crate::euler::JumpSizes<T>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<T>>,
  lift: Option<crate::euler::LiftSpec<'_, T>>,
  series: Option<u32>,
  table: Option<crate::euler::TableSpec<T>>,
  program: Option<crate::euler::ProgramSpec<'_>>,
  finish: impl FnOnce(&[T], usize) -> O,
) -> Result<O> {
  {
    let (family, params) = spec.encode();
    let arity = super::families::Family::from_code(family).expect("a declared family");
    let (components, noises) = (arity.components() as u32, arity.noises() as u32);
    if n == 0 || m == 0 {
      return Ok(finish(&[], components as usize));
    }
    let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
    let (program_t, program_n) = crate::euler::encode_programs::<T>(program.as_ref());
    let program64: Vec<f64> = program_t
      .iter()
      .map(|v| v.to_f64().unwrap_or(0.0))
      .collect();
    let program32: Vec<f32> = program64.iter().map(|v| *v as f32).collect();
    let (lift_tables, has_lift, lift_n, lift_db, lift_fb, lift_x0) =
      crate::euler::encode_lift(lift.as_ref());
    let lift_tables64: Vec<Vec<f64>> = lift_tables
      .iter()
      .map(|t| t.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect())
      .collect();
    let (lift_db, lift_fb, lift_x0) = (
      lift_db.to_f64().unwrap_or(0.0),
      lift_fb.to_f64().unwrap_or(0.0),
      lift_x0.to_f64().unwrap_or(0.0),
    );
    let hist_slot = crate::euler::history_slot(family, n);
    let series_n = crate::euler::series_terms(family, n, series);
    let series_live = crate::euler::series_live(family);
    let (table_n, table_u0) = crate::euler::table_terms(family, table);
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
    let dt = dt.to_f64().unwrap_or(0.0);
    let key = super::philox::key(seed);
    let p64: [f64; crate::euler::PARAM_SLOTS] =
      std::array::from_fn(|i| params[i].to_f64().unwrap_or(0.0));
    let streams = fgn.as_ref().map_or(1, |spec| spec.streams) as u32;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
      let incs = match fgn.as_ref() {
        Some(spec) => {
          let eigs: Vec<f64> = spec
            .sqrt_eigenvalues
            .iter()
            .map(|v| v.to_f64().unwrap_or(0.0))
            .collect();
          Some(crate::noise::fgn::cuda::sampler::sample_f64_device(
            &eigs,
            spec.n,
            spec.streams * m,
            spec.offset,
            spec.hurst,
            spec.t,
            seed,
            spec.streams * first,
            ordinal,
          )?)
        }
        None => None,
      };
      let curve64: Vec<f64> = curve.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
      return run::<f64, O>(
        ordinal,
        "double",
        p64,
        std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0)),
        dt,
        family,
        components,
        noises,
        key,
        first,
        n,
        m,
        incs.as_ref().map(|slice| (slice, streams)),
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
        [&lift_tables64[0], &lift_tables64[1], &lift_tables64[2]],
        has_lift,
        lift_n,
        lift_db,
        lift_fb,
        lift_x0,
        hist_slot,
        series_n,
        series_live,
        table_n,
        table_u0.to_f64().unwrap_or(0.0),
        &program64,
        program_n,
        // The branch this is in establishes that `T` is `f64`, so the values
        // are already in the caller's precision.
        |data| {
          finish(
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len()) },
            components as usize,
          )
        },
      );
    }
    let p32: [f32; crate::euler::PARAM_SLOTS] = std::array::from_fn(|i| p64[i] as f32);
    let lift_tables32: Vec<Vec<f32>> = lift_tables64
      .iter()
      .map(|t| t.iter().map(|v| *v as f32).collect())
      .collect();
    let incs = match fgn.as_ref() {
      Some(spec) => {
        let eigs: Vec<f32> = spec
          .sqrt_eigenvalues
          .iter()
          .map(|v| v.to_f32().unwrap_or(0.0))
          .collect();
        Some(crate::noise::fgn::cuda::sampler::sample_f32_device(
          &eigs,
          spec.n,
          spec.streams * m,
          spec.offset,
          spec.hurst,
          spec.t,
          seed,
          spec.streams * first,
          ordinal,
        )?)
      }
      None => None,
    };
    let curve32: Vec<f32> = curve.iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
    assert!(
      TypeId::of::<T>() == TypeId::of::<f32>(),
      "FloatExt is implemented for f32 and f64 only"
    );
    run::<f32, O>(
      ordinal,
      "float",
      p32,
      std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0) as f32),
      dt as f32,
      family,
      components,
      noises,
      key,
      first,
      n,
      m,
      incs.as_ref().map(|slice| (slice, streams)),
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
      [&lift_tables32[0], &lift_tables32[1], &lift_tables32[2]],
      has_lift,
      lift_n,
      lift_db as f32,
      lift_fb as f32,
      lift_x0 as f32,
      hist_slot,
      series_n,
      series_live,
      table_n,
      table_u0.to_f64().unwrap_or(0.0) as f32,
      &program32,
      program_n,
      // The assert above says `T` is `f32` here, so the values are already
      // the caller's precision.
      |data| {
        finish(
          unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len()) },
          components as usize,
        )
      },
    )
  }
}
