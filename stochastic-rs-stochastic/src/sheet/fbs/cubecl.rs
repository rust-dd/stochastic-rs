//! # CubeCL sheets
//!
//! The two-dimensional circulant embedding on whichever runtime a `Cubecl`
//! handle names: hashed complex Gaussian noise scaled by the embedding's
//! eigenvalue roots and scattered bit-reversed along each row, the row
//! transforms by a bounds-guarded radix-2 butterfly, a transpose that
//! bit-reverses the new rows, the column transforms, and the read-out of
//! the leading `m × n` block less its corner plus the low-rank correction.
//! Every kernel guards its thread count, so an embedding smaller than a
//! workgroup is stepped as exactly as a large one.

use cubecl::prelude::*;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fbs;
use super::SheetLaunch;
use crate::device::DeviceError;
use crate::euler::cubecl::CubeclRuntime;
use crate::noise::fgn::cubecl::backend::count_2d;
use crate::traits::FloatExt;

type DeviceResult<T> = std::result::Result<T, DeviceError>;

const WG_SIZE: usize = 256;

/// One complex normal per embedding cell — two decorrelated uniforms by
/// integer hashing on the batch-global cell, Box–Muller — scaled by the
/// eigenvalue root and written to its row's bit-reversed slot.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn sheet_gen<F: Float>(
  real: &mut Array<F>,
  imag: &mut Array<F>,
  lam: &Array<F>,
  rev: &Array<u32>,
  seed: u32,
  first_cell: u32,
  total: u32,
  #[comptime] cells: usize,
  #[comptime] cols: usize,
) {
  let tid = ABSOLUTE_POS;
  if (tid as u32) < total {
    let g = tid as u32 + first_cell;
    let mut a = (g * 2u32) ^ (seed * 2654435761u32);
    a ^= a >> 16;
    a *= 2246822519u32;
    a ^= a >> 13;
    a *= 3266489917u32;
    a ^= a >> 16;
    let mut b = (g * 2u32 + 1u32) ^ (seed * 668265263u32);
    b ^= b >> 16;
    b *= 2246822519u32;
    b ^= b >> 13;
    b *= 3266489917u32;
    b ^= b >> 16;
    let inv = F::new(2.3283064e-10_f32);
    let u1 = F::cast_from(a) * inv * F::new(0.999998_f32) + F::new(1.0e-6_f32);
    let u2 = F::cast_from(b) * inv;
    let radius = F::sqrt(F::new(-2.0_f32) * F::ln(u1));
    let angle = F::new(6.2831853071_f32) * u2;
    let local = tid % cells;
    let row = local / cols;
    let col = local % cols;
    let l = lam[local];
    let dst = tid - local + row * cols + rev[col] as usize;
    real[dst] = radius * F::cos(angle) * l;
    imag[dst] = radius * F::sin(angle) * l;
  }
}

/// One radix-2 stage of a batch of length-`n` transforms, guarded so a batch
/// smaller than the dispatch leaves the buffer's tail alone.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn sheet_butterfly<F: Float>(
  real: &mut Array<F>,
  imag: &mut Array<F>,
  pairs: u32,
  #[comptime] n: usize,
  #[comptime] half_stride: usize,
) {
  let tid = ABSOLUTE_POS;
  if (tid as u32) < pairs {
    let batch = tid / (n / 2);
    let local = tid % (n / 2);
    let stride = half_stride * 2;
    let group = local / half_stride;
    let pos = local % half_stride;
    let base = batch * n;
    let i = base + group * stride + pos;
    let j = i + half_stride;

    let a =
      F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(pos) / F::cast_from(stride);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (
      real[j] * tw_r - imag[j] * tw_i,
      real[j] * tw_i + imag[j] * tw_r,
    );
    let (ar, ai) = (real[i], imag[i]);
    real[i] = ar + tr;
    imag[i] = ai + ti;
    real[j] = ar - tr;
    imag[j] = ai - ti;
  }
}

/// Rows become columns, each new row bit-reversed for its own transform.
#[cube(launch)]
fn sheet_transpose<F: Float>(
  src_real: &Array<F>,
  src_imag: &Array<F>,
  dst_real: &mut Array<F>,
  dst_imag: &mut Array<F>,
  rev: &Array<u32>,
  total: u32,
  #[comptime] cells: usize,
  #[comptime] rows: usize,
  #[comptime] cols: usize,
) {
  let tid = ABSOLUTE_POS;
  if (tid as u32) < total {
    let local = tid % cells;
    let row = local / cols;
    let col = local % cols;
    let dst = tid - local + col * rows + rev[row] as usize;
    dst_real[dst] = src_real[tid];
    dst_imag[dst] = src_imag[tid];
  }
}

/// The real part of the leading `m × n` block less its corner, plus the
/// correction `√(2c₂) · (r (i+1)/m) z₁ · (r (j+1)/n) z₂` from two normals
/// hashed on a counter past every cell of the batch. `scal` carries `r` and
/// `√(2c₂)`.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn sheet_extract<F: Float>(
  freq_real: &Array<F>,
  output: &mut Array<F>,
  scal: &Array<F>,
  seed: u32,
  corr_cell: u32,
  total: u32,
  #[comptime] cells: usize,
  #[comptime] rows: usize,
  #[comptime] m: usize,
  #[comptime] n: usize,
) {
  let tid = ABSOLUTE_POS;
  if (tid as u32) < total {
    let per = m * n;
    let sheet = tid / per;
    let local = tid % per;
    let i = local / n;
    let j = local % n;
    let base = sheet * cells;
    let value = freq_real[base + j * rows + i] - freq_real[base];

    let g = corr_cell + sheet as u32;
    let mut a = (g * 2u32) ^ (seed * 2654435761u32);
    a ^= a >> 16;
    a *= 2246822519u32;
    a ^= a >> 13;
    a *= 3266489917u32;
    a ^= a >> 16;
    let mut b = (g * 2u32 + 1u32) ^ (seed * 668265263u32);
    b ^= b >> 16;
    b *= 2246822519u32;
    b ^= b >> 13;
    b *= 3266489917u32;
    b ^= b >> 16;
    let inv = F::new(2.3283064e-10_f32);
    let u1 = F::cast_from(a) * inv * F::new(0.999998_f32) + F::new(1.0e-6_f32);
    let u2 = F::cast_from(b) * inv;
    let radius = F::sqrt(F::new(-2.0_f32) * F::ln(u1));
    let angle = F::new(6.2831853071_f32) * u2;
    let z1 = radius * F::cos(angle);
    let z2 = radius * F::sin(angle);

    let r = scal[0];
    let corr = scal[1];
    let ty = r * F::cast_from(i + 1) / F::cast_from(m);
    let tx = r * F::cast_from(j + 1) / F::cast_from(n);
    output[tid] = value + corr * ty * z1 * tx * z2;
  }
}

/// Where each index of a length-`n` transform lands when its bits are
/// reversed.
fn bit_reverse_table(n: usize) -> Vec<u32> {
  let log_n = n.trailing_zeros() as usize;
  let bits = usize::BITS as usize;
  (0..n)
    .map(|i| (i.reverse_bits() >> (bits - log_n)) as u32)
    .collect()
}

/// The radix-2 stages of a batch of length-`n` transforms on the handle pair
/// the batch lies in.
fn launch_fft<C: CubeclRuntime>(
  cl: &ComputeClient<C::Rt>,
  real: &cubecl::server::Handle,
  imag: &cubecl::server::Handle,
  n: usize,
  total: usize,
) -> DeviceResult<()> {
  let pairs = total / 2;
  for stage in 0..n.trailing_zeros() as usize {
    unsafe {
      sheet_butterfly::launch::<f32, C::Rt>(
        cl,
        count_2d((pairs as u32).div_ceil(WG_SIZE as u32)),
        CubeDim::new_1d(WG_SIZE as u32),
        ArrayArg::from_raw_parts::<f32>(real, total, 1),
        ArrayArg::from_raw_parts::<f32>(imag, total, 1),
        ScalarArg::new(pairs as u32),
        n,
        1 << stage,
      )
      .map_err(|e| DeviceError::Launch(format!("sheet_butterfly stage {stage}: {e}")))?;
    }
  }
  Ok(())
}

/// One chunk of the batch: `sheets` sheets from `first` on, as their
/// `sheets · m · n` values. `corr_cell` is the batch-global counter the
/// correction's normals hash from, past every cell of the whole batch.
fn sample_chunk<C: CubeclRuntime>(
  sheet: &SheetLaunch<'_, f32>,
  sheets: usize,
  first: usize,
  corr_cell: u32,
  seed: u32,
  ordinal: usize,
) -> DeviceResult<Vec<f32>> {
  let (m, n) = (sheet.m, sheet.n);
  let big_m = 2 * (m - 1);
  let big_n = 2 * (n - 1);
  let cells = big_m * big_n;
  let total = sheets * cells;
  let out_len = sheets * m * n;

  let client = C::client(ordinal)?;
  let cl = &client;
  let lam_h = cl.create_from_slice(f32::as_bytes(sheet.lam));
  let rev_cols = cl.create_from_slice(u32::as_bytes(&bit_reverse_table(big_n)));
  let rev_rows = cl.create_from_slice(u32::as_bytes(&bit_reverse_table(big_m)));
  let hr = cl.empty(total * 4);
  let hi = cl.empty(total * 4);
  let htr = cl.empty(total * 4);
  let hti = cl.empty(total * 4);

  // 1. Draw, scale, scatter bit-reversed along the rows.
  unsafe {
    sheet_gen::launch::<f32, C::Rt>(
      cl,
      count_2d((total as u32).div_ceil(WG_SIZE as u32)),
      CubeDim::new_1d(WG_SIZE as u32),
      ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
      ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
      ArrayArg::from_raw_parts::<f32>(&lam_h, cells, 1),
      ArrayArg::from_raw_parts::<u32>(&rev_cols, big_n, 1),
      ScalarArg::new(seed & 0xffff),
      ScalarArg::new((first * cells) as u32),
      ScalarArg::new(total as u32),
      cells,
      big_n,
    )
    .map_err(|e| DeviceError::Launch(format!("sheet_gen: {e}")))?;
  }

  // 2. The row transforms.
  launch_fft::<C>(cl, &hr, &hi, big_n, total)?;

  // 3. Transpose, bit-reversing the new rows.
  unsafe {
    sheet_transpose::launch::<f32, C::Rt>(
      cl,
      count_2d((total as u32).div_ceil(WG_SIZE as u32)),
      CubeDim::new_1d(WG_SIZE as u32),
      ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
      ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
      ArrayArg::from_raw_parts::<f32>(&htr, total, 1),
      ArrayArg::from_raw_parts::<f32>(&hti, total, 1),
      ArrayArg::from_raw_parts::<u32>(&rev_rows, big_m, 1),
      ScalarArg::new(total as u32),
      cells,
      big_m,
      big_n,
    )
    .map_err(|e| DeviceError::Launch(format!("sheet_transpose: {e}")))?;
  }

  // 4. The column transforms.
  launch_fft::<C>(cl, &htr, &hti, big_m, total)?;

  // 5. Read out the leading block, shifted and corrected.
  let oh = cl.empty(out_len * 4);
  let sh = cl.create_from_slice(f32::as_bytes(&[sheet.r, sheet.corr]));
  unsafe {
    sheet_extract::launch::<f32, C::Rt>(
      cl,
      count_2d((out_len as u32).div_ceil(WG_SIZE as u32)),
      CubeDim::new_1d(WG_SIZE as u32),
      ArrayArg::from_raw_parts::<f32>(&htr, total, 1),
      ArrayArg::from_raw_parts::<f32>(&oh, out_len, 1),
      ArrayArg::from_raw_parts::<f32>(&sh, 2, 1),
      ScalarArg::new(seed & 0xffff),
      ScalarArg::new(corr_cell),
      ScalarArg::new(out_len as u32),
      cells,
      big_m,
      m,
      n,
    )
    .map_err(|e| DeviceError::Launch(format!("sheet_extract: {e}")))?;
  }

  let bytes = cl.read_one(oh);
  Ok(f32::from_bytes(&bytes).to_vec())
}

impl<T: FloatExt, S: SeedExt, B> Fbs<T, S, B> {
  /// `sheets` sheets on the runtime the handle names, in chunks that fit the
  /// batch budget: one seed for the whole batch, the cell hash offset per
  /// chunk by the cells already produced and the correction's hash by the
  /// sheets, so the result is the same whatever the budget.
  pub(crate) fn sample_cubecl_sheets<R: CubeclRuntime>(
    &self,
    sheets: usize,
    device: &crate::device::Cubecl<R>,
  ) -> DeviceResult<Vec<Array2<T>>> {
    let (m, n) = (self.m, self.n);
    let cells = self.cells();
    let lam: Vec<f32> = self.lam.iter().map(|x| x.to_f32().unwrap()).collect();
    let launch = SheetLaunch {
      lam: &lam,
      m,
      n,
      r: self.r.to_f32().unwrap(),
      corr: self.correction().to_f32().unwrap(),
      key: self.launch_key(),
    };
    let seed = self.seed.seed_value() as u32;
    let rows = crate::device::chunk_rows(device.batch_budget, 4 * cells + m * n, 4);
    let mut out = Vec::with_capacity(sheets);
    let mut first = 0;
    while first < sheets {
      let len = rows.min(sheets - first);
      let corr_cell = (sheets * cells + first) as u32;
      let flat = sample_chunk::<R>(&launch, len, first, corr_cell, seed, device.ordinal)?;
      out.extend(self.sheets_from_flat(&flat));
      first += len;
    }
    Ok(out)
  }
}

#[cfg(all(test, feature = "cubecl-wgpu"))]
mod wgpu_tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::device::Cubecl;
  use crate::device::WgpuRuntime;
  use crate::traits::ProcessExt;

  /// The interquartile range of a grid point across sheets: the spread a
  /// product of normals in the correction leaves a robust statistic of.
  fn iqr(sheets: &[Array2<f32>], i: usize, j: usize) -> f64 {
    let mut v: Vec<f64> = sheets.iter().map(|s| s[(i, j)] as f64).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[3 * v.len() / 4] - v[v.len() / 4]
  }

  /// The runtime's sheets carry the host field's law: the spread at the far
  /// corner and at the centre agree, and a batch produced in chunks equals
  /// one launch sheet for sheet.
  #[test]
  fn wgpu_sheets_match_the_host_law_and_chunk_exactly() {
    let build = || Fbs::<f32, _>::new(0.7, 9, 9, 1.0, Deterministic::new(11));
    let device = build().on::<Cubecl<WgpuRuntime>>().sample_par(12_000);
    let host = build().sample_par(12_000);
    for (i, j) in [(8, 8), (4, 4), (0, 8)] {
      let (h, d) = (iqr(&host, i, j), iqr(&device, i, j));
      assert!(
        (h / d - 1.0).abs() < 0.08,
        "Fbs ({i},{j}) spread: host {h}, device {d}"
      );
    }
    let per_sheet = (4 * 16 * 16 + 81) * 4;
    let chunked = build()
      .with_backend(Cubecl::<WgpuRuntime>::default().with_batch_budget(3 * per_sheet))
      .sample_par(12_000);
    assert_eq!(device, chunked);
  }
}
