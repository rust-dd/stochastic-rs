//! # CUDA sheets
//!
//! The two-dimensional circulant embedding on the native CUDA runtime:
//! Philox complex Gaussian noise scaled by the embedding's eigenvalue roots
//! in natural order, the row transforms by a batched cuFFT plan, a
//! transpose, the column transforms by a second batched plan, and the
//! read-out of the leading `m × n` block less its corner plus Stein's linear
//! correction from two more Philox normals. Templated on `float` and
//! `double`, so an `f64` sheet is computed in double.

use std::any::TypeId;
use std::sync::Arc;

use cudarc::cufft;
use cudarc::driver::*;
use cudarc::nvrtc;
use ndarray::Array2;
use parking_lot::Mutex;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fbs;
use super::SheetLaunch;
use crate::device::DeviceError;
use crate::device::launch_len;
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

const CUFFT_FORWARD: i32 = -1;

/// The three kernels with `REAL` standing for the scalar; each normal is a Philox draw on the
/// batch-global cell (the draw) or sheet (the correction), so a chunk continues one launch.
const KERNELS: &str = r#"
extern "C" __global__ void sheet_gen_REAL(
    REAL* __restrict__ data,
    const REAL* __restrict__ lam,
    int cells, int total,
    unsigned int seed_lo, unsigned int seed_hi, unsigned long long first_cell)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    unsigned long long cell = first_cell + (unsigned long long)tid;
    unsigned int words[4];
    stoch_philox((unsigned int)cell, (unsigned int)(cell >> 32), 0u, site_sheet, seed_lo, seed_hi, words);
    REAL n_re = stoch_normal(words[0], words[1], words[2], words[3]);
    stoch_philox((unsigned int)cell, (unsigned int)(cell >> 32), 1u, site_sheet, seed_lo, seed_hi, words);
    REAL n_im = stoch_normal(words[0], words[1], words[2], words[3]);
    REAL l = lam[tid % cells];
    data[2 * tid]     = n_re * l;
    data[2 * tid + 1] = n_im * l;
}

extern "C" __global__ void sheet_transpose_REAL(
    const REAL* __restrict__ src,
    REAL* __restrict__ dst,
    int cells, int rows, int cols, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int sheet = tid / cells;
    int local = tid % cells;
    int row = local / cols;
    int col = local % cols;
    int d = sheet * cells + col * rows + row;
    dst[2 * d]     = src[2 * tid];
    dst[2 * d + 1] = src[2 * tid + 1];
}

extern "C" __global__ void sheet_extract_REAL(
    const REAL* __restrict__ freq,
    REAL* __restrict__ out,
    int cells, int rows, int m, int n,
    REAL r, REAL corr,
    unsigned int seed_lo, unsigned int seed_hi, unsigned long long first_sheet,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int per = m * n;
    int sheet = tid / per;
    int local = tid % per;
    int i = local / n;
    int j = local % n;
    int base = sheet * cells;
    REAL value = freq[2 * (base + j * rows + i)] - freq[2 * base];
    unsigned long long global = first_sheet + (unsigned long long)sheet;
    unsigned int words[4];
    stoch_philox((unsigned int)global, (unsigned int)(global >> 32), 0u, site_sheet_correction, seed_lo, seed_hi, words);
    REAL z1 = stoch_normal(words[0], words[1], words[2], words[3]);
    stoch_philox((unsigned int)global, (unsigned int)(global >> 32), 1u, site_sheet_correction, seed_lo, seed_hi, words);
    REAL z2 = stoch_normal(words[0], words[1], words[2], words[3]);
    REAL ty = r * (REAL)(i + 1) / (REAL)m;
    REAL tx = r * (REAL)(j + 1) / (REAL)n;
    out[tid] = value + corr * (ty * z1 + tx * z2);
}
"#;

pub(crate) fn kernel_source(real: &'static str) -> String {
  format!(
    "{}{}",
    crate::euler::kernel::philox_prelude(&crate::euler::kernel::cuda_language(real)),
    KERNELS.replace("REAL", real)
  )
}

/// Persistent GPU state: the stream and the six compiled kernels.
#[derive(Clone)]
struct Kernels {
  ordinal: usize,
  stream: Arc<CudaStream>,
  gen_f32: CudaFunction,
  transpose_f32: CudaFunction,
  extract_f32: CudaFunction,
  gen_f64: CudaFunction,
  transpose_f64: CudaFunction,
  extract_f64: CudaFunction,
}

// SAFETY: all GPU ops are serialised through the stream.
unsafe impl Send for Kernels {}

static GPU: Mutex<Option<Kernels>> = Mutex::new(None);

/// Drops the sheet kernels, cuFFT plans and buffers cached for `ordinal`.
pub(crate) fn release(ordinal: usize) {
  let mut gpu = GPU.lock();
  if crate::noise::fgn::cuda::frees(gpu.as_ref().map(|k| k.ordinal), ordinal) {
    invalidate(&mut gpu);
  }
}

/// Empties the cache: the per-size buffers and plans belong to the context that built them.
fn invalidate(gpu: &mut Option<Kernels>) {
  *gpu = None;
  SIZED_F32.lock().clear();
  SIZED_F64.lock().clear();
}

#[cfg(test)]
pub(crate) fn cached_sizes() -> usize {
  SIZED_F32.lock().len() + SIZED_F64.lock().len()
}

#[cfg(test)]
pub(crate) fn forget_kernels() {
  *GPU.lock() = None;
}

#[cfg(test)]
pub(crate) fn cached_context() -> Option<std::sync::Weak<CudaContext>> {
  GPU
    .lock()
    .as_ref()
    .map(|k| Arc::downgrade(k.stream.context()))
}

/// The stream and kernels of the device at `ordinal`, opened if need be and cloned under one lock.
fn ensure_kernels(ordinal: usize) -> Result<Kernels> {
  let mut g = GPU.lock();
  if let Some(k) = g.as_ref().filter(|k| k.ordinal == ordinal) {
    return Ok(k.clone());
  }
  invalidate(&mut g);
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  let c = stream.context();

  let load = |real: &'static str| -> Result<(CudaFunction, CudaFunction, CudaFunction)> {
    let src = kernel_source(real);
    let ptx = nvrtc::compile_ptx(src)
      .map_err(|e| DeviceError::Compile(format!("NVRTC sheet {real}: {e}")))?;
    let module = c
      .load_module(ptx)
      .map_err(|e| DeviceError::Launch(format!("load sheet {real}: {e}")))?;
    let f = |name: &str| {
      module
        .load_function(name)
        .map_err(|e| DeviceError::Launch(format!("fn {name}: {e}")))
    };
    Ok((
      f(&format!("sheet_gen_{real}"))?,
      f(&format!("sheet_transpose_{real}"))?,
      f(&format!("sheet_extract_{real}"))?,
    ))
  };
  let (gen_f32, transpose_f32, extract_f32) = load("float")?;
  let (gen_f64, transpose_f64, extract_f64) = load("double")?;

  let kernels = Kernels {
    ordinal,
    stream,
    gen_f32,
    transpose_f32,
    extract_f32,
    gen_f64,
    transpose_f64,
    extract_f64,
  };
  *g = Some(kernels.clone());
  Ok(kernels)
}

/// A batched one-dimensional plan on the stream, destroyed with its cache
/// entry.
fn make_plan(
  stream: &Arc<CudaStream>,
  len: usize,
  batch: usize,
  ty: cufft::sys::cufftType,
) -> Result<cufft::sys::cufftHandle> {
  let plan = cufft::result::plan_1d(launch_len(len, "nx")?, ty, launch_len(batch, "batch")?)
    .map_err(|e| DeviceError::Launch(format!("cuFFT plan: {e}")))?;
  unsafe {
    if let Err(e) = cufft::result::set_stream(plan, stream.cu_stream() as _) {
      let _ = cufft::result::destroy(plan);
      return Err(DeviceError::Launch(format!("cuFFT set_stream: {e}")));
    }
  }
  Ok(plan)
}

macro_rules! sheet_precision {
  ($sized:ident, $cache:ident, $chunk:ident, $t:ty, $cufft_ty:ident, $exec:ident, $gen:ident, $transpose:ident, $extract:ident) => {
    /// Per-size state: the two batched plans and the device buffers.
    struct $sized {
      plan_rows: cufft::sys::cufftHandle,
      plan_cols: cufft::sys::cufftHandle,
      d_lam: CudaSlice<$t>,
      d_data: CudaSlice<$t>,
      d_tmp: CudaSlice<$t>,
      d_out: CudaSlice<$t>,
      m: usize,
      n: usize,
      sheets: usize,
      key: (u64, u64),
    }

    impl Drop for $sized {
      fn drop(&mut self) {
        unsafe {
          let _ = cufft::result::destroy(self.plan_rows);
          let _ = cufft::result::destroy(self.plan_cols);
        }
      }
    }

    unsafe impl Send for $sized {}

    /// The last [`crate::device::CACHE_SLOTS`] per-size states, least recent first; a launch reuses
    /// one only on the stream that built it, which its plans run on.
    static $cache: Mutex<Vec<$sized>> = Mutex::new(Vec::new());

    /// One chunk of the batch: `sheets` sheets from `first` on, as their `sheets · m · n` values.
    fn $chunk(
      sheet: &SheetLaunch<'_, $t>,
      sheets: usize,
      first: usize,
      seed: u64,
      ordinal: usize,
    ) -> Result<Vec<$t>> {
      let (m, n) = (sheet.m, sheet.n);
      let big_m = 2 * (m - 1);
      let big_n = 2 * (n - 1);
      let cells = big_m * big_n;
      let total = sheets * cells;
      let out_len = sheets * m * n;
      // The draw and the transpose index `data[2 * tid + 1]` for each `tid` below `total` in `int`,
      // so the work buffers' whole length must fit it; the read-out stays inside that buffer.
      launch_len::<i32>(2 * total, "data")?;
      let cells_i = launch_len::<i32>(cells, "cells")?;
      let total_i = launch_len::<i32>(total, "total")?;
      let rows_i = launch_len::<i32>(big_m, "rows")?;
      let cols_i = launch_len::<i32>(big_n, "cols")?;
      let m_i = launch_len::<i32>(m, "m")?;
      let n_i = launch_len::<i32>(n, "n")?;
      let out_i = launch_len::<i32>(out_len, "total")?;
      let grid = launch_len::<u32>(total, "sheet grid")?;
      let grid_out = launch_len::<u32>(out_len, "sheet_extract grid")?;

      let Kernels {
        stream,
        $gen: draw,
        $transpose: transpose,
        $extract: extract,
        ..
      } = ensure_kernels(ordinal)?;
      let mut sized = $cache.lock();
      let s = crate::device::lru_slot(
        &mut sized,
        |s| {
          Arc::ptr_eq(s.d_out.stream(), &stream)
            && s.m == m
            && s.n == n
            && s.sheets == sheets
            && s.key == sheet.key
        },
        || {
          let plan_rows = make_plan(
            &stream,
            big_n,
            sheets * big_m,
            cufft::sys::cufftType::$cufft_ty,
          )?;
          let plan_cols = match make_plan(
            &stream,
            big_m,
            sheets * big_n,
            cufft::sys::cufftType::$cufft_ty,
          ) {
            Ok(plan) => plan,
            Err(e) => {
              unsafe {
                let _ = cufft::result::destroy(plan_rows);
              }
              return Err(e);
            }
          };
          let alloc = |len: usize, what: &str| {
            stream
              .alloc_zeros::<$t>(len)
              .map_err(|e| DeviceError::Launch(format!("alloc {what}: {e}")))
          };
          let built = (|| {
            Ok($sized {
              plan_rows,
              plan_cols,
              d_lam: stream
                .clone_htod(sheet.lam)
                .map_err(|e| DeviceError::Launch(format!("htod lam: {e}")))?,
              d_data: alloc(2 * total, "data")?,
              d_tmp: alloc(2 * total, "tmp")?,
              d_out: alloc(out_len, "out")?,
              m,
              n,
              sheets,
              key: sheet.key,
            })
          })();
          if built.is_err() {
            unsafe {
              let _ = cufft::result::destroy(plan_rows);
              let _ = cufft::result::destroy(plan_cols);
            }
          }
          built
        },
      )?;

      let [seed_lo, seed_hi] = crate::euler::philox::key(seed);
      let (first_cell, first_sheet) = ((first * cells) as u64, first as u64);

      // 1. Draw and scale, in natural order.
      unsafe {
        stream
          .launch_builder(&draw)
          .arg(&mut s.d_data)
          .arg(&s.d_lam)
          .arg(&cells_i)
          .arg(&total_i)
          .arg(&seed_lo)
          .arg(&seed_hi)
          .arg(&first_cell)
          .launch(LaunchConfig::for_num_elems(grid))
          .map_err(|e| DeviceError::Launch(format!("sheet_gen: {e}")))?;
      }

      // 2. The row transforms.
      {
        let (ptr, _g) = s.d_data.device_ptr_mut(&stream);
        unsafe {
          cufft::result::$exec(s.plan_rows, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
            .map_err(|e| DeviceError::Launch(format!("cuFFT rows: {e}")))?;
        }
      }

      // 3. Transpose.
      unsafe {
        stream
          .launch_builder(&transpose)
          .arg(&s.d_data)
          .arg(&mut s.d_tmp)
          .arg(&cells_i)
          .arg(&rows_i)
          .arg(&cols_i)
          .arg(&total_i)
          .launch(LaunchConfig::for_num_elems(grid))
          .map_err(|e| DeviceError::Launch(format!("sheet_transpose: {e}")))?;
      }

      // 4. The column transforms.
      {
        let (ptr, _g) = s.d_tmp.device_ptr_mut(&stream);
        unsafe {
          cufft::result::$exec(s.plan_cols, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
            .map_err(|e| DeviceError::Launch(format!("cuFFT columns: {e}")))?;
        }
      }

      // 5. Read out the leading block, shifted and corrected.
      unsafe {
        stream
          .launch_builder(&extract)
          .arg(&s.d_tmp)
          .arg(&mut s.d_out)
          .arg(&cells_i)
          .arg(&rows_i)
          .arg(&m_i)
          .arg(&n_i)
          .arg(&sheet.r)
          .arg(&sheet.corr)
          .arg(&seed_lo)
          .arg(&seed_hi)
          .arg(&first_sheet)
          .arg(&out_i)
          .launch(LaunchConfig::for_num_elems(grid_out))
          .map_err(|e| DeviceError::Launch(format!("sheet_extract: {e}")))?;
      }

      stream
        .synchronize()
        .map_err(|e| DeviceError::Launch(format!("sync: {e}")))?;
      let out = stream
        .clone_dtoh(&s.d_out)
        .map_err(|e| DeviceError::Launch(format!("dtoh: {e}")))?;
      drop(sized);
      Ok(out)
    }
  };
}

sheet_precision!(
  SizedF32,
  SIZED_F32,
  sample_chunk_f32,
  f32,
  CUFFT_C2C,
  exec_c2c,
  gen_f32,
  transpose_f32,
  extract_f32
);
sheet_precision!(
  SizedF64,
  SIZED_F64,
  sample_chunk_f64,
  f64,
  CUFFT_Z2Z,
  exec_z2z,
  gen_f64,
  transpose_f64,
  extract_f64
);

impl<T: FloatExt, S: SeedExt, B> Fbs<T, S, B> {
  /// `sheets` sheets in chunks that fit the batch budget, one seed for the batch, so the result is
  /// the same whatever the budget; a failing `f64` launch is reported, never replaced by `f32`.
  pub(crate) fn sample_cuda_sheets(
    &self,
    sheets: usize,
    device: &crate::device::Cuda,
  ) -> Result<Vec<Array2<T>>> {
    let (m, n) = (self.rows, self.cols);
    let cells = self.cells();
    let seed = self.seed.next_seed();
    // A sheet is `cells` complex values of work buffer, which the kernels
    // index as `data[2 * tid + 1]` in `int`.
    let rows = crate::device::rows_within_index_limit(
      crate::device::chunk_rows(
        device.batch_budget,
        4 * cells + m * n,
        std::mem::size_of::<T>(),
      ),
      2 * cells,
      i32::MAX as usize,
    );
    let key = self.launch_key();
    let mut out = Vec::with_capacity(sheets);
    let mut first = 0;
    while first < sheets {
      let len = rows.min(sheets - first);
      if TypeId::of::<T>() == TypeId::of::<f32>() {
        let lam: Vec<f32> = self.lam.iter().map(|x| x.to_f32().unwrap()).collect();
        let launch = SheetLaunch {
          lam: &lam,
          m,
          n,
          r: self.r.to_f32().unwrap(),
          corr: self.correction().to_f32().unwrap(),
          key,
        };
        let flat = sample_chunk_f32(&launch, len, first, seed, device.ordinal)?;
        out.extend(self.sheets_from_flat(&flat));
      } else {
        let lam: Vec<f64> = self.lam.iter().map(|x| x.to_f64().unwrap()).collect();
        let launch = SheetLaunch {
          lam: &lam,
          m,
          n,
          r: self.r.to_f64().unwrap(),
          corr: self.correction().to_f64().unwrap(),
          key,
        };
        let flat = sample_chunk_f64(&launch, len, first, seed, device.ordinal)?;
        out.extend(self.sheets_from_flat(&flat));
      }
      first += len;
    }
    Ok(out)
  }
}
