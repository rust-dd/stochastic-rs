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
use crate::noise::fgn::cuda::sampler::counter_offset;
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

const CUFFT_FORWARD: i32 = -1;

/// The three kernels with `REAL` standing for the scalar; the Philox-2x32-10
/// counter runs on the batch-global cell (the draw) or on a counter past
/// every cell of the batch (the correction), so a chunk continues one
/// launch's stream.
const KERNELS: &str = r#"
__device__ inline void philox_pair(unsigned long long counter, unsigned long long seed, REAL* n1, REAL* n2)
{
    unsigned int lo = (unsigned int)counter;
    unsigned int hi = (unsigned int)(counter >> 32);
    unsigned int k  = (unsigned int)seed;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned long long p = (unsigned long long)0xD2511F53u * lo;
        lo = ((unsigned int)(p >> 32)) ^ hi ^ k;
        hi = (unsigned int)p;
        k += 0x9E3779B9u;
    }
    REAL u1 = ((REAL)lo + (REAL)0.5) * (REAL)2.3283064365386963e-10;
    REAL u2 = ((REAL)hi + (REAL)0.5) * (REAL)2.3283064365386963e-10;
    REAL r  = sqrt((REAL)-2.0 * log(u1));
    REAL angle = (REAL)6.283185307179586 * u2;
    *n1 = r * cos(angle);
    *n2 = r * sin(angle);
}

extern "C" __global__ void sheet_gen_REAL(
    REAL* __restrict__ data,
    const REAL* __restrict__ lam,
    int cells, int total,
    unsigned long long seed, unsigned long long seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    REAL n_re, n_im;
    philox_pair((unsigned long long)tid + seq, seed, &n_re, &n_im);
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
    unsigned long long seed, unsigned long long seq_corr,
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
    REAL z1, z2;
    philox_pair((unsigned long long)sheet + seq_corr, seed, &z1, &z2);
    REAL ty = r * (REAL)(i + 1) / (REAL)m;
    REAL tx = r * (REAL)(j + 1) / (REAL)n;
    out[tid] = value + corr * (ty * z1 + tx * z2);
}
"#;

fn kernel_source(real: &str) -> String {
  KERNELS.replace("REAL", real)
}

/// Persistent GPU state: the stream and the six compiled kernels.
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

fn ensure_kernels(ordinal: usize) -> Result<()> {
  let mut g = GPU.lock();
  if g.as_ref().is_some_and(|k| k.ordinal == ordinal) {
    return Ok(());
  }
  // A new context invalidates the per-size buffers and plans of the old one.
  *g = None;
  SIZED_F32.lock().clear();
  SIZED_F64.lock().clear();
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  let c = stream.context();

  let load = |real: &str| -> Result<(CudaFunction, CudaFunction, CudaFunction)> {
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

  *g = Some(Kernels {
    ordinal,
    stream,
    gen_f32,
    transpose_f32,
    extract_f32,
    gen_f64,
    transpose_f64,
    extract_f64,
  });
  Ok(())
}

/// A batched one-dimensional plan on the stream, destroyed with its cache
/// entry.
fn make_plan(
  stream: &Arc<CudaStream>,
  len: usize,
  batch: usize,
  ty: cufft::sys::cufftType,
) -> Result<cufft::sys::cufftHandle> {
  let plan = cufft::result::plan_1d(len as i32, ty, batch as i32)
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

    /// The last [`crate::device::CACHE_SLOTS`] per-size states, least recent first.
    static $cache: Mutex<Vec<$sized>> = Mutex::new(Vec::new());

    /// One chunk of the batch: `sheets` sheets from `first` on, as their
    /// `sheets · m · n` values. `corr_cell` is the batch-global counter the
    /// correction's normals run from, past every cell of the whole batch.
    fn $chunk(
      sheet: &SheetLaunch<'_, $t>,
      sheets: usize,
      first: usize,
      corr_cell: u64,
      seed: u64,
      ordinal: usize,
    ) -> Result<Vec<$t>> {
      let (m, n) = (sheet.m, sheet.n);
      let big_m = 2 * (m - 1);
      let big_n = 2 * (n - 1);
      let cells = big_m * big_n;
      let total = sheets * cells;
      let out_len = sheets * m * n;

      ensure_kernels(ordinal)?;
      // Clone the handles out of the global lock so another size can launch
      // concurrently; the per-size state below keeps its own lock.
      let (stream, draw, transpose, extract) = {
        let g = GPU.lock();
        let k = g.as_ref().unwrap();
        (
          k.stream.clone(),
          k.$gen.clone(),
          k.$transpose.clone(),
          k.$extract.clone(),
        )
      };
      let mut sized = $cache.lock();
      let s = crate::device::lru_slot(
        &mut sized,
        |s| s.m == m && s.n == n && s.sheets == sheets && s.key == sheet.key,
        || {
          let plan_rows = make_plan(&stream, big_n, sheets * big_m, cufft::sys::cufftType::$cufft_ty)?;
          let plan_cols = match make_plan(&stream, big_m, sheets * big_n, cufft::sys::cufftType::$cufft_ty) {
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

      let cells_i = cells as i32;
      let total_i = total as i32;
      let rows_i = big_m as i32;
      let cols_i = big_n as i32;
      let m_i = m as i32;
      let n_i = n as i32;
      let out_i = out_len as i32;
      let base = counter_offset(seed);
      let seq = base + (first * cells) as u64;
      let seq_corr = base + corr_cell;

      // 1. Draw and scale, in natural order.
      unsafe {
        stream
          .launch_builder(&draw)
          .arg(&mut s.d_data)
          .arg(&s.d_lam)
          .arg(&cells_i)
          .arg(&total_i)
          .arg(&seed)
          .arg(&seq)
          .launch(LaunchConfig::for_num_elems(total as u32))
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
          .launch(LaunchConfig::for_num_elems(total as u32))
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
          .arg(&seed)
          .arg(&seq_corr)
          .arg(&out_i)
          .launch(LaunchConfig::for_num_elems(out_len as u32))
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
  /// `sheets` sheets on the selected CUDA device, in chunks that fit the
  /// batch budget: one seed for the whole batch, the Philox counter offset
  /// per chunk by the cells already produced and the correction's counter by
  /// the sheets, so the result is the same whatever the budget. `f64` runs the
  /// double-precision kernels; a failing double launch is reported, never
  /// quietly replaced by the `f32` one.
  pub(crate) fn sample_cuda_sheets(
    &self,
    sheets: usize,
    device: &crate::device::Cuda,
  ) -> Result<Vec<Array2<T>>> {
    let (m, n) = (self.m, self.n);
    let cells = self.cells();
    let seed = self.seed.seed_value();
    let rows = crate::device::chunk_rows(
      device.batch_budget,
      4 * cells + m * n,
      std::mem::size_of::<T>(),
    );
    let key = self.launch_key();
    let mut out = Vec::with_capacity(sheets);
    let mut first = 0;
    while first < sheets {
      let len = rows.min(sheets - first);
      let corr_cell = (sheets * cells + first) as u64;
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
        let flat = sample_chunk_f32(&launch, len, first, corr_cell, seed, device.ordinal)?;
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
        let flat = sample_chunk_f64(&launch, len, first, corr_cell, seed, device.ordinal)?;
        out.extend(self.sheets_from_flat(&flat));
      }
      first += len;
    }
    Ok(out)
  }
}
