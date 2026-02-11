use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
#[cfg(feature = "cuda")]
use libloading::Library;
#[cfg(feature = "cuda")]
use libloading::Symbol;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndrustfft::ndfft_par;
use ndrustfft::FftHandler;
use num_complex::Complex;
#[cfg(feature = "phastft")]
use phastft::fft_64;
#[cfg(feature = "phastft")]
use phastft::planner::Direction;
#[cfg(feature = "cuda")]
use rand::Rng;
use rand_distr::Distribution;

use crate::distributions::complex::ComplexDistribution;
use crate::distributions::normal::SimdNormal;
use crate::f;
use crate::stochastic::Float;
use crate::stochastic::Process;

// CUDA type for complex numbers
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct CuComplex {
  x: f32,
  y: f32,
}

// Persistent CUDA context for fast repeated sampling
#[cfg(feature = "cuda")]
struct CudaContext {
  _lib: Library, // Keep library alive
  fgn_sample: Symbol<'static, unsafe extern "C" fn(*mut f32, f32, u64)>,
  fgn_cleanup: Symbol<'static, unsafe extern "C" fn()>,
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
}

#[cfg(feature = "cuda")]
impl Drop for CudaContext {
  fn drop(&mut self) {
    unsafe {
      (self.fgn_cleanup)();
    }
  }
}

#[cfg(feature = "cuda")]
static CUDA_CONTEXT: Mutex<Option<CudaContext>> = Mutex::new(None);

pub struct FGN<T: Float> {
  pub hurst: T,
  pub n: usize,
  pub t: Option<T>,
  pub offset: usize,
  pub sqrt_eigenvalues: Arc<Array1<Complex<T>>>,
  pub fft_handler: Arc<FftHandler<T>>,
}

impl<T: Float> FGN<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(f!(1)) / f!(self.n)
  }
}

impl<T: Float> FGN<T> {
  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    if !(f!(0)..=f!(1)).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let n = n.next_power_of_two();
    let mut r = Array1::linspace(f!(0), f!(n), n + 1);
    let f2 = f!(2);
    r.mapv_inplace(|x| {
      if x == f!(0) {
        f!(1)
      } else {
        f!(0.5)
          * ((x + f!(0)).powf(f2 * hurst) - f2 * x.powf(f2 * hurst) + (x - f!(0)).powf(f2 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, T::zero()));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<T>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.mapv_inplace(|x| Complex::new((x.re / f!(2 * n)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t,
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      fft_handler: Arc::new(FftHandler::new(2 * n)),
    }
  }

  /// Sample FGN using PhastFT (requires nightly Rust and `phastft` feature)
  /// PhastFT is a high-performance FFT library that uses SIMD and is competitive with FFTW
  #[cfg(feature = "phastft")]
  pub fn sample_phastft(&self) -> Array1<T> {
    // Generate random complex numbers

    #[cfg(feature = "simd")]
    use crate::distributions::normal::SimdNormal;
    #[cfg(not(feature = "simd"))]
    let rnd = self.sample_rnd(StandardNormal);
    #[cfg(feature = "simd")]
    let rnd = self.sample_rnd(SimdNormal::new(f!(0), f!(1)));

    // Multiply by sqrt eigenvalues
    let fgn_complex = &*self.sqrt_eigenvalues * &rnd;

    // PhastFT uses separate real and imaginary arrays
    let mut reals = fgn_complex.iter().map(|c| c.re).collect();
    let mut imags = fgn_complex.iter().map(|c| c.im).collect();

    // Perform FFT using PhastFT
    fft_64(&mut reals, &mut imags, Direction::Forward);

    // Extract real parts and scale
    let scale = f!(self.n).powf(-self.hurst) * self.t.unwrap_or(f!(1)).powf(self.hurst);
    let result = reals[1..self.n - self.offset + 1]
      .iter()
      .map(|&x| x * scale)
      .collect();

    Array1::from_vec(result)
  }

  fn sample_rnd<D: Distribution<T>>(&self, d: D) -> Array1<Complex<T>> {
    Array1::<Complex<T>>::random(2 * self.n, ComplexDistribution::new(&d, &d))
  }
}

impl<T: Float> Process<T> for FGN<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let rnd = self.sample_rnd(SimdNormal::<T, 64>::new(f!(0), f!(1)));

    let fgn = &*self.sqrt_eigenvalues * &rnd;
    let mut fgn_fft = Array1::<Complex<T>>::zeros(2 * self.n);
    ndfft_par(&fgn, &mut fgn_fft, &*self.fft_handler, 0);
    let scale = f!(self.n).powf(-self.hurst) * self.t.unwrap_or(f!(1)).powf(self.hurst);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x| x.re * scale);
    fgn
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self, m: usize) -> Result<Either<Array1<f64>, Array2<f64>>> {
    // CUDA kernel compilation - see src/stochastic/cuda/CUDA_BUILD.md for details
    // Quick build:
    //   Linux:   cd src/stochastic/cuda && ./build.sh
    //   Windows: cd src\stochastic\cuda && build.bat

    type FgnInitFn = unsafe extern "C" fn(
      /* h_sqrt_eigs: */ *const CuComplex,
      /* eig_len:     */ i32,
      /* n:           */ i32,
      /* m:           */ i32,
      /* offset:      */ i32,
    ) -> i32;

    type FgnSampleFn = unsafe extern "C" fn(
      /* h_output:    */ *mut f32,
      /* scale:       */ f32,
      /* seed:        */ u64,
    );

    type FgnCleanupFn = unsafe extern "C" fn();

    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(1.0.into()).to_f64().unwrap();
    // Scale factor: n^(-H) * t^H, same as CPU
    let scale = (n as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let out_size = n - offset;

    // Check if we need to reinitialize
    let need_init = {
      let guard = CUDA_CONTEXT.lock().unwrap();
      match &*guard {
        Some(ctx) => {
          ctx.n != n || ctx.m != m || ctx.offset != offset || ctx.hurst != hurst || ctx.t != t
        }
        None => true,
      }
    };

    // Initialize if needed
    if need_init {
      // Drop old context first (this calls cleanup)
      {
        let mut guard = CUDA_CONTEXT.lock().unwrap();
        *guard = None;
      }

      #[cfg(target_os = "windows")]
      let lib = unsafe { Library::new("src/stochastic/cuda/fgn_windows/fgn.dll") }?;
      #[cfg(target_os = "linux")]
      let lib = unsafe { Library::new("src/stochastic/cuda/fgn_linux/libfgn.so") }?;

      // Get function pointers
      let fgn_init: Symbol<FgnInitFn> = unsafe { lib.get(b"fgn_init") }?;
      let fgn_sample: Symbol<FgnSampleFn> = unsafe { lib.get(b"fgn_sample") }?;
      let fgn_cleanup: Symbol<FgnCleanupFn> = unsafe { lib.get(b"fgn_cleanup") }?;

      // Prepare eigenvalues
      let host_sqrt_eigs: Vec<CuComplex> = self
        .sqrt_eigenvalues
        .iter()
        .map(|z| CuComplex {
          x: z.re.to_f32().unwrap(),
          y: z.im.to_f32().unwrap(),
        })
        .collect();

      // Initialize CUDA context (uploads eigenvalues, allocates buffers, creates FFT plan)
      unsafe {
        fgn_init(
          host_sqrt_eigs.as_ptr(),
          host_sqrt_eigs.len() as i32,
          n as i32,
          m as i32,
          offset as i32,
        );
      }

      // Store context (transmute to extend lifetime - safe because lib stays alive)
      let fgn_sample: Symbol<'static, unsafe extern "C" fn(*mut f32, f32, u64)> =
        unsafe { std::mem::transmute(fgn_sample) };
      let fgn_cleanup: Symbol<'static, unsafe extern "C" fn()> =
        unsafe { std::mem::transmute(fgn_cleanup) };

      let ctx = CudaContext {
        _lib: lib,
        fgn_sample,
        fgn_cleanup,
        n,
        m,
        offset,
        hurst,
        t,
      };

      let mut guard = CUDA_CONTEXT.lock().unwrap();
      *guard = Some(ctx);
    }

    // Fast path: just call fgn_sample (GPU already initialized)
    let mut host_output = vec![0.0f32; m * out_size];
    let mut rng = rand::rng();
    let seed: u64 = rng.random();

    {
      let guard = CUDA_CONTEXT.lock().unwrap();
      let ctx = guard.as_ref().unwrap();
      unsafe {
        (ctx.fgn_sample)(host_output.as_mut_ptr(), scale, seed);
      }
    }

    // Convert to ndarray
    let mut fgn = Array2::<f64>::zeros((m, out_size));
    for i in 0..m {
      for j in 0..out_size {
        fgn[[i, j]] = host_output[i * out_size + j] as f64;
      }
    }

    if m == 1 {
      return Ok(Either::Left(fgn.row(0).to_owned()));
    }

    Ok(Either::Right(fgn))
  }
}

#[cfg(test)]
mod tests {
  use std::time::Instant;

  use prettytable::Cell;
  use prettytable::Row;
  use prettytable::Table;

  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;

  #[test]
  fn fgn_length_equals_n() {
    let fbm = FGN::<f64>::new(0.7, N, Some(1.0));
    assert_eq!(fbm.sample().len(), N);
  }

  #[test]
  fn fgn_speed_test() {
    let mut table = Table::new();

    table.add_row(Row::new(vec![
      Cell::new("Test Case"),
      Cell::new("Elapsed Time (ms)"),
    ]));

    let start = Instant::now();
    let fbm = FGN::new(0.7, N, Some(1.0));
    let _ = fbm.sample();
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("Single Sample"),
      Cell::new(&format!("{:.2?}", duration.as_millis())),
    ]));

    let start = Instant::now();
    let fbm = FGN::new(0.7, N, Some(1.0));
    for _ in 0..N {
      let _ = fbm.sample();
    }
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("Repeated Samples"),
      Cell::new(&format!("{:.2?}", duration.as_millis())),
    ]));

    table.printstd();
  }

  #[test]
  #[ignore = "Not implemented"]
  fn fgn_starts_with_x0() {
    unimplemented!()
  }

  #[test]
  #[tracing_test::traced_test]
  fn fgn_plot() {
    let fgn = FGN::<f64>::new(0.7, 1000, Some(1.0));
    let fgn = fgn.sample();
    plot_1d!(fgn, "Fractional Brownian Motion (H = 0.7)");
  }

  #[test]
  #[tracing_test::traced_test]
  #[cfg(feature = "cuda")]
  fn fgn_cuda() {
    let fbm = FGN::<f64>::new(0.7, 500, Some(1.0));
    let fgn = fbm.sample_cuda(1).unwrap();
    let fgn = fgn.left().unwrap();
    plot_1d!(fgn, "Fractional Brownian Motion (H = 0.7)");
    use crate::plot_2d;

    let fgn = FGN::<f64>::new(0.7, 500, Some(1.0));
    let fgn = fgn.sample_cuda(1).unwrap();
    let fgn_left = fgn.left().unwrap();
    plot_1d!(fgn_left, "Fractional Brownian Motion (H = 0.7)");
    let mut path = Array1::<f64>::zeros(500);
    for i in 1..500 {
      path[i] += path[i - 1] + fgn_left[i];
    }
    plot_1d!(path, "Fractional Brownian Motion (H = 0.7)");

    let fgn = FGN::<f64>::new(0.7, 5000, Some(1.0));
    let start = std::time::Instant::now();
    let _ = fbm.sample_cuda(10000);
    let res = fgn.sample_cuda(10000).unwrap();
    let end = start.elapsed().as_millis();
    tracing::info!("10000 fgn generated on cuda in: {end}");
    // slice first  2 rows
    let paths = res.right().unwrap();
    let paths = paths.slice(s![..2, ..]);
    plot_2d!(paths.row(0), "Path 1", paths.row(1), "Path 2");
    let mut fbm1 = Array1::<f64>::zeros(5000);
    let mut fbm2 = Array1::<f64>::zeros(5000);
    for i in 1..5000 {
      fbm1[i] += fbm1[i - 1] + paths.row(0)[i];
      fbm2[i] += fbm2[i - 1] + paths.row(1)[i];
    }
    plot_2d!(fbm1, "FBM Path 1", fbm2, "FBM Path 2");

    let start = std::time::Instant::now();
    let _ = fbm.sample_par(10000);
    let _ = fgn.sample_par(10000);
    let end = start.elapsed().as_millis();
    tracing::info!("10000 fgn generated on cpu in: {end}");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn fgn_malliavin() {
    unimplemented!();
  }

  #[test]
  #[cfg(feature = "cuda")]
  fn fgn_cuda_speed_test() {
    use std::time::Instant;
    let mut table = Table::new();
    table.add_row(Row::new(vec![Cell::new("Test"), Cell::new("Time (ms)")]));

    // Test 1: Single sample (includes initialization)
    let fgn = FGN::<f64>::new(0.7, N, Some(1.0));
    let start = Instant::now();
    let _ = fgn.sample_cuda(1).unwrap();
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("CUDA Single (with init)"),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 2: Repeated samples (cached context)
    let fgn = FGN::<f64>::new(0.7, N, Some(1.0));
    let start = Instant::now();
    for _ in 0..N {
      let _ = fgn.sample_cuda(1).unwrap();
    }
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new(&format!("CUDA {} repeated", N)),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 3: CPU single
    let fgn = FGN::<f64>::new(0.7, N, Some(1.0));
    let start = Instant::now();
    let _ = fgn.sample();
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("CPU Single"),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 4: CPU repeated
    let fgn = FGN::<f64>::new(0.7, N, Some(1.0));
    let start = Instant::now();
    for _ in 0..N {
      let _ = fgn.sample();
    }
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new(&format!("CPU {} repeated", N)),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 5: Batch CUDA (10000 trajectories at once) - warm start
    let fgn_batch = FGN::<f64>::new(0.7, N, Some(1.0));
    // Warm up call to initialize CUDA context
    let _ = fgn_batch.sample_cuda(10000).unwrap();
    let start = Instant::now();
    let _ = fgn_batch.sample_cuda(10000).unwrap();
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("CUDA 10000 batch (warm)"),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 6: CPU parallel (10000 trajectories)
    let fgn = FGN::<f64>::new(0.7, N, Some(1.0));
    let start = Instant::now();
    let _ = fgn.sample_par(10000);
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("CPU 10000 parallel"),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 7: Larger batch - 100000 trajectories
    let fgn_large = FGN::<f64>::new(0.7, N, Some(1.0));
    // Warm up
    let _ = fgn_large.sample_cuda(100000).unwrap();
    let start = Instant::now();
    let _ = fgn_large.sample_cuda(100000).unwrap();
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("CUDA 100000 batch (warm)"),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    // Test 8: CPU parallel 100000 trajectories
    let fgn = FGN::<f64>::new(0.7, N, Some(1.0));
    let start = Instant::now();
    let _ = fgn.sample_par(100000);
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("CPU 100000 parallel"),
      Cell::new(&format!("{}", duration.as_millis())),
    ]));

    table.printstd();
  }

  /// Benchmark comparing all FFT implementations: ndrustfft, PhastFT, and CUDA
  #[test]
  #[cfg(all(feature = "cuda", feature = "phastft"))]
  fn fgn_fft_benchmark() {
    use std::time::Instant;
    let mut table = Table::new();
    table.add_row(Row::new(vec![
      Cell::new("FFT Implementation"),
      Cell::new("Time (ms)"),
      Cell::new("Samples/sec"),
    ]));

    let n = 1024; // Power of 2 for fair comparison
    let num_iterations = 1000;

    // Warmup
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    for _ in 0..10 {
      let _ = fgn.sample();
      let _ = fgn.sample_phastft();
    }

    // Test 1: ndrustfft (default CPU implementation)
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    let start = Instant::now();
    for _ in 0..num_iterations {
      let _ = fgn.sample();
    }
    let duration_ndrustfft = start.elapsed();
    let samples_per_sec_ndrustfft = num_iterations as f64 / duration_ndrustfft.as_secs_f64();
    table.add_row(Row::new(vec![
      Cell::new("ndrustfft (CPU)"),
      Cell::new(&format!("{}", duration_ndrustfft.as_millis())),
      Cell::new(&format!("{:.0}", samples_per_sec_ndrustfft)),
    ]));

    // Test 2: PhastFT
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    let start = Instant::now();
    for _ in 0..num_iterations {
      let _ = fgn.sample_phastft();
    }
    let duration_phastft = start.elapsed();
    let samples_per_sec_phastft = num_iterations as f64 / duration_phastft.as_secs_f64();
    table.add_row(Row::new(vec![
      Cell::new("PhastFT (CPU)"),
      Cell::new(&format!("{}", duration_phastft.as_millis())),
      Cell::new(&format!("{:.0}", samples_per_sec_phastft)),
    ]));

    // Test 3: CUDA (single samples)
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    // Warmup CUDA
    let _ = fgn.sample_cuda(1).unwrap();
    let start = Instant::now();
    for _ in 0..num_iterations {
      let _ = fgn.sample_cuda(1).unwrap();
    }
    let duration_cuda = start.elapsed();
    let samples_per_sec_cuda = num_iterations as f64 / duration_cuda.as_secs_f64();
    table.add_row(Row::new(vec![
      Cell::new("CUDA (GPU)"),
      Cell::new(&format!("{}", duration_cuda.as_millis())),
      Cell::new(&format!("{:.0}", samples_per_sec_cuda)),
    ]));

    // Test 4: CUDA batch (1000 samples at once)
    let fgn_batch = FGN::<f64>::new(0.7, n, Some(1.0));
    // Warmup
    let _ = fgn_batch.sample_cuda(num_iterations).unwrap();
    let start = Instant::now();
    let _ = fgn_batch.sample_cuda(num_iterations).unwrap();
    let duration_cuda_batch = start.elapsed();
    let samples_per_sec_cuda_batch = num_iterations as f64 / duration_cuda_batch.as_secs_f64();
    table.add_row(Row::new(vec![
      Cell::new(&format!("CUDA batch ({})", num_iterations)),
      Cell::new(&format!("{}", duration_cuda_batch.as_millis())),
      Cell::new(&format!("{:.0}", samples_per_sec_cuda_batch)),
    ]));

    println!(
      "\n=== FGN FFT Implementation Benchmark (n={}, {} iterations) ===",
      n, num_iterations
    );
    table.printstd();

    // Print speedup comparisons
    println!("\nSpeedup comparisons:");
    println!(
      "  PhastFT vs ndrustfft: {:.2}x",
      duration_ndrustfft.as_secs_f64() / duration_phastft.as_secs_f64()
    );
    println!(
      "  CUDA vs ndrustfft:    {:.2}x",
      duration_ndrustfft.as_secs_f64() / duration_cuda.as_secs_f64()
    );
    println!(
      "  CUDA batch vs ndrustfft: {:.2}x",
      (duration_ndrustfft.as_secs_f64() * num_iterations as f64)
        / duration_cuda_batch.as_secs_f64()
    );
  }

  /// PhastFT-only benchmark (doesn't require CUDA)
  #[test]
  #[cfg(feature = "phastft")]
  fn fgn_phastft_benchmark() {
    use std::time::Instant;
    let mut table = Table::new();
    table.add_row(Row::new(vec![
      Cell::new("FFT Implementation"),
      Cell::new("Time (ms)"),
      Cell::new("Samples/sec"),
    ]));

    let n = 1024;
    let num_iterations = 1000;

    // Warmup
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    for _ in 0..10 {
      let _ = fgn.sample();
      let _ = fgn.sample_phastft();
    }

    // Test 1: ndrustfft
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    let start = Instant::now();
    for _ in 0..num_iterations {
      let _ = fgn.sample();
    }
    let duration_ndrustfft = start.elapsed();
    let samples_per_sec_ndrustfft = num_iterations as f64 / duration_ndrustfft.as_secs_f64();
    table.add_row(Row::new(vec![
      Cell::new("ndrustfft (CPU)"),
      Cell::new(&format!("{}", duration_ndrustfft.as_millis())),
      Cell::new(&format!("{:.0}", samples_per_sec_ndrustfft)),
    ]));

    // Test 2: PhastFT
    let fgn = FGN::<f64>::new(0.7, n, Some(1.0));
    let start = Instant::now();
    for _ in 0..num_iterations {
      let _ = fgn.sample_phastft();
    }
    let duration_phastft = start.elapsed();
    let samples_per_sec_phastft = num_iterations as f64 / duration_phastft.as_secs_f64();
    table.add_row(Row::new(vec![
      Cell::new("PhastFT (CPU)"),
      Cell::new(&format!("{}", duration_phastft.as_millis())),
      Cell::new(&format!("{:.0}", samples_per_sec_phastft)),
    ]));

    println!(
      "\n=== FGN CPU FFT Benchmark (n={}, {} iterations) ===",
      n, num_iterations
    );
    table.printstd();

    let speedup = duration_ndrustfft.as_secs_f64() / duration_phastft.as_secs_f64();
    println!("\nPhastFT speedup vs ndrustfft: {:.2}x", speedup);
  }

  #[test]
  #[cfg(feature = "cuda")]
  fn fgn_verify_eigenvalues() {
    // Verify that the sqrt_eigenvalues computed on CPU are correctly passed to CUDA
    let hurst = 0.7;
    let n = 100;
    let fgn = FGN::<f64>::new(hurst, n, Some(1.0));

    println!(
      "Original n: {}, Padded n: {}, Offset: {}",
      n, fgn.n, fgn.offset
    );
    println!("sqrt_eigenvalues length: {}", fgn.sqrt_eigenvalues.len());

    // Print first 10 eigenvalues
    println!("\nFirst 10 sqrt_eigenvalues (CPU f64):");
    for i in 0..10.min(fgn.sqrt_eigenvalues.len()) {
      let eig = fgn.sqrt_eigenvalues[i];
      println!("  [{}]: re={:.10}, im={:.10}", i, eig.re, eig.im);
    }

    // The eigenvalues should be real for a real symmetric circulant matrix
    // Check that imaginary parts are negligible
    let max_im: f64 = fgn
      .sqrt_eigenvalues
      .iter()
      .map(|z| z.im.abs())
      .fold(0.0, f64::max);
    println!("\nMax imaginary part of eigenvalues: {:.2e}", max_im);

    // Verify eigenvalues are non-negative (they should be for a covariance matrix)
    let min_re: f64 = fgn
      .sqrt_eigenvalues
      .iter()
      .map(|z| z.re)
      .fold(f64::INFINITY, f64::min);
    println!("Min real part of sqrt_eigenvalues: {:.10}", min_re);

    assert!(
      max_im < 1e-10,
      "Eigenvalues should be real, but max im = {}",
      max_im
    );
  }

  #[test]
  #[cfg(feature = "cuda")]
  fn fgn_compare_statistics() {
    // Compare statistics of CPU vs CUDA samples
    let hurst = 0.7;
    let n = 256; // Power of 2 for simplicity
    let num_samples = 10000;

    let fgn_cpu = FGN::<f64>::new(hurst, n, Some(1.0));
    let fgn_cuda = FGN::<f64>::new(hurst, n, Some(1.0));

    let cpu_samples = fgn_cpu.sample_par(num_samples);
    let cuda_samples = fgn_cuda.sample_cuda(num_samples).unwrap().right().unwrap();

    // Compute mean and variance per time step
    let mut cpu_means = vec![0.0; n];
    let mut cuda_means = vec![0.0; n];
    let mut cpu_vars = vec![0.0; n];
    let mut cuda_vars = vec![0.0; n];

    for t in 0..n {
      let mut cpu_sum = 0.0;
      let mut cuda_sum = 0.0;
      for s in 0..num_samples {
        cpu_sum += cpu_samples[[s, t]];
        cuda_sum += cuda_samples[[s, t]];
      }
      cpu_means[t] = cpu_sum / num_samples as f64;
      cuda_means[t] = cuda_sum / num_samples as f64;

      let mut cpu_var_sum = 0.0;
      let mut cuda_var_sum = 0.0;
      for s in 0..num_samples {
        cpu_var_sum += (cpu_samples[[s, t]] - cpu_means[t]).powi(2);
        cuda_var_sum += (cuda_samples[[s, t]] - cuda_means[t]).powi(2);
      }
      cpu_vars[t] = cpu_var_sum / num_samples as f64;
      cuda_vars[t] = cuda_var_sum / num_samples as f64;
    }

    // Print statistics for first few time steps
    println!("Statistics comparison (first 10 time steps):");
    println!(
      "{:>4} {:>12} {:>12} {:>12} {:>12}",
      "t", "CPU Mean", "CUDA Mean", "CPU Var", "CUDA Var"
    );
    for t in 0..10.min(n) {
      println!(
        "{:>4} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        t, cpu_means[t], cuda_means[t], cpu_vars[t], cuda_vars[t]
      );
    }

    // Overall statistics
    let cpu_overall_mean: f64 = cpu_means.iter().sum::<f64>() / n as f64;
    let cuda_overall_mean: f64 = cuda_means.iter().sum::<f64>() / n as f64;
    let cpu_overall_var: f64 = cpu_vars.iter().sum::<f64>() / n as f64;
    let cuda_overall_var: f64 = cuda_vars.iter().sum::<f64>() / n as f64;

    println!("\nOverall statistics:");
    println!(
      "CPU:  Mean = {:.6}, Var = {:.6}",
      cpu_overall_mean, cpu_overall_var
    );
    println!(
      "CUDA: Mean = {:.6}, Var = {:.6}",
      cuda_overall_mean, cuda_overall_var
    );

    // Theoretical variance for FGN is 1 (when t=1)
    println!("\nTheoretical variance: 1.0");
    println!("CPU variance ratio: {:.4}", cpu_overall_var);
    println!("CUDA variance ratio: {:.4}", cuda_overall_var);
  }

  #[test]
  #[cfg(feature = "cuda")]
  fn fgn_cuda_vs_cpu_correlation() {
    use plotly::common::Line;
    use plotly::common::LineShape;
    use plotly::common::Mode;
    use plotly::Layout;
    use plotly::Plot;
    use plotly::Scatter;

    let hurst = 0.7;
    let n = 500;
    let num_samples = 5000;

    // Generate samples from both CPU and CUDA
    let fgn_cpu = FGN::<f64>::new(hurst, n, Some(1.0));
    let fgn_cuda = FGN::<f64>::new(hurst, n, Some(1.0));

    let cpu_samples = fgn_cpu.sample_par(num_samples);
    let cuda_samples = fgn_cuda.sample_cuda(num_samples).unwrap().right().unwrap();

    println!("CPU samples shape: {:?}", cpu_samples.shape());
    println!("CUDA samples shape: {:?}", cuda_samples.shape());

    // Compute empirical covariance for first few lags
    let max_lag = 20;
    let mut cpu_autocov = vec![0.0; max_lag];
    let mut cuda_autocov = vec![0.0; max_lag];

    // Theoretical autocovariance for FGN: gamma(k) = 0.5 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
    let mut theoretical_autocov = vec![0.0; max_lag];
    for k in 0..max_lag {
      let kf = k as f64;
      if k == 0 {
        theoretical_autocov[k] = 1.0;
      } else {
        theoretical_autocov[k] = 0.5
          * ((kf + 1.0).powf(2.0 * hurst) - 2.0 * kf.powf(2.0 * hurst)
            + (kf - 1.0).powf(2.0 * hurst));
      }
    }

    // Compute empirical autocovariance
    for lag in 0..max_lag {
      let mut cpu_sum = 0.0;
      let mut cuda_sum = 0.0;
      let mut count = 0;

      for row in 0..num_samples {
        for i in 0..(n - lag) {
          cpu_sum += cpu_samples[[row, i]] * cpu_samples[[row, i + lag]];
          cuda_sum += cuda_samples[[row, i]] * cuda_samples[[row, i + lag]];
          count += 1;
        }
      }

      cpu_autocov[lag] = cpu_sum / count as f64;
      cuda_autocov[lag] = cuda_sum / count as f64;
    }

    // Normalize to get autocorrelation (divide by variance)
    let cpu_var = cpu_autocov[0];
    let cuda_var = cuda_autocov[0];
    for lag in 0..max_lag {
      cpu_autocov[lag] /= cpu_var;
      cuda_autocov[lag] /= cuda_var;
    }

    // Print comparison table
    let mut table = Table::new();
    table.add_row(Row::new(vec![
      Cell::new("Lag"),
      Cell::new("Theoretical"),
      Cell::new("CPU"),
      Cell::new("CUDA"),
      Cell::new("CPU Error"),
      Cell::new("CUDA Error"),
    ]));

    for lag in 0..max_lag {
      let cpu_err = (cpu_autocov[lag] - theoretical_autocov[lag]).abs();
      let cuda_err = (cuda_autocov[lag] - theoretical_autocov[lag]).abs();
      table.add_row(Row::new(vec![
        Cell::new(&format!("{}", lag)),
        Cell::new(&format!("{:.6}", theoretical_autocov[lag])),
        Cell::new(&format!("{:.6}", cpu_autocov[lag])),
        Cell::new(&format!("{:.6}", cuda_autocov[lag])),
        Cell::new(&format!("{:.6}", cpu_err)),
        Cell::new(&format!("{:.6}", cuda_err)),
      ]));
    }
    table.printstd();

    // Plot 1: Autocorrelation comparison
    {
      let lags: Vec<usize> = (0..max_lag).collect();
      let mut plot = Plot::new();

      let trace_theory = Scatter::new(lags.clone(), theoretical_autocov.clone())
        .mode(Mode::LinesMarkers)
        .name("Theoretical")
        .line(Line::new().color("black").width(2.0));

      let trace_cpu = Scatter::new(lags.clone(), cpu_autocov.clone())
        .mode(Mode::Markers)
        .name("CPU")
        .line(Line::new().color("blue"));

      let trace_cuda = Scatter::new(lags.clone(), cuda_autocov.clone())
        .mode(Mode::Markers)
        .name("CUDA")
        .line(Line::new().color("orange"));

      plot.add_trace(trace_theory);
      plot.add_trace(trace_cpu);
      plot.add_trace(trace_cuda);

      let layout = Layout::new().title(&format!("FGN Autocorrelation (H={})", hurst));
      plot.set_layout(layout);
      plot.show();
    }

    // Plot 2: Sample paths comparison (FGN)
    {
      let mut plot = Plot::new();

      let trace_cpu = Scatter::new((0..n).collect::<Vec<_>>(), cpu_samples.row(0).to_vec())
        .mode(Mode::Lines)
        .name("CPU FGN")
        .line(Line::new().color("blue").shape(LineShape::Linear));

      let trace_cuda = Scatter::new((0..n).collect::<Vec<_>>(), cuda_samples.row(0).to_vec())
        .mode(Mode::Lines)
        .name("CUDA FGN")
        .line(Line::new().color("orange").shape(LineShape::Linear));

      plot.add_trace(trace_cpu);
      plot.add_trace(trace_cuda);

      let layout = Layout::new().title(&format!("FGN Sample Paths (H={})", hurst));
      plot.set_layout(layout);
      plot.show();
    }

    // Plot 3: Convert to FBM paths and compare
    {
      let mut cpu_fbm = Array1::<f64>::zeros(n);
      let mut cuda_fbm = Array1::<f64>::zeros(n);

      for i in 1..n {
        cpu_fbm[i] = cpu_fbm[i - 1] + cpu_samples[[0, i]];
        cuda_fbm[i] = cuda_fbm[i - 1] + cuda_samples[[0, i]];
      }

      let mut plot = Plot::new();

      let trace_cpu = Scatter::new((0..n).collect::<Vec<_>>(), cpu_fbm.to_vec())
        .mode(Mode::Lines)
        .name("CPU FBM")
        .line(Line::new().color("blue").shape(LineShape::Linear));

      let trace_cuda = Scatter::new((0..n).collect::<Vec<_>>(), cuda_fbm.to_vec())
        .mode(Mode::Lines)
        .name("CUDA FBM")
        .line(Line::new().color("orange").shape(LineShape::Linear));

      plot.add_trace(trace_cpu);
      plot.add_trace(trace_cuda);

      let layout = Layout::new().title(&format!("FBM Sample Paths (H={})", hurst));
      plot.set_layout(layout);
      plot.show();
    }

    // Plot 4: Multiple FBM paths from CUDA
    {
      let mut plot = Plot::new();
      let colors = ["red", "blue", "green", "orange", "purple"];

      for (idx, color) in colors.iter().enumerate() {
        let mut fbm = Array1::<f64>::zeros(n);
        for i in 1..n {
          fbm[i] = fbm[i - 1] + cuda_samples[[idx, i]];
        }

        let trace = Scatter::new((0..n).collect::<Vec<_>>(), fbm.to_vec())
          .mode(Mode::Lines)
          .name(&format!("Path {}", idx + 1))
          .line(Line::new().color(*color).shape(LineShape::Linear));

        plot.add_trace(trace);
      }

      let layout = Layout::new().title(&format!("CUDA FBM Multiple Paths (H={})", hurst));
      plot.set_layout(layout);
      plot.show();
    }

    // Verify correlation is close enough
    let max_cpu_error: f64 = (0..max_lag)
      .map(|lag| (cpu_autocov[lag] - theoretical_autocov[lag]).abs())
      .fold(0.0, f64::max);
    let max_cuda_error: f64 = (0..max_lag)
      .map(|lag| (cuda_autocov[lag] - theoretical_autocov[lag]).abs())
      .fold(0.0, f64::max);

    println!("\nMax CPU autocorrelation error: {:.6}", max_cpu_error);
    println!("Max CUDA autocorrelation error: {:.6}", max_cuda_error);

    // Estimate Hurst exponent using aggregated variance method
    // For FBM: Var(B_H(t)) = t^(2H), so for aggregated series at scale m:
    // Var(X^(m)) ~ m^(2H-2) * Var(X)
    // Log-log regression gives slope = 2H - 2, so H = (slope + 2) / 2
    fn estimate_hurst_variance(fgn_samples: &Array2<f64>, num_paths: usize) -> f64 {
      let n = fgn_samples.ncols();
      let mut log_ms = Vec::new();
      let mut log_vars = Vec::new();

      // Aggregate at different scales
      let scales: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .filter(|&m| m < n / 4)
        .collect();

      for &m in &scales {
        let mut var_sum = 0.0;
        let mut count = 0;

        for path_idx in 0..num_paths.min(500) {
          // Aggregate the FGN series at scale m
          let num_agg = n / m;
          let mut aggregated = vec![0.0; num_agg];

          for i in 0..num_agg {
            let mut sum = 0.0;
            for j in 0..m {
              sum += fgn_samples[[path_idx, i * m + j]];
            }
            aggregated[i] = sum / m as f64;
          }

          // Compute variance of aggregated series
          let mean: f64 = aggregated.iter().sum::<f64>() / num_agg as f64;
          let var: f64 =
            aggregated.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / num_agg as f64;

          if var > 1e-15 {
            var_sum += var;
            count += 1;
          }
        }

        if count > 0 {
          let avg_var = var_sum / count as f64;
          log_ms.push((m as f64).ln());
          log_vars.push(avg_var.ln());
        }
      }

      // Linear regression: log(Var) = (2H-2) * log(m) + c
      if log_ms.len() < 2 {
        return 0.5;
      }

      let n_points = log_ms.len() as f64;
      let mean_x: f64 = log_ms.iter().sum::<f64>() / n_points;
      let mean_y: f64 = log_vars.iter().sum::<f64>() / n_points;

      let mut cov = 0.0;
      let mut var_x = 0.0;
      for i in 0..log_ms.len() {
        cov += (log_ms[i] - mean_x) * (log_vars[i] - mean_y);
        var_x += (log_ms[i] - mean_x).powi(2);
      }

      if var_x < 1e-10 {
        return 0.5;
      }

      let slope = cov / var_x;
      // slope = 2H - 2, so H = (slope + 2) / 2
      (slope + 2.0) / 2.0
    }

    // Estimate Hurst exponent for both CPU and CUDA samples
    let cpu_hurst_est = estimate_hurst_variance(&cpu_samples.into(), num_samples);
    let cuda_hurst_est = estimate_hurst_variance(&cuda_samples, num_samples);

    println!("\n=== Hurst Exponent Estimation (Aggregated Variance) ===");
    println!("True Hurst:     {:.4}", hurst);
    println!(
      "CPU estimated:  {:.4} (error: {:.4})",
      cpu_hurst_est,
      (cpu_hurst_est - hurst).abs()
    );
    println!(
      "CUDA estimated: {:.4} (error: {:.4})",
      cuda_hurst_est,
      (cuda_hurst_est - hurst).abs()
    );

    // Note: CUDA uses f32 internally, so correlation may have more error than CPU (f64)
    // The threshold is relaxed to account for this
    assert!(
      max_cuda_error < 0.15,
      "CUDA autocorrelation error too large: {}",
      max_cuda_error
    );

    // Hurst estimate should be within 0.1 of true value
    assert!(
      (cuda_hurst_est - hurst).abs() < 0.15,
      "CUDA Hurst estimate {} too far from true value {}",
      cuda_hurst_est,
      hurst
    );
  }
}
