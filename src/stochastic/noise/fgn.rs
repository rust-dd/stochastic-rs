use std::sync::{Arc, RwLock};

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;

use ndarray::parallel::prelude::*;
use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndrustfft::{ndfft_par, FftHandler};
use num_complex::{Complex, ComplexDistribution};

use crate::stochastic::SamplingExt;

pub struct FGN<T> {
  pub hurst: T,
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub offset: usize,
  pub sqrt_eigenvalues: Arc<Array1<Complex<T>>>,
  pub fft_handler: Arc<FftHandler<T>>,
}

impl FGN<f64> {
  #[must_use]
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let n = n.next_power_of_two();
    let mut r = Array1::linspace(0.0, n as f64, n + 1);
    r.mapv_inplace(|x| {
      if x == 0.0 {
        1.0
      } else {
        0.5
          * ((x + 1.0).powf(2.0 * hurst) - 2.0 * x.powf(2.0 * hurst) + (x - 1.0).powf(2.0 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, 0.0));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f64>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.mapv_inplace(|x| Complex::new((x.re / (2.0 * n as f64)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t,
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      m,
      fft_handler: Arc::new(FftHandler::new(2 * n)),
    }
  }
}

impl SamplingExt<f64> for FGN<f64> {
  fn sample(&self) -> Array1<f64> {
    // let rnd = Array1::<Complex<f64>>::random(
    //   2 * self.n,
    //   ComplexDistribution::new(StandardNormal, StandardNormal),
    // );
    let num_threads = rayon::current_num_threads();
    let chunk_size = (2 * self.n) / num_threads;
    let rnd = Arc::new(RwLock::new(Array1::<Complex<f64>>::zeros(2 * self.n)));

    (0..num_threads).into_par_iter().for_each(|i| {
      let chunk = Array1::<Complex<f64>>::random(
        chunk_size,
        ComplexDistribution::new(StandardNormal, StandardNormal),
      );

      let mut result_lock = rnd.write().unwrap();
      result_lock
        .slice_mut(s![i * chunk_size..(i + 1) * chunk_size])
        .assign(&chunk);
    });

    let fgn = &*self.sqrt_eigenvalues * &*rnd.read().unwrap();
    let mut fgn_fft = Array1::<Complex<f64>>::zeros(2 * self.n);
    ndfft_par(&fgn, &mut fgn_fft, &*self.fft_handler, 0);
    let scale = (self.n as f64).powf(-self.hurst) * self.t.unwrap_or(1.0).powf(self.hurst);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x: Complex<f64>| x.re * scale);
    fgn
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self) -> Result<Either<Array1<f64>, Array2<f64>>> {
    // nvcc -shared -Xcompiler -fPIC fgn.cu -o libfgn.so -lcufft // ELF header error
    // nvcc -shared -o libfgn.so fgn.cu -Xcompiler -fPIC
    // nvcc -O3 -use_fast_math -o libfgn.so fgn.cu -Xcompiler -fPIC
    // nvcc -shared fgn.cu -o fgn.dll -lcufft
    use std::ffi::c_void;

    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut, DeviceRepr};

    use libloading::{Library, Symbol};

    #[repr(C)]
    #[derive(Debug, Default, Copy, Clone)]
    pub struct cuComplex {
      pub x: f32,
      pub y: f32,
    }

    unsafe impl DeviceRepr for cuComplex {
      fn as_kernel_param(&self) -> *mut c_void {
        self as *const Self as *mut _
      }
    }

    type FgnKernelFn = unsafe extern "C" fn(
      /* d_sqrt_eigs: */ *const cuComplex,
      /* d_output:    */ *mut f32,
      /* n:           */ i32,
      /* m:           */ i32,
      /* offset:      */ i32,
      /* hurst:       */ f32,
      /* t:           */ f32,
      /* seed:        */ u64,
    );

    #[cfg(target_os = "windows")]
    let lib = unsafe { Library::new("src/stochastic/cuda/fgn_windows/fgn.dll") }?;

    #[cfg(target_os = "linux")]
    let lib = unsafe { Library::new("src/stochastic/cuda/fgn_linux/libfgn.so") }?;

    let fgn_kernel: Symbol<FgnKernelFn> = unsafe { lib.get(b"fgn_kernel") }?;
    let device = CudaDevice::new(0)?;

    let m = self.m.unwrap_or(1);
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst;
    let t = self.t.unwrap_or(1.0);
    let seed = 42u64;

    let host_sqrt_eigs: Vec<cuComplex> = self
      .sqrt_eigenvalues
      .iter()
      .map(|z| cuComplex {
        x: z.re as f32,
        y: z.im as f32,
      })
      .collect();
    let d_sqrt_eigs = device.htod_copy(host_sqrt_eigs)?;
    let mut d_output = device.alloc_zeros::<f32>(m * (n - offset))?;

    unsafe {
      fgn_kernel(
        (*d_sqrt_eigs.device_ptr()) as *const cuComplex,
        (*d_output.device_ptr_mut()) as *mut f32,
        n as i32,
        m as i32,
        offset as i32,
        hurst as f32,
        t as f32,
        seed,
      );
    }

    let host_output = device.sync_reclaim(d_output)?;
    let mut fgn = Array2::<f64>::zeros((m, n - offset));
    for i in 0..m {
      for j in 0..(n - offset) {
        fgn[[i, j]] = host_output[i * (n - offset) + j] as f64;
      }
    }

    if m == 1 {
      let fgn = fgn.row(0).to_owned();
      return Ok(Either::Left(fgn));
    }

    Ok(Either::Right(fgn))
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n - self.offset
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl FGN<f32> {
  #[must_use]
  pub fn new(hurst: f32, n: usize, t: Option<f32>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let n = n.next_power_of_two();
    let mut r = Array1::linspace(0.0, n as f32, n + 1);
    r.mapv_inplace(|x| {
      if x == 0.0 {
        1.0
      } else {
        0.5
          * ((x + 1.0).powf(2.0 * hurst) - 2.0 * x.powf(2.0 * hurst) + (x - 1.0).powf(2.0 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, 0.0));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f32>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.mapv_inplace(|x| Complex::new((x.re / (2.0 * n as f32)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t,
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      m,
      fft_handler: Arc::new(FftHandler::new(2 * n)),
    }
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for FGN<f32> {
  fn sample(&self) -> Array1<f32> {
    let num_threads = rayon::current_num_threads();
    let chunk_size = (2 * self.n) / num_threads;
    let rnd = Arc::new(RwLock::new(Array1::<Complex<f32>>::zeros(2 * self.n)));

    (0..num_threads).into_par_iter().for_each(|i| {
      let chunk = Array1::<Complex<f32>>::random(
        chunk_size,
        ComplexDistribution::new(StandardNormal, StandardNormal),
      );

      let mut result_lock = rnd.write().unwrap();
      result_lock
        .slice_mut(s![i * chunk_size..(i + 1) * chunk_size])
        .assign(&chunk);
    });

    let fgn = &*self.sqrt_eigenvalues * &*rnd.read().unwrap();
    let mut fgn_fft = Array1::<Complex<f32>>::zeros(2 * self.n);
    ndfft_par(&fgn, &mut fgn_fft, &*self.fft_handler, 0);
    let scale = (self.n as f32).powf(-self.hurst) * self.t.unwrap_or(1.0).powf(self.hurst);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x: Complex<f32>| x.re * scale);
    fgn
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n - self.offset
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use std::time::Instant;

  use prettytable::{Cell, Row, Table};

  use crate::{plot_1d, stochastic::N};

  use super::*;

  #[test]
  fn fgn_length_equals_n() {
    let fbm = FGN::new(0.7, N, Some(1.0), None);
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
    let fbm = FGN::new(0.7, N, Some(1.0), None);
    let _ = fbm.sample();
    let duration = start.elapsed();
    table.add_row(Row::new(vec![
      Cell::new("Single Sample"),
      Cell::new(&format!("{:.2?}", duration.as_millis())),
    ]));

    let start = Instant::now();
    let fbm = FGN::new(0.7, N, Some(1.0), None);
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
    let fgn = FGN::new(0.7, 100, Some(1.0), None);
    let fgn = fgn.sample();
    plot_1d!(fgn, "Fractional Brownian Motion (H = 0.7)");
  }

  #[test]
  #[tracing_test::traced_test]
  #[cfg(feature = "cuda")]
  fn fgn_cuda() {
    let fbm = FGN::new(0.7, 500, Some(1.0), Some(1));
    let fgn = fbm.sample_cuda().unwrap();
    let fgn = fgn.left().unwrap();
    plot_1d!(fgn, "Fractional Brownian Motion (H = 0.7)");
    let mut path = Array1::<f64>::zeros(500);
    for i in 1..500 {
      path[i] += path[i - 1] + fgn[i];
    }
    plot_1d!(path, "Fractional Brownian Motion (H = 0.7)");

    let start = std::time::Instant::now();
    let _ = fbm.sample_cuda();
    let end = start.elapsed().as_millis();
    tracing::info!("10000 fgn generated on cuda in: {end}");

    let start = std::time::Instant::now();
    let _ = fbm.sample_par();
    let end = start.elapsed().as_millis();
    tracing::info!("10000 fgn generated on cuda in: {end}");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fgn_malliavin() {
    unimplemented!();
  }
}
