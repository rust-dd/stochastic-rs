//! Circulant-embedding FFT stages beyond one tile, and the paired row readout.
//! Dietrich & Newsam (1997), DOI: 10.1137/S1064827592240555.

use metal_oxide_device::{ReadBuffer, ReadWriteBuffer, WriteBuffer, kernel, uninit_array};

#[derive(Clone, Copy)]
pub(crate) struct Complex {
  pub(crate) re: f32,
  pub(crate) im: f32,
}

impl Complex {
  pub(crate) fn add(self, other: Self) -> Self {
    Self {
      re: self.re + other.re,
      im: self.im + other.im,
    }
  }

  pub(crate) fn sub(self, other: Self) -> Self {
    Self {
      re: self.re - other.re,
      im: self.im - other.im,
    }
  }

  pub(crate) fn mul(self, other: Self) -> Self {
    Self {
      re: self.re * other.re - self.im * other.im,
      im: self.re * other.im + self.im * other.re,
    }
  }
}

/// Entry `index` of an interleaved cos/sin twiddle table, which must hold more than `index`.
pub(crate) unsafe fn twiddle(table: ReadBuffer<f32>, index: u32) -> Complex {
  // SAFETY: the caller bounds the entry.
  unsafe {
    Complex {
      re: table.load_unchecked(2 * index),
      im: table.load_unchecked(2 * index + 1),
    }
  }
}

#[derive(Clone, Copy)]
pub(crate) struct Readout {
  pub(crate) size: u32,
  pub(crate) scale: f32,
  pub(crate) rows: u32,
  pub(crate) parity: u32,
}

/// Writes frequency `pos` of transform `batch`, the real part to one row and the imaginary part
/// to the next; output must cover `rows * size` values.
pub(crate) unsafe fn put(
  output: WriteBuffer<f32>,
  config: Readout,
  batch: u32,
  pos: u32,
  value: Complex,
) {
  if pos >= 1 && pos - 1 < config.size {
    let row = (2 * batch) as i32 - config.parity as i32;
    // SAFETY: the caller sizes output for config.rows * config.size entries.
    unsafe {
      if row >= 0 && (row as u32) < config.rows {
        output.store_unchecked(row as u32 * config.size + pos - 1, value.re * config.scale);
      }
      if row + 1 >= 0 && ((row + 1) as u32) < config.rows {
        output.store_unchecked(
          (row + 1) as u32 * config.size + pos - 1,
          value.im * config.scale,
        );
      }
    }
  }
}

/// The `1 << LOG` positions `stride` apart that one thread combines.
#[derive(Clone, Copy)]
struct Column {
  batch: u32,
  offset: u32,
  base: u32,
  stride: u32,
}

impl Column {
  fn new<const LOG: u32>(tid: u32, log_n: u32, s0: u32) -> Self {
    let per = log_n - LOG;
    let c = tid & ((1 << per) - 1);
    let stride = 1 << s0;
    Self {
      batch: tid >> per,
      offset: (tid >> per) << log_n,
      base: (c & (stride - 1)) + ((c >> s0) << (s0 + LOG)),
      stride,
    }
  }

  fn position(self, k: usize) -> u32 {
    self.base + k as u32 * self.stride
  }
}

/// Applies DIT stages s0..s0 + LOG to `column`'s values; needs s0 + LOG <= log_n and a
/// 2^(log_n - 1)-entry twiddle table.
unsafe fn stages<const R: usize, const LOG: u32>(
  mut values: [Complex; R],
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  s0: u32,
  column: Column,
) -> [Complex; R] {
  let low = column.base & (column.stride - 1);
  let mut j = 0;
  while j < LOG {
    let half = 1_usize << j;
    let shift = log_n - (s0 + j) - 1;
    let mut k = 0;
    while k < R {
      if k & half == 0 {
        let pos = low + (k & (half - 1)) as u32 * column.stride;
        // SAFETY: pos < 2^(s0 + j), so the entry is below 2^(log_n - 1).
        let t = values[k + half].mul(unsafe { twiddle(twiddles, pos << shift) });
        values[k + half] = values[k].sub(t);
        values[k] = values[k].add(t);
      }
      k += 1;
    }
    j += 1;
  }
  values
}

/// Safety: see the `fft_columns_*` entry points.
unsafe fn pass<const R: usize, const LOG: u32>(
  real: ReadWriteBuffer<f32>,
  imag: ReadWriteBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  s0: u32,
  threads: u32,
) {
  let tid = crate::index();
  if tid < threads {
    let column = Column::new::<LOG>(tid, log_n, s0);
    // SAFETY: each thread owns R positions of one complete transform.
    unsafe {
      let mut values: [Complex; R] = uninit_array();
      let mut k = 0;
      while k < R {
        let i = column.offset + column.position(k);
        values[k] = Complex {
          re: real.load_unchecked(i),
          im: imag.load_unchecked(i),
        };
        k += 1;
      }
      let values = stages::<R, LOG>(values, twiddles, log_n, s0, column);
      let mut k = 0;
      while k < R {
        let i = column.offset + column.position(k);
        real.store_unchecked(i, values[k].re);
        imag.store_unchecked(i, values[k].im);
        k += 1;
      }
    }
  }
}

/// Safety: see the `fft_columns_extract_*` entry points.
#[allow(clippy::too_many_arguments)]
unsafe fn extract<const R: usize, const LOG: u32>(
  real: ReadBuffer<f32>,
  imag: ReadBuffer<f32>,
  twiddles: ReadBuffer<f32>,
  log_n: u32,
  s0: u32,
  output: WriteBuffer<f32>,
  config: Readout,
  threads: u32,
) {
  let tid = crate::index();
  if tid < threads {
    let column = Column::new::<LOG>(tid, log_n, s0);
    // SAFETY: each thread owns R positions of one complete transform and their output cells.
    unsafe {
      let mut values: [Complex; R] = uninit_array();
      let mut k = 0;
      while k < R {
        let i = column.offset + column.position(k);
        values[k] = Complex {
          re: real.load_unchecked(i),
          im: imag.load_unchecked(i),
        };
        k += 1;
      }
      let values = stages::<R, LOG>(values, twiddles, log_n, s0, column);
      let mut k = 0;
      while k < R {
        put(output, config, column.batch, column.position(k), values[k]);
        k += 1;
      }
    }
  }
}

macro_rules! pass {
  ($name:ident, $count:literal, $log:literal) => {
    /// # Safety
    /// Buffers hold whole length-2^log_n transforms done below stage s0, s0 + LOG <= log_n,
    /// threads is their length >> LOG and the twiddle table holds 2^(log_n - 1) entries.
    #[kernel]
    pub unsafe fn $name(
      real: ReadWriteBuffer<f32>,
      imag: ReadWriteBuffer<f32>,
      twiddles: ReadBuffer<f32>,
      log_n: u32,
      s0: u32,
      threads: u32,
    ) {
      // SAFETY: forwarded entry contract.
      unsafe { pass::<$count, $log>(real, imag, twiddles, log_n, s0, threads) }
    }
  };
}

macro_rules! extract {
  ($name:ident, $count:literal, $log:literal) => {
    /// # Safety
    /// As for `fft_columns_*` with s0 + LOG == log_n; output covers rows * size values and is
    /// disjoint from every input.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn $name(
      real: ReadBuffer<f32>,
      imag: ReadBuffer<f32>,
      twiddles: ReadBuffer<f32>,
      log_n: u32,
      s0: u32,
      output: WriteBuffer<f32>,
      size: u32,
      scale: f32,
      rows: u32,
      parity: u32,
      threads: u32,
    ) {
      let config = Readout {
        size,
        scale,
        rows,
        parity,
      };
      // SAFETY: forwarded entry contract.
      unsafe { extract::<$count, $log>(real, imag, twiddles, log_n, s0, output, config, threads) }
    }
  };
}

pass!(fft_columns_2, 4, 2);
pass!(fft_columns_3, 8, 3);
extract!(fft_columns_extract_1, 2, 1);
extract!(fft_columns_extract_2, 4, 2);
extract!(fft_columns_extract_3, 8, 3);
