//! Paired circulant-embedding FFT and readout.
//! Dietrich & Newsam (1997), DOI: 10.1137/S1064827592240555.

use metal_oxide_device::{ReadBuffer, ReadWriteBuffer, WriteBuffer, kernel, math};

#[derive(Clone, Copy)]
struct Complex {
  re: f32,
  im: f32,
}

impl Complex {
  fn add(self, other: Self) -> Self {
    Self {
      re: self.re + other.re,
      im: self.im + other.im,
    }
  }

  fn sub(self, other: Self) -> Self {
    Self {
      re: self.re - other.re,
      im: self.im - other.im,
    }
  }

  fn mul(self, other: Self) -> Self {
    Self {
      re: self.re * other.re - self.im * other.im,
      im: self.re * other.im + self.im * other.re,
    }
  }
}

fn twiddle(pos: u32, stride: u32) -> Complex {
  let angle = -2.0 * core::f32::consts::PI * pos as f32 / stride as f32;
  Complex {
    re: math::cos(angle),
    im: math::sin(angle),
  }
}

fn radix4(a0: Complex, a1: Complex, a2: Complex, a3: Complex, pos: u32, half: u32) -> [Complex; 4] {
  let w0 = twiddle(pos, half * 2);
  let t1 = a1.mul(w0);
  let b0 = a0.add(t1);
  let b1 = a0.sub(t1);
  let t3 = a3.mul(w0);
  let b2 = a2.add(t3);
  let b3 = a2.sub(t3);
  let t2 = b2.mul(twiddle(pos, half * 4));
  let t3 = b3.mul(twiddle(pos + half, half * 4));
  [b0.add(t2), b1.add(t3), b0.sub(t2), b1.sub(t3)]
}

/// # Safety
/// Both buffers cover `4 * quads` elements in power-of-two transforms of length n.
/// stage and stage+1 are valid transform stages; buffers must be disjoint.
#[kernel]
pub unsafe fn fft_radix4(
  real: ReadWriteBuffer<f32>,
  imag: ReadWriteBuffer<f32>,
  n: u32,
  stage: u32,
  quads: u32,
) {
  let tid = crate::index();
  let half = 1_u32 << stage;
  if tid < quads && n >= 4 && half > 0 {
    let per = n / 4;
    let batch = tid / per;
    let local = tid % per;
    let pos = local % half;
    let i0 = batch * n + (local / half) * (half * 4) + pos;
    let i1 = i0 + half;
    let i2 = i1 + half;
    let i3 = i2 + half;
    // SAFETY: each active lane owns one closed quadruple in both stages.
    unsafe {
      let a0 = Complex {
        re: real.load_unchecked(i0),
        im: imag.load_unchecked(i0),
      };
      let a1 = Complex {
        re: real.load_unchecked(i1),
        im: imag.load_unchecked(i1),
      };
      let a2 = Complex {
        re: real.load_unchecked(i2),
        im: imag.load_unchecked(i2),
      };
      let a3 = Complex {
        re: real.load_unchecked(i3),
        im: imag.load_unchecked(i3),
      };
      let result = radix4(a0, a1, a2, a3, pos, half);
      real.store_unchecked(i0, result[0].re);
      imag.store_unchecked(i0, result[0].im);
      real.store_unchecked(i1, result[1].re);
      imag.store_unchecked(i1, result[1].im);
      real.store_unchecked(i2, result[2].re);
      imag.store_unchecked(i2, result[2].im);
      real.store_unchecked(i3, result[3].re);
      imag.store_unchecked(i3, result[3].im);
    }
  }
}

#[derive(Clone, Copy)]
struct Readout {
  size: u32,
  scale: f32,
  rows: u32,
  parity: u32,
}

unsafe fn put(output: WriteBuffer<f32>, config: Readout, batch: u32, pos: u32, value: Complex) {
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

/// # Safety
/// Inputs cover 4 * quads values in length-n transforms with only the final two
/// stages remaining. Output covers rows * size values and is disjoint from both inputs.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fft_radix4_extract(
  real: ReadBuffer<f32>,
  imag: ReadBuffer<f32>,
  n: u32,
  output: WriteBuffer<f32>,
  size: u32,
  scale: f32,
  rows: u32,
  parity: u32,
  quads: u32,
) {
  let config = Readout {
    size,
    scale,
    rows,
    parity,
  };
  let tid = crate::index();
  if tid < quads && n >= 4 {
    let half = n / 4;
    let batch = tid / half;
    let pos = tid % half;
    let i0 = batch * n + pos;
    let i1 = i0 + half;
    let i2 = i1 + half;
    let i3 = i2 + half;
    // SAFETY: every lane owns four positions in one transform and disjoint output cells.
    unsafe {
      let a0 = Complex {
        re: real.load_unchecked(i0),
        im: imag.load_unchecked(i0),
      };
      let a1 = Complex {
        re: real.load_unchecked(i1),
        im: imag.load_unchecked(i1),
      };
      let a2 = Complex {
        re: real.load_unchecked(i2),
        im: imag.load_unchecked(i2),
      };
      let a3 = Complex {
        re: real.load_unchecked(i3),
        im: imag.load_unchecked(i3),
      };
      let result = radix4(a0, a1, a2, a3, pos, half);
      put(output, config, batch, pos, result[0]);
      put(output, config, batch, pos + half, result[1]);
      put(output, config, batch, pos + half * 2, result[2]);
      put(output, config, batch, pos + half * 3, result[3]);
    }
  }
}

/// # Safety
/// Inputs cover 2 * butterflies values in complete length-n transforms, with
/// only the final stage remaining. Output covers rows * size values and is disjoint
/// from both inputs.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fft_butterfly_extract(
  real: ReadBuffer<f32>,
  imag: ReadBuffer<f32>,
  n: u32,
  output: WriteBuffer<f32>,
  size: u32,
  scale: f32,
  rows: u32,
  parity: u32,
  butterflies: u32,
) {
  let config = Readout {
    size,
    scale,
    rows,
    parity,
  };
  let tid = crate::index();
  if tid < butterflies && n >= 2 {
    let half = n / 2;
    let batch = tid / half;
    let pos = tid % half;
    let i = batch * n + pos;
    let j = i + half;
    // SAFETY: each lane owns a pair of positions in a complete transform.
    unsafe {
      let a = Complex {
        re: real.load_unchecked(i),
        im: imag.load_unchecked(i),
      };
      let b = Complex {
        re: real.load_unchecked(j),
        im: imag.load_unchecked(j),
      };
      let t = b.mul(twiddle(pos, n));
      put(output, config, batch, pos, a.add(t));
      put(output, config, batch, pos + half, a.sub(t));
    }
  }
}
