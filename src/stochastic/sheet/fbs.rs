use impl_new_derive::ImplNew;
use ndarray::{linalg::kron, s, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndrustfft::{ndfft, ndfft_par, FftHandler};
use num_complex::{Complex64, ComplexDistribution};
#[cfg(feature = "f32")]
use num_complex::Complex32;
use rand_distr::StandardNormal;

#[derive(ImplNew)]
pub struct FBS<T> {
  pub hurst: T,
  pub m: usize,
  pub n: usize,
  pub R: T,
}

impl FBS<f64> {
  pub fn sample(&self) -> Array2<f64> {
    let (m, n, H, R) = (self.m, self.n, self.hurst, self.R);
    let alpha = 2.0 * H;

    let tx = Array1::linspace(R / n as f64, R, n);
    let ty = Array1::linspace(R / m as f64, R, m);

    let mut cov = Array2::<f64>::zeros((m, n));
    for i in 0..n {
      for j in 0..m {
        cov[[j, i]] = Self::rho((tx[i], ty[j]), (tx[0], ty[0]), R, alpha).0;
      }
    }

    let big_m = 2 * (m - 1);
    let big_n = 2 * (n - 1);
    let mut blk = Array2::<f64>::zeros((big_m, big_n));

    // top-left block
    blk.slice_mut(s![..m, ..n]).assign(&cov);

    // top-right: flip columns except first
    blk
      .slice_mut(s![..m, n..])
      .assign(&cov.slice(s![.., 1..n - 1;-1]));

    // bottom-left: flip rows except first
    blk
      .slice_mut(s![m.., ..n])
      .assign(&cov.slice(s![1..m - 1;-1, ..]));

    // bottom-right: flip both rows and cols except first
    blk
      .slice_mut(s![m.., n..])
      .assign(&cov.slice(s![1..m - 1, 1..n - 1]).slice(s![..;-1, ..;-1]));

    // 4. Compute eigenvalues via FFT
    let scale = 4.0 * (m - 1) as f64 * (n - 1) as f64;
    let mut fft_handler0 = FftHandler::<f64>::new(big_m);
    let mut fft_handler1 = FftHandler::<f64>::new(big_n);

    let blk_c = blk.mapv(Complex64::from);
    let mut fft_tmp = Array2::<Complex64>::zeros((big_m, big_n));
    ndfft(&blk_c, &mut fft_tmp, &mut fft_handler0, 0);
    let mut fft_freq = Array2::<Complex64>::zeros((big_m, big_n));
    ndfft(&fft_tmp, &mut fft_freq, &mut fft_handler1, 1);

    let lam = fft_freq.mapv(|c| (c.re / scale).max(0.0).sqrt());

    let z = Array2::random(
      (big_m, big_n),
      ComplexDistribution::new(StandardNormal, StandardNormal),
    );

    let mut Z = lam.mapv(Complex64::from) * z;
    let mut inv_tmp = Array2::<Complex64>::zeros((big_m, big_n));
    ndfft_par(&Z, &mut inv_tmp, &mut fft_handler0, 0);
    ndfft_par(&inv_tmp, &mut Z, &mut fft_handler1, 1);

    let mut field = Array2::<f64>::zeros((m, n));
    for i in 0..m {
      for j in 0..n {
        field[[i, j]] = Z[[i, j]].re;
      }
    }

    let (_, _, c2) = Self::rho((0.0, 0.0), (0.0, 0.0), R, alpha);

    let shift = field[[0, 0]];
    field.mapv_inplace(|v| v - shift);

    let rand_x = Array1::<f64>::random(n, StandardNormal);
    let rand_y = Array1::<f64>::random(m, StandardNormal);

    let ty_rand = &ty * &rand_y;
    let tx_rand = &tx * &rand_x;
    let ty_mat = ty_rand.insert_axis(Axis(1)); // (m, 1)
    let tx_mat = tx_rand.insert_axis(Axis(0)); // (1, n)

    let correction = kron(&ty_mat, &tx_mat) * (2.0 * c2).sqrt();
    field = &field + &correction;

    field
  }

  fn rho(x: (f64, f64), y: (f64, f64), R: f64, alpha: f64) -> (f64, f64, f64) {
    let (beta, c2, c0) = if alpha <= 1.5 {
      let c2 = alpha / 2.0;
      let c0 = 1.0 - alpha / 2.0;
      (0.0, c2, c0)
    } else {
      let beta = alpha * (2.0 - alpha) / (3.0 * R * (R * R - 1.0));
      let c2 = (alpha - beta * (R - 1.0).powi(2) * (R + 2.0)) / 2.0;
      let c0 = beta * (R - 1.0).powi(3) + 1.0 - c2;
      (beta, c2, c0)
    };

    let dx = x.0 - y.0;
    let dy = x.1 - y.1;
    let r = (dx * dx + dy * dy).sqrt();
    let out = if r <= 1.0 {
      c0 - r.powf(alpha) + c2 * r * r
    } else if r <= R {
      beta * (R - r).powi(3) / r
    } else {
      0.0
    };
    (out, c0, c2)
  }
}

#[cfg(feature = "f32")]
impl FBS<f32> {
  pub fn sample(&self) -> Array2<f32> {
    let (m, n, H, R) = (self.m, self.n, self.hurst, self.R);
    let alpha = 2.0 * H;

    let tx = Array1::linspace(R / n as f32, R, n);
    let ty = Array1::linspace(R / m as f32, R, m);

    let mut cov = Array2::<f32>::zeros((m, n));
    for i in 0..n {
      for j in 0..m {
        cov[[j, i]] = Self::rho((tx[i], ty[j]), (tx[0], ty[0]), R, alpha).0;
      }
    }

    let big_m = 2 * (m - 1);
    let big_n = 2 * (n - 1);
    let mut blk = Array2::<f32>::zeros((big_m, big_n));

    blk.slice_mut(s![..m, ..n]).assign(&cov);
    blk
      .slice_mut(s![..m, n..])
      .assign(&cov.slice(s![.., 1..n - 1;-1]));
    blk
      .slice_mut(s![m.., ..n])
      .assign(&cov.slice(s![1..m - 1;-1, ..]));
    blk
      .slice_mut(s![m.., n..])
      .assign(&cov.slice(s![1..m - 1, 1..n - 1]).slice(s![..;-1, ..;-1]));

    let scale = 4.0 * (m - 1) as f32 * (n - 1) as f32;
    let mut fft_handler0 = FftHandler::<f32>::new(big_m);
    let mut fft_handler1 = FftHandler::<f32>::new(big_n);

    let blk_c = blk.mapv(Complex32::from);
    let mut fft_tmp = Array2::<Complex32>::zeros((big_m, big_n));
    ndfft(&blk_c, &mut fft_tmp, &mut fft_handler0, 0);
    let mut fft_freq = Array2::<Complex32>::zeros((big_m, big_n));
    ndfft(&fft_tmp, &mut fft_freq, &mut fft_handler1, 1);

    let lam = fft_freq.mapv(|c| (c.re / scale).max(0.0).sqrt());

    let z = Array2::random(
      (big_m, big_n),
      ComplexDistribution::new(StandardNormal, StandardNormal),
    ).mapv(|c: num_complex::Complex<f64>| Complex32::new(c.re as f32, c.im as f32));

    let mut Z = lam.mapv(Complex32::from) * z;
    let mut inv_tmp = Array2::<Complex32>::zeros((big_m, big_n));
    ndfft_par(&Z, &mut inv_tmp, &mut fft_handler0, 0);
    ndfft_par(&inv_tmp, &mut Z, &mut fft_handler1, 1);

    let mut field = Array2::<f32>::zeros((m, n));
    for i in 0..m {
      for j in 0..n {
        field[[i, j]] = Z[[i, j]].re;
      }
    }

    let (_, _, c2) = Self::rho((0.0, 0.0), (0.0, 0.0), R, alpha);

    let shift = field[[0, 0]];
    field.mapv_inplace(|v| v - shift);

    let rand_x = Array1::random(n, StandardNormal).mapv(|x: f64| x as f32);
    let rand_y = Array1::random(m, StandardNormal).mapv(|x: f64| x as f32);

    let ty_rand = &ty * &rand_y;
    let tx_rand = &tx * &rand_x;
    let ty_mat = ty_rand.insert_axis(Axis(1));
    let tx_mat = tx_rand.insert_axis(Axis(0));

    let correction = kron(&ty_mat, &tx_mat) * (2.0 * c2 as f32).sqrt();
    field = &field + &correction;

    field
  }

  fn rho(x: (f32, f32), y: (f32, f32), R: f32, alpha: f32) -> (f32, f32, f32) {
    let (beta, c2, c0) = if alpha <= 1.5 {
      let c2 = alpha / 2.0;
      let c0 = 1.0 - alpha / 2.0;
      (0.0, c2, c0)
    } else {
      let beta = alpha * (2.0 - alpha) / (3.0 * R * (R * R - 1.0));
      let c2 = (alpha - beta * (R - 1.0).powi(2) * (R + 2.0)) / 2.0;
      let c0 = beta * (R - 1.0).powi(3) + 1.0 - c2;
      (beta, c2, c0)
    };

    let dx = x.0 - y.0;
    let dy = x.1 - y.1;
    let r = (dx * dx + dy * dy).sqrt();
    let out = if r <= 1.0 {
      c0 - r.powf(alpha) + c2 * r * r
    } else if r <= R {
      beta * (R - r).powi(3) / r
    } else {
      0.0
    };
    (out, c0, c2)
  }
}

#[cfg(test)]
mod tests {
  use super::FBS;
  use ndarray::s;
  use plotly::{surface::PlaneContours, Layout, Plot, Surface};

  #[test]
  fn test_fbm_plot_matlab_like() {
    let m = 100;
    let n = 100;
    let hurst = 0.8;
    let R = 2.0;

    let half_m = m / 2;
    let half_n = n / 2;

    let fbs = FBS::new(hurst, m, n, R);
    let sheet = fbs.sample();

    let x: Vec<f64> = (1..=half_n).map(|i| i as f64 * R / n as f64).collect();
    let y: Vec<f64> = (1..=half_m).map(|j| j as f64 * R / m as f64).collect();

    let mut masked_half = sheet.slice(s![..half_m, ..half_n]).to_owned();

    for (i, yi) in y.iter().enumerate() {
      for (j, xj) in x.iter().enumerate() {
        if xj.powi(2) + yi.powi(2) > 1.0 {
          masked_half[[i, j]] = f64::NAN; // NaN works as mask
        }
      }
    }

    let z: Vec<Vec<f64>> = masked_half.outer_iter().map(|row| row.to_vec()).collect();

    let surface = Surface::new(z)
      .x(x.clone())
      .y(y.clone())
      .name(&format!("FBS H={}", hurst))
      .color_scale(plotly::common::ColorScale::Palette(
        plotly::common::ColorScalePalette::Hot,
      ))
      .show_scale(true)
      .contours(plotly::surface::SurfaceContours::new().z(PlaneContours::new()));

    let mut plot = Plot::new();
    plot.add_trace(surface);
    plot.set_layout(
      Layout::new()
        .width(800)
        .height(800)
        .title("Fractional Gaussian Field")
        .x_axis(plotly::layout::Axis::new().title("X"))
        .y_axis(plotly::layout::Axis::new().title("Y")),
    );
    plot.show();

    assert_eq!(sheet.shape(), &[m, n]);
  }
}
