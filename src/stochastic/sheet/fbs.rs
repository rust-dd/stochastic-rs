use impl_new_derive::ImplNew;
use ndarray::linalg::kron;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_rand::RandomExt;
use ndrustfft::ndfft;
use ndrustfft::FftHandler;
use num_complex::Complex;

use crate::distributions::complex::ComplexDistribution;
use crate::distributions::normal::SimdNormal;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(ImplNew)]
pub struct FBS<T: Float> {
  pub hurst: T,
  pub m: usize,
  pub n: usize,
  pub r: T,
}

impl<T: Float> Process<T> for FBS<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Array2<T> {
    let (m, n, r) = (self.m, self.n, self.r);
    let alpha = T::from_usize(2).unwrap() * self.hurst;

    let tx = Array1::linspace(r / T::from_usize(n).unwrap(), r, n);
    let ty = Array1::linspace(r / T::from_usize(m).unwrap(), r, m);

    let mut cov = Array2::<T>::zeros((m, n));
    for i in 0..n {
      for j in 0..m {
        cov[[j, i]] = Self::rho((tx[i], ty[j]), (tx[0], ty[0]), r, alpha).0;
      }
    }

    let big_m = 2 * (m - 1);
    let big_n = 2 * (n - 1);
    let mut blk = Array2::<T>::zeros((big_m, big_n));

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

    let scale =
      T::from_usize(4).unwrap() * T::from_usize(m - 1).unwrap() * T::from_usize(n - 1).unwrap();
    let mut fft_handler0 = FftHandler::<T>::new(big_m);
    let mut fft_handler1 = FftHandler::<T>::new(big_n);

    let blk_c = blk.mapv(|v| Complex::new(v, T::zero()));
    let mut fft_tmp = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&blk_c, &mut fft_tmp, &mut fft_handler0, 0);
    let mut fft_freq = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&fft_tmp, &mut fft_freq, &mut fft_handler1, 1);

    let lam = fft_freq.mapv(|c| (c.re / scale).max(T::zero()).sqrt());

    let z: Array2<Complex<T>> = Array2::random(
      (big_m, big_n),
      ComplexDistribution::new(
        SimdNormal::<T, 64>::new(T::zero(), T::one()),
        SimdNormal::<T, 64>::new(T::zero(), T::one()),
      ),
    );

    let prod = lam.mapv(|v| Complex::new(v, T::zero())) * z;
    let mut fft_tmp2 = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&prod, &mut fft_tmp2, &mut fft_handler0, 0);
    let mut result = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&fft_tmp2, &mut result, &mut fft_handler1, 1);

    let mut field = Array2::<T>::zeros((m, n));
    for i in 0..m {
      for j in 0..n {
        field[[i, j]] = result[[i, j]].re;
      }
    }

    let (_, _, c2) = Self::rho((T::zero(), T::zero()), (T::zero(), T::zero()), r, alpha);

    let shift = field[[0, 0]];
    field.mapv_inplace(|v| v - shift);

    let z1 = T::normal_array(1, T::zero(), T::one())[0];
    let z2 = T::normal_array(1, T::zero(), T::one())[0];

    let ty_scaled = &ty * z1;
    let tx_scaled = &tx * z2;
    let ty_mat = ty_scaled.insert_axis(Axis(1));
    let tx_mat = tx_scaled.insert_axis(Axis(0));

    let correction = kron(&ty_mat, &tx_mat) * (T::from_usize(2).unwrap() * c2).sqrt();
    field = &field + &correction;

    field
  }
}

impl<T: Float> FBS<T> {
  fn rho(x: (T, T), y: (T, T), r: T, alpha: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from_usize(2).unwrap();
    let three = T::from_usize(3).unwrap();
    let half = one / two;
    let one_point_five = three * half;

    let (beta, c2, c0) = if alpha <= one_point_five {
      let c2 = alpha * half;
      let c0 = one - alpha * half;
      (T::zero(), c2, c0)
    } else {
      let beta = alpha * (two - alpha) / (three * r * (r * r - one));
      let c2 = (alpha - beta * (r - one).powi(2) * (r + two)) * half;
      let c0 = beta * (r - one).powi(3) + one - c2;
      (beta, c2, c0)
    };

    let dx = x.0 - y.0;
    let dy = x.1 - y.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let out = if dist <= one {
      c0 - dist.powf(alpha) + c2 * dist * dist
    } else if dist <= r {
      beta * (r - dist).powi(3) / dist
    } else {
      T::zero()
    };
    (out, c0, c2)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::s;
  use plotly::surface::PlaneContours;
  use plotly::Layout;
  use plotly::Plot;
  use plotly::Surface;

  use super::FBS;
  use crate::stochastic::Process;

  #[test]
  fn test_fbs_plot() {
    let m = 1000;
    let n = 1000;
    let hurst = 0.7;
    let r = 2.0;

    let half_m = m / 2;
    let half_n = n / 2;

    let fbs: FBS<f64> = FBS::new(hurst, m, n, r);
    let sheet = fbs.sample();

    let x: Vec<f64> = (1..=half_n).map(|i| i as f64 * r / n as f64).collect();
    let y: Vec<f64> = (1..=half_m).map(|j| j as f64 * r / m as f64).collect();

    let mut masked_half = sheet.slice(s![..half_m, ..half_n]).to_owned();

    for (i, yi) in y.iter().enumerate() {
      for (j, xj) in x.iter().enumerate() {
        if xj.powi(2) + yi.powi(2) > 1.0 {
          masked_half[[i, j]] = f64::NAN;
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
