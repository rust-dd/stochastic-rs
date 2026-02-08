use rand::Rng;
use wide::f32x8;
use wide::f64x8;

pub mod beta;
pub mod binomial;
pub mod cauchy;
pub mod chi_square;
pub mod exp;
pub mod gamma;
pub mod geometric;
pub mod hypergeometric;
pub mod inverse_gauss;
pub mod lognormal;
pub mod normal;
pub mod normal_inverse_gauss;
pub mod pareto;
pub mod poisson;
pub mod studentt;
pub mod uniform;
pub mod weibull;

/// Fills a slice with random floating-point values in the range [0, 1).
fn fill_f32_zero_one<R: Rng + ?Sized>(rng: &mut R, out: &mut [f32]) {
  for x in out.iter_mut() {
    *x = rng.random_range(0.0..1.0);
  }
}

fn fill_f64_zero_one<R: Rng + ?Sized>(rng: &mut R, out: &mut [f64]) {
  for x in out.iter_mut() {
    *x = rng.random_range(0.0..1.0);
  }
}

pub trait SimdFloat: num_traits::Float + Default + Send + Sync + 'static {
  type Simd: Copy
    + std::ops::Mul<Output = Self::Simd>
    + std::ops::Add<Output = Self::Simd>
    + std::ops::Sub<Output = Self::Simd>
    + std::ops::Div<Output = Self::Simd>
    + std::ops::Neg<Output = Self::Simd>;

  fn splat(val: Self) -> Self::Simd;
  fn simd_from_array(arr: [Self; 8]) -> Self::Simd;
  fn simd_to_array(v: Self::Simd) -> [Self; 8];
  fn simd_ln(v: Self::Simd) -> Self::Simd;
  fn simd_sqrt(v: Self::Simd) -> Self::Simd;
  fn simd_cos(v: Self::Simd) -> Self::Simd;
  fn simd_sin(v: Self::Simd) -> Self::Simd;
  fn simd_exp(v: Self::Simd) -> Self::Simd;
  fn simd_tan(v: Self::Simd) -> Self::Simd;
  fn simd_max(a: Self::Simd, b: Self::Simd) -> Self::Simd;
  fn simd_powf(v: Self::Simd, exp: Self) -> Self::Simd;
  fn simd_floor(v: Self::Simd) -> Self::Simd;
  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [Self]);
  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self;
  fn simd_from_i32x8(v: wide::i32x8) -> Self::Simd;
  fn pi() -> Self;
  fn two_pi() -> Self;
  fn min_positive_val() -> Self;
}

impl SimdFloat for f32 {
  type Simd = f32x8;

  fn splat(val: f32) -> f32x8 {
    f32x8::splat(val)
  }

  fn simd_from_array(arr: [f32; 8]) -> f32x8 {
    f32x8::from(arr)
  }

  fn simd_to_array(v: f32x8) -> [f32; 8] {
    v.to_array()
  }

  fn simd_ln(v: f32x8) -> f32x8 {
    v.ln()
  }

  fn simd_sqrt(v: f32x8) -> f32x8 {
    v.sqrt()
  }

  fn simd_cos(v: f32x8) -> f32x8 {
    v.cos()
  }

  fn simd_sin(v: f32x8) -> f32x8 {
    v.sin()
  }

  fn simd_exp(v: f32x8) -> f32x8 {
    v.exp()
  }

  fn simd_tan(v: f32x8) -> f32x8 {
    v.tan()
  }

  fn simd_max(a: f32x8, b: f32x8) -> f32x8 {
    a.max(b)
  }

  fn simd_powf(v: f32x8, exp: f32) -> f32x8 {
    v.powf(exp)
  }

  fn simd_floor(v: f32x8) -> f32x8 {
    v.floor()
  }

  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [f32]) {
    fill_f32_zero_one(rng, out)
  }

  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    rng.random_range(0.0f32..1.0f32)
  }

  fn simd_from_i32x8(v: wide::i32x8) -> f32x8 {
    v.round_float()
  }

  fn pi() -> f32 {
    std::f32::consts::PI
  }

  fn two_pi() -> f32 {
    2.0 * std::f32::consts::PI
  }

  fn min_positive_val() -> f32 {
    f32::MIN_POSITIVE
  }
}

impl SimdFloat for f64 {
  type Simd = f64x8;

  fn splat(val: f64) -> f64x8 {
    f64x8::splat(val)
  }

  fn simd_from_array(arr: [f64; 8]) -> f64x8 {
    f64x8::from(arr)
  }

  fn simd_to_array(v: f64x8) -> [f64; 8] {
    v.to_array()
  }

  fn simd_ln(v: f64x8) -> f64x8 {
    v.ln()
  }

  fn simd_sqrt(v: f64x8) -> f64x8 {
    v.sqrt()
  }

  fn simd_cos(v: f64x8) -> f64x8 {
    v.cos()
  }

  fn simd_sin(v: f64x8) -> f64x8 {
    v.sin()
  }

  fn simd_exp(v: f64x8) -> f64x8 {
    v.exp()
  }

  fn simd_tan(v: f64x8) -> f64x8 {
    v.tan()
  }

  fn simd_max(a: f64x8, b: f64x8) -> f64x8 {
    a.max(b)
  }

  fn simd_powf(v: f64x8, exp: f64) -> f64x8 {
    v.powf(exp)
  }

  fn simd_floor(v: f64x8) -> f64x8 {
    v.floor()
  }

  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [f64]) {
    fill_f64_zero_one(rng, out)
  }

  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rng.random_range(0.0f64..1.0f64)
  }

  fn simd_from_i32x8(v: wide::i32x8) -> f64x8 {
    f64x8::from_i32x8(v)
  }

  fn pi() -> f64 {
    std::f64::consts::PI
  }

  fn two_pi() -> f64 {
    2.0 * std::f64::consts::PI
  }

  fn min_positive_val() -> f64 {
    f64::MIN_POSITIVE
  }
}

#[cfg(test)]
mod tests {
  use plotly::common::Line;
  use plotly::common::LineShape;
  use plotly::common::Mode;
  use plotly::layout::GridPattern;
  use plotly::layout::LayoutGrid;
  use plotly::Layout;
  use plotly::Plot;
  use plotly::Scatter;
  use rand::rng;
  use rand_distr::Distribution;

  use crate::distributions::beta::SimdBeta;
  use crate::distributions::binomial::SimdBinomial;
  use crate::distributions::cauchy::SimdCauchy;
  use crate::distributions::chi_square::SimdChiSquared;
  use crate::distributions::exp::SimdExp;
  use crate::distributions::gamma::SimdGamma;
  use crate::distributions::geometric::SimdGeometric;
  use crate::distributions::hypergeometric::SimdHypergeometric;
  use crate::distributions::inverse_gauss::SimdInverseGauss;
  use crate::distributions::lognormal::SimdLogNormal;
  use crate::distributions::normal::SimdNormal;
  use crate::distributions::normal_inverse_gauss::SimdNormalInverseGauss;
  use crate::distributions::pareto::SimdPareto;
  use crate::distributions::poisson::SimdPoisson;
  use crate::distributions::studentt::SimdStudentT;
  use crate::distributions::uniform::SimdUniform;
  use crate::distributions::weibull::SimdWeibull;

  /// A small helper to create a PDF-like histogram for continuous data in [min_x, max_x].
  fn make_histogram(
    samples: &[f32],
    bins_count: usize,
    min_x: f32,
    max_x: f32,
  ) -> (Vec<f32>, Vec<f32>) {
    let bin_width = (max_x - min_x) / bins_count as f32;
    let mut bins = vec![0.0; bins_count + 1];
    for &val in samples {
      if val >= min_x && val < max_x {
        let idx = ((val - min_x) / bin_width) as usize;
        bins[idx] += 1.0;
      }
    }
    let total = samples.len() as f32;
    // convert to PDF-like
    for b in bins.iter_mut() {
      *b /= total * bin_width;
    }
    let xs: Vec<f32> = (0..bins_count)
      .map(|i| min_x + (i as f32 + 0.5) * bin_width)
      .collect();
    (xs, bins)
  }

  /// A small helper for discrete data: we just do a 0..=max_range “histogram”
  fn make_discrete_pmf(samples: &[u32], max_range: u32) -> (Vec<f32>, Vec<f32>) {
    let mut bins = vec![0.0; (max_range + 1) as usize];
    for &val in samples {
      if val <= max_range {
        bins[val as usize] += 1.0;
      }
    }
    let n = samples.len() as f32;
    for b in bins.iter_mut() {
      *b /= n;
    }
    let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
    (xs, bins)
  }

  /// We’ll place each distribution in a 4x4 grid => up to 16 subplots.
  /// Let’s define a small function to map (row,col) -> "xN", "yN".
  /// row, col are in [1..4].
  fn subplot_axes(row: usize, col: usize) -> (String, String) {
    // Subplot index = (row-1)*4 + col
    let index = (row - 1) * 4 + col;
    let xaxis = format!("x{}", index);
    let yaxis = format!("y{}", index);
    (xaxis, yaxis)
  }

  #[test]
  fn combined_all_distributions() {
    // Create a 5x4 grid for 17 distributions
    let mut plot = Plot::new();
    plot.set_layout(
      Layout::new().grid(
        LayoutGrid::new()
          .rows(5)
          .columns(4)
          .pattern(GridPattern::Independent),
      ),
    );

    // We'll store a large number of samples for each distribution:
    // typically 50k or 100k.
    let sample_size = 50_000;

    // 1) Normal => subplot (row=1, col=1)
    {
      let (xa, ya) = subplot_axes(1, 1);
      let mut rng = rand::rng();
      let dist = SimdNormal::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, -4.0, 4.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Normal(0,1) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, -4.0, 4.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Normal(0,1) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 2) Cauchy => subplot (1,2)
    {
      let (xa, ya) = subplot_axes(1, 2);
      let mut rng = rand::rng();
      let dist = SimdCauchy::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, -10.0, 10.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Cauchy(0,1) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, -10.0, 10.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Cauchy(0,1) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 3) LogNormal => (1,3)
    {
      let (xa, ya) = subplot_axes(1, 3);
      let mut rng = rand::rng();
      let dist = SimdLogNormal::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 8.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("LogNormal(0,1) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::LogNormal::<f32>::new(0.0, 1.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 8.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("LogNormal(0,1) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 4) Pareto => (1,4)
    {
      let (xa, ya) = subplot_axes(1, 4);
      let mut rng = rand::rng();
      let dist = SimdPareto::new(1.0, 1.5);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 10.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Pareto(1,1.5) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Pareto::<f32>::new(1.0, 1.5).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 10.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Pareto(1,1.5) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 5) Weibull => (2,1)
    {
      let (xa, ya) = subplot_axes(2, 1);
      let mut rng = rand::rng();
      let dist = SimdWeibull::new(1.0, 1.5);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 3.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Weibull(1,1.5) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 3.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Weibull(1,1.5) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 6) Gamma => (2,2)
    {
      let (xa, ya) = subplot_axes(2, 2);
      let mut rng = rand::rng();
      let dist = SimdGamma::new(2.0, 2.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 120, 0.0, 20.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Gamma(2,2) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 120, 0.0, 20.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Gamma(2,2) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 7) Beta => (2,3)
    {
      let (xa, ya) = subplot_axes(2, 3);
      let mut rng = rand::rng();
      let dist = SimdBeta::new(2.0, 2.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 1.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Beta(2,2) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 1.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Beta(2,2) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 8) Inverse Gaussian => (2,4)
    {
      let (xa, ya) = subplot_axes(2, 4);
      let mut rng = rng();
      let dist = SimdInverseGauss::new(1.0, 2.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 3.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("InverseGauss(1,2) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison (InverseGaussian in rand_distr)
      let mut rng = rand::rng();
      let rd = rand_distr::InverseGaussian::<f32>::new(1.0, 2.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 3.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("InverseGauss(1,2) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 9) Normal-Inverse Gauss => (3,1)
    {
      let (xa, ya) = subplot_axes(3, 1);
      let mut rng = rand::rng();
      let dist = SimdNormalInverseGauss::new(2.0, 0.0, 1.0, 0.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, -3.0, 3.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("NIG(2,0) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr has NormalInverseGaussian
      let mut rng = rand::rng();
      let rd = rand_distr::NormalInverseGaussian::<f32>::new(2.0, 0.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, -3.0, 3.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("NIG(2,0) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 10) StudentT => (3,2)
    {
      let (xa, ya) = subplot_axes(3, 2);
      let mut rng = rng();
      let dist = SimdStudentT::new(5.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 120, -5.0, 5.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("StudentT(5) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::StudentT::<f32>::new(5.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 120, -5.0, 5.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("StudentT(5) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 11) Binomial => (3,3) (discrete)
    {
      let (xa, ya) = subplot_axes(3, 3);
      let mut rng = rand::rng();
      let dist = SimdBinomial::new(10, 0.3);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_discrete_pmf(&samples, 10);

      let trace = Scatter::new(xs.clone(), bins)
        .name("Binomial(10,0.3) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Binomial::new(10, 0.3).unwrap();
      let samples_rd: Vec<u32> = (0..sample_size)
        .map(|_| rd.sample(&mut rng) as u32)
        .collect();
      let (_, bins_rd) = make_discrete_pmf(&samples_rd, 10);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Binomial(10,0.3) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 12) Geometric => (3,4)
    {
      let (xa, ya) = subplot_axes(3, 4);
      let mut rng = rng();
      let dist = SimdGeometric::new(0.25);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_discrete_pmf(&samples, 20);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Geometric(0.25) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Geometric::new(0.25).unwrap();
      let samples_rd: Vec<u32> = (0..sample_size)
        .map(|_| rd.sample(&mut rng) as u32)
        .collect();
      let (_, bins_rd) = make_discrete_pmf(&samples_rd, 20);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Geometric(0.25) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 13) HyperGeometric => (4,1)
    {
      let (xa, ya) = subplot_axes(4, 1);
      let mut rng = rand::rng();
      // N=20, K=5, n=6
      let dist = SimdHypergeometric::new(20, 5, 6);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_discrete_pmf(&samples, 6);

      let trace = Scatter::new(xs.clone(), bins)
        .name("HyperGeo(20,5,6) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Hypergeometric::new(20, 5, 6).unwrap();
      let samples_rd: Vec<u32> = (0..sample_size)
        .map(|_| rd.sample(&mut rng) as u32)
        .collect();
      let (_, bins_rd) = make_discrete_pmf(&samples_rd, 6);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("HyperGeo(20,5,6) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 14) Poisson => (4,2)
    {
      let (xa, ya) = subplot_axes(4, 2);
      let mut rng = rand::rng();
      let dist = SimdPoisson::new(4.0);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_discrete_pmf(&samples, 15);

      let trace = Scatter::new(xs.clone(), bins)
        .name("Poisson(4) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Poisson::<f64>::new(4.0).unwrap();
      let samples_rd: Vec<u32> = (0..sample_size)
        .map(|_| rd.sample(&mut rng) as u32)
        .collect();
      let (_, bins_rd) = make_discrete_pmf(&samples_rd, 15);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Poisson(4) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 15) Uniform (0,1) => (4,3)
    {
      let (xa, ya) = subplot_axes(4, 3);
      let mut rng = rand::rng();
      let dist = SimdUniform::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 1.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Uniform(0,1) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Uniform::<f32>::new(0.0, 1.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 1.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Uniform(0,1) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 16) Exponential => (4,4)
    {
      let (xa, ya) = subplot_axes(4, 4);
      let mut rng = rand::rng();
      let dist = SimdExp::new(1.5);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 4.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("Exp(1.5) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::Exp::<f32>::new(1.5).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 4.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("Exp(1.5) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    // 17) Chi-Squared => (5,1)
    {
      let (xa, ya) = subplot_axes(5, 1);
      let mut rng = rand::rng();
      let dist = SimdChiSquared::new(5.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 20.0);
      let trace = Scatter::new(xs.clone(), bins)
        .name("ChiSquared(5) - SIMD")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);

      // rand_distr comparison
      let mut rng = rand::rng();
      let rd = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
      let samples_rd: Vec<f32> = (0..sample_size).map(|_| rd.sample(&mut rng)).collect();
      let (_, bins_rd) = make_histogram(&samples_rd, 100, 0.0, 20.0);
      let trace_rd = Scatter::new(xs, bins_rd)
        .name("ChiSquared(5) - rand_distr")
        .mode(Mode::Lines)
        .line(
          Line::new()
            .shape(LineShape::Linear)
            .dash(plotly::common::DashType::Dash),
        )
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace_rd);
    }

    plot.show();
  }

  // ========== Benchmarks: compare SIMD vs rand_distr ==========
  use std::time::Instant;

  #[test]
  fn bench_normal_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdNormal::new(0.0, 1.0);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "Normal single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_lognormal_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdLogNormal::new(0.2, 0.8);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "LogNormal single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_exp_simd_vs_rand() {
    let n = 10_000_000usize;
    let lambda = 1.5f32;

    let mut rng = rand::rng();
    let simd = SimdExp::new(lambda);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Exp::<f32>::new(lambda).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "Exp single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_cauchy_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdCauchy::new(0.0, 1.0);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "Cauchy single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_gamma_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdGamma::new(2.0, 2.0);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "Gamma single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_weibull_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdWeibull::new(1.0, 1.5);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "Weibull single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_beta_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdBeta::new(2.0, 2.0);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "Beta single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_chisq_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdChiSquared::new(5.0);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "ChiSq single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_studentt_simd_vs_rand() {
    let n = 10_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdStudentT::new(5.0);
    let mut s_sum = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng);
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::StudentT::<f32>::new(5.0).unwrap();
    let mut r_sum = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng);
    }
    let dt_r = t1.elapsed();

    println!(
      "StudentT single: simd {:?}, sum={:.3} | rand_distr {:?}, sum={:.3}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(!s_sum.is_nan() && !r_sum.is_nan());
  }

  #[test]
  fn bench_poisson_simd_vs_rand() {
    let n = 5_000_000usize;

    let mut rng = rand::rng();
    let simd = SimdPoisson::<u32>::new(4.0);
    let mut s_sum: u64 = 0;
    let t0 = Instant::now();
    for _ in 0..n {
      s_sum += simd.sample(&mut rng) as u64;
    }
    let dt_s = t0.elapsed();

    let mut rng = rand::rng();
    let rd = rand_distr::Poisson::<f64>::new(4.0).unwrap();
    let mut r_sum: u64 = 0;
    let t1 = Instant::now();
    for _ in 0..n {
      r_sum += rd.sample(&mut rng) as u64;
    }
    let dt_r = t1.elapsed();

    println!(
      "Poisson single: simd {:?}, sum={} | rand_distr {:?}, sum={}",
      dt_s, s_sum, dt_r, r_sum
    );
    assert!(s_sum > 0 && r_sum > 0);
  }

  // Helpers for timing benchmarks
  struct Row {
    name: &'static str,
    simd_ms: f64,
    rand_ms: f64,
  }
  fn time_f32<F1, F2>(
    rows: &mut Vec<Row>,
    n: usize,
    name: &'static str,
    mut simd_fn: F1,
    mut rand_fn: F2,
  ) where
    F1: FnMut() -> f32,
    F2: FnMut() -> f32,
  {
    use std::hint::black_box;
    let t0 = Instant::now();
    let mut s_sum = 0.0f32;
    for _ in 0..n {
      s_sum += simd_fn();
    }
    let dt_simd = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut r_sum = 0.0f32;
    for _ in 0..n {
      r_sum += rand_fn();
    }
    let dt_rand = t1.elapsed().as_secs_f64() * 1000.0;

    black_box(s_sum);
    black_box(r_sum);
    rows.push(Row {
      name,
      simd_ms: dt_simd,
      rand_ms: dt_rand,
    });
  }
  fn time_u32<F1, F2>(
    rows: &mut Vec<Row>,
    n: usize,
    name: &'static str,
    mut simd_fn: F1,
    mut rand_fn: F2,
  ) where
    F1: FnMut() -> u32,
    F2: FnMut() -> u32,
  {
    use std::hint::black_box;
    let t0 = Instant::now();
    let mut s_sum: u64 = 0;
    for _ in 0..n {
      s_sum += simd_fn() as u64;
    }
    let dt_simd = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut r_sum: u64 = 0;
    for _ in 0..n {
      r_sum += rand_fn() as u64;
    }
    let dt_rand = t1.elapsed().as_secs_f64() * 1000.0;

    black_box(s_sum);
    black_box(r_sum);
    rows.push(Row {
      name,
      simd_ms: dt_simd,
      rand_ms: dt_rand,
    });
  }

  #[test]
  fn bench_summary_table() {
    let n_f = 5_000_000usize; // samples for continuous/f32
    let n_i = 5_000_000usize; // samples for discrete/u32

    let mut rows: Vec<Row> = Vec::new();

    // Normal
    {
      let mut rng = rand::rng();
      let simd = SimdNormal::new(0.0, 1.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "Normal",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // LogNormal
    {
      let mut rng = rand::rng();
      let simd = SimdLogNormal::new(0.2, 0.8);
      let mut rng2 = rand::rng();
      let rd = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "LogNormal",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Exp
    {
      let mut rng = rand::rng();
      let simd = SimdExp::new(1.5);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Exp::<f32>::new(1.5).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "Exp",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Cauchy
    {
      let mut rng = rand::rng();
      let simd = SimdCauchy::new(0.0, 1.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "Cauchy",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Gamma
    {
      let mut rng = rand::rng();
      let simd = SimdGamma::new(2.0, 2.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "Gamma",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Weibull
    {
      let mut rng = rand::rng();
      let simd = SimdWeibull::new(1.0, 1.5);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "Weibull",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Beta
    {
      let mut rng = rand::rng();
      let simd = SimdBeta::new(2.0, 2.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "Beta",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Chi-Squared
    {
      let mut rng = rand::rng();
      let simd = SimdChiSquared::new(5.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "ChiSquared",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // StudentT
    {
      let mut rng = rand::rng();
      let simd = SimdStudentT::new(5.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::StudentT::<f32>::new(5.0).unwrap();
      time_f32(
        &mut rows,
        n_f,
        "StudentT",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2),
      );
    }

    // Poisson (discrete)
    {
      let mut rng = rand::rng();
      let simd = SimdPoisson::new(4.0);
      let mut rng2 = rand::rng();
      let rd = rand_distr::Poisson::<f64>::new(4.0).unwrap();
      time_u32(
        &mut rows,
        n_i,
        "Poisson",
        || simd.sample(&mut rng),
        || rd.sample(&mut rng2) as u32,
      );
    }

    // Optionally Pareto if available in rand_distr
    #[allow(unused)]
    {
      // If rand_distr had Pareto<f32>, uncomment below lines.
      let _ = SimdPareto::new(1.0, 1.5);
    }

    // Print table
    println!(
      "{:<14} {:>12} {:>14}",
      "Distribution", "simd (ms)", "rand_distr (ms)"
    );
    println!("{:-<14} {:-<12} {:-<14}", "", "", "");
    for r in rows {
      println!("{:<14} {:>12.2} {:>14.2}", r.name, r.simd_ms, r.rand_ms);
    }
  }
}
