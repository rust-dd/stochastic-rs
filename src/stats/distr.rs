use rand::Rng;

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
pub mod weibull;

/// Fills a slice with random floating-point values in the range [0, 1).
fn fill_f32_zero_one<R: Rng + ?Sized>(rng: &mut R, out: &mut [f32]) {
  for x in out.iter_mut() {
    *x = rng.gen_range(0.0..1.0);
  }
}

#[cfg(test)]
mod tests {
  use plotly::layout::LayoutGrid;
  use rand::thread_rng;
  use rand_distr::Distribution;

  use crate::stats::distr::{
    beta::SimdBeta, binomial::SimdBinomial, cauchy::SimdCauchy, gamma::SimdGamma,
    geometric::SimdGeometric, hypergeometric::SimdHypergeometric, inverse_gauss::SimdInverseGauss,
    lognormal::SimdLogNormal, normal::SimdNormal, normal_inverse_gauss::SimdNormalInverseGauss,
    pareto::SimdPareto, poisson::SimdPoisson, studentt::SimdStudentT, weibull::SimdWeibull,
  };

  use plotly::{
    common::{Line, LineShape, Mode},
    layout::GridPattern,
    Layout, Plot, Scatter,
  };

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
    // Create a 4x4 grid for 14 distributions (2 leftover)
    let mut plot = Plot::new();
    plot.set_layout(
      Layout::new().grid(
        LayoutGrid::new()
          .rows(4)
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
      let mut rng = thread_rng();
      let dist = SimdNormal::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, -4.0, 4.0);
      let trace = Scatter::new(xs, bins)
        .name("Normal(0,1)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 2) Cauchy => subplot (1,2)
    {
      let (xa, ya) = subplot_axes(1, 2);
      let mut rng = thread_rng();
      let dist = SimdCauchy::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, -10.0, 10.0);
      let trace = Scatter::new(xs, bins)
        .name("Cauchy(0,1)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 3) LogNormal => (1,3)
    {
      let (xa, ya) = subplot_axes(1, 3);
      let mut rng = thread_rng();
      let dist = SimdLogNormal::new(0.0, 1.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 8.0);
      let trace = Scatter::new(xs, bins)
        .name("LogNormal(0,1)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 4) Pareto => (1,4)
    {
      let (xa, ya) = subplot_axes(1, 4);
      let mut rng = thread_rng();
      let dist = SimdPareto::new(1.0, 1.5);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 10.0);
      let trace = Scatter::new(xs, bins)
        .name("Pareto(x_m=1,alpha=1.5)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 5) Weibull => (2,1)
    {
      let (xa, ya) = subplot_axes(2, 1);
      let mut rng = thread_rng();
      let dist = SimdWeibull::new(1.0, 1.5);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 3.0);
      let trace = Scatter::new(xs, bins)
        .name("Weibull(lambda=1,k=1.5)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 6) Gamma => (2,2)
    {
      let (xa, ya) = subplot_axes(2, 2);
      let mut rng = thread_rng();
      let dist = SimdGamma::new(2.0, 2.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 120, 0.0, 20.0);
      let trace = Scatter::new(xs, bins)
        .name("Gamma(alpha=2,scale=2)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 7) Beta => (2,3)
    {
      let (xa, ya) = subplot_axes(2, 3);
      let mut rng = thread_rng();
      let dist = SimdBeta::new(2.0, 2.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 1.0);
      let trace = Scatter::new(xs, bins)
        .name("Beta(2,2)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 8) Inverse Gaussian => (2,4)
    {
      let (xa, ya) = subplot_axes(2, 4);
      let mut rng = thread_rng();
      let dist = SimdInverseGauss::new(1.0, 2.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, 0.0, 3.0);
      let trace = Scatter::new(xs, bins)
        .name("InverseGauss(mu=1,lambda=2)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 9) Normal-Inverse Gauss => (3,1)
    {
      let (xa, ya) = subplot_axes(3, 1);
      let mut rng = thread_rng();
      let dist = SimdNormalInverseGauss::new(2.0, 0.0, 1.0, 0.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 100, -3.0, 3.0);
      let trace = Scatter::new(xs, bins)
        .name("NIG(alpha=2,beta=0)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 10) StudentT => (3,2)
    {
      let (xa, ya) = subplot_axes(3, 2);
      let mut rng = thread_rng();
      let dist = SimdStudentT::new(5.0);
      let samples: Vec<f32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let (xs, bins) = make_histogram(&samples, 120, -5.0, 5.0);
      let trace = Scatter::new(xs, bins)
        .name("StudentT(df=5)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 11) Binomial => (3,3) (discrete)
    {
      let (xa, ya) = subplot_axes(3, 3);
      let mut rng = thread_rng();
      let dist = SimdBinomial::new(10, 0.3);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      // we do 0..10 pmf
      let mut bins = vec![0.0; 11];
      for &val in &samples {
        if val <= 10 {
          bins[val as usize] += 1.0;
        }
      }
      let total = samples.len() as f32;
      for b in bins.iter_mut() {
        *b /= total;
      }
      let xs: Vec<f32> = (0..=10).map(|i| i as f32).collect();

      let trace = Scatter::new(xs, bins)
        .name("Binomial(10,0.3)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 12) Geometric => (3,4)
    {
      let (xa, ya) = subplot_axes(3, 4);
      let mut rng = thread_rng();
      let dist = SimdGeometric::new(0.25);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      // let's do 0..20
      let max_range = 20;
      let mut bins = vec![0.0; max_range + 1];
      for &val in &samples {
        if val as usize <= max_range {
          bins[val as usize] += 1.0;
        }
      }
      let total = samples.len() as f32;
      for b in bins.iter_mut() {
        *b /= total;
      }
      let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
      let trace = Scatter::new(xs, bins)
        .name("Geometric(p=0.25)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 13) HyperGeometric => (4,1)
    {
      let (xa, ya) = subplot_axes(4, 1);
      let mut rng = thread_rng();
      // N=20, K=5, n=6
      let dist = SimdHypergeometric::new(20, 5, 6);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();
      let mut bins = vec![0.0; 7];
      for &val in &samples {
        if val <= 6 {
          bins[val as usize] += 1.0;
        }
      }
      let total = samples.len() as f32;
      for b in bins.iter_mut() {
        *b /= total;
      }
      let xs: Vec<f32> = (0..=6).map(|i| i as f32).collect();

      let trace = Scatter::new(xs, bins)
        .name("HyperGeo(N=20,K=5,n=6)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    // 14) Poisson => (4,2)
    {
      let (xa, ya) = subplot_axes(4, 2);
      let mut rng = thread_rng();
      let dist = SimdPoisson::new(4.0);
      let samples: Vec<u32> = (0..sample_size).map(|_| dist.sample(&mut rng)).collect();

      let max_range = 15;
      let mut bins = vec![0.0; max_range + 1];
      for &val in &samples {
        if val <= max_range as u32 {
          bins[val as usize] += 1.0;
        }
      }
      let total = samples.len() as f32;
      for b in bins.iter_mut() {
        *b /= total;
      }
      let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
      let trace = Scatter::new(xs, bins)
        .name("Poisson(lambda=4)")
        .mode(Mode::Lines)
        .line(Line::new().shape(LineShape::Linear))
        .x_axis(&xa)
        .y_axis(&ya);
      plot.add_trace(trace);
    }

    plot.show();
  }
}
