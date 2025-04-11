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

  use plotly::{
    common::{Line, LineShape, Mode},
    Plot, Scatter,
  };
  use rand::thread_rng;
  use rand_distr::Distribution;
  use statrs::function::gamma::gamma;

  use crate::stats::distr::{
    beta::SimdBeta, binomial::SimdBinomial, cauchy::SimdCauchy, gamma::SimdGamma,
    geometric::SimdGeometric, hypergeometric::SimdHypergeometric, inverse_gauss::SimdInverseGauss,
    lognormal::SimdLogNormal, normal::SimdNormal, normal_inverse_gauss::SimdNormalInverseGauss,
    pareto::SimdPareto, poisson::SimdPoisson, studentt::SimdStudentT, weibull::SimdWeibull,
  }; // needed for some checks

  /// Helper to build a histogram in [min_x, max_x] with `bins_count` bins.
  /// Returns (xs, bins) where xs are bin centers, bins[i] is PDF-like height.
  fn make_histogram(
    samples: &[f32],
    bins_count: usize,
    min_x: f32,
    max_x: f32,
  ) -> (Vec<f32>, Vec<f32>) {
    let bin_width = (max_x - min_x) / bins_count as f32;
    let mut bins = vec![0.0; bins_count];
    for &val in samples {
      if val >= min_x && val < max_x {
        let idx = ((val - min_x) / bin_width) as usize;
        bins[idx] += 1.0;
      }
    }
    let total = samples.len() as f32;
    // PDF-like normalization
    for b in bins.iter_mut() {
      *b /= total * bin_width;
    }
    // bin centers
    let xs: Vec<f32> = (0..bins_count)
      .map(|i| min_x + (i as f32 + 0.5) * bin_width)
      .collect();
    (xs, bins)
  }

  /// Plots the given xs, bins as a line chart (optional).
  fn plot_pdf_estimate(title: &str, xs: &[f32], bins: &[f32]) {
    let trace = Scatter::new(xs.to_vec(), bins.to_vec())
      .mode(Mode::Lines)
      .line(Line::new().shape(LineShape::Linear))
      .name(title);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();
  }

  //-------------------------------------------------------------------------------
  // Continuous distributions
  //-------------------------------------------------------------------------------

  #[test]
  fn test_normal() {
    let mut rng = thread_rng();
    let normal = SimdNormal::new(0.0, 1.0);

    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let (xs, bins) = make_histogram(&samples, 100, -4.0, 4.0);
    plot_pdf_estimate("SimdNormal PDF", &xs, &bins);

    // Empirical checks
    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    let var_sample = samples
      .iter()
      .map(|&x| (x - mean_sample) * (x - mean_sample))
      .sum::<f32>()
      / (n as f32 - 1.0);

    println!("[Normal] mean: {:.4}, var: {:.4}", mean_sample, var_sample);
    assert!(mean_sample.abs() < 0.05);
    assert!((var_sample - 1.0).abs() < 0.05);
  }

  #[test]
  fn test_cauchy() {
    let mut rng = thread_rng();
    let cauchy = SimdCauchy::new(0.0, 1.0);
    // For Cauchy, we can't rely on mean or variance, so let's just do a histogram
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| cauchy.sample(&mut rng)).collect();
    // We'll pick a wide range, e.g. [-10, 10]
    let (xs, bins) = make_histogram(&samples, 200, -10.0, 10.0);
    plot_pdf_estimate("SimdCauchy PDF", &xs, &bins);

    // Possibly we check that the median is near x0 = 0
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[n / 2];
    println!("[Cauchy] median ~ {}", median);
    assert!(median.abs() < 0.2); // loose check
  }

  #[test]
  fn test_lognormal() {
    let mut rng = thread_rng();
    let logn = SimdLogNormal::new(0.0, 1.0); // lognormal( mu=0, sigma=1 )
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| logn.sample(&mut rng)).collect();
    // Range in [0..maybe 10?]
    let (xs, bins) = make_histogram(&samples, 200, 0.0, 10.0);
    plot_pdf_estimate("SimdLogNormal PDF", &xs, &bins);

    // For Lognormal(0,1), mean = exp(1/2), var = (exp(1)-1)*exp(1).
    let mean_theoretical = (0.5f32).exp();
    let var_theoretical = ((1.0f32).exp() - 1.0) * (1.0f32).exp();

    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    let var_sample = samples
      .iter()
      .map(|&x| (x - mean_sample) * (x - mean_sample))
      .sum::<f32>()
      / (n as f32 - 1.0);

    println!(
      "[LogNormal] mean: {:.4} ~ {:.4}, var: {:.4} ~ {:.4}",
      mean_sample, mean_theoretical, var_sample, var_theoretical
    );
    assert!((mean_sample - mean_theoretical).abs() < 0.2);
    assert!((var_sample - var_theoretical).abs() < 1.0);
  }

  #[test]
  fn test_pareto() {
    let mut rng = thread_rng();
    let pareto = SimdPareto::new(1.0, 1.5);
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| pareto.sample(&mut rng)).collect();

    // We'll do a range [0..10] for the histogram
    let (xs, bins) = make_histogram(&samples, 200, 0.0, 10.0);
    plot_pdf_estimate("SimdPareto PDF", &xs, &bins);

    // If alpha>1, mean is finite: mean = x_m * alpha / (alpha - 1).
    let alpha = 1.5;
    let x_m = 1.0;
    let mean_theoretical = x_m * alpha / (alpha - 1.0); // 1.0*1.5/0.5=3.0
    let mean_sample = samples.iter().sum::<f32>() / n as f32;
    println!(
      "[Pareto] empirical mean: {:.4}, theory: {:.4}",
      mean_sample, mean_theoretical
    );
    // let's just check rough closeness
    assert!((mean_sample - 3.0).abs() < 0.5);
  }

  #[test]
  fn test_weibull() {
    let mut rng = thread_rng();
    let weibull = SimdWeibull::new(1.0, 1.5);
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| weibull.sample(&mut rng)).collect();

    let (xs, bins) = make_histogram(&samples, 200, 0.0, 3.0);
    plot_pdf_estimate("SimdWeibull PDF", &xs, &bins);

    // For Weibull(lambda=1, k=1.5), mean = Gamma(1+1/k), var etc.
    let k = 1.5;
    let mean_theoretical = gamma(1.0 + 1.0 / k); // or approx
    let mean_sample = samples.iter().sum::<f32>() / n as f32;
    println!(
      "[Weibull] empirical mean: {:.4}, theory ~ {:.4}",
      mean_sample, mean_theoretical
    );
  }

  #[test]
  fn test_gamma() {
    let mut rng = thread_rng();
    // alpha=2, scale=2 => mean=4, var=8
    let gamma = SimdGamma::new(2.0, 2.0);
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let (xs, bins) = make_histogram(&samples, 200, 0.0, 20.0);
    plot_pdf_estimate("SimdGamma PDF", &xs, &bins);

    let mean_sample = samples.iter().sum::<f32>() / n as f32;
    let var_sample = samples
      .iter()
      .map(|&x| (x - mean_sample) * (x - mean_sample))
      .sum::<f32>()
      / (n as f32 - 1.0);

    println!(
      "[Gamma(2,2)] mean ~ {:.4}, var~{:.4}",
      mean_sample, var_sample
    );
    // theoretical mean= alpha*scale=4, variance= alpha*scale^2=8
    assert!((mean_sample - 4.0).abs() < 0.5);
    assert!((var_sample - 8.0).abs() < 2.0);
  }

  #[test]
  fn test_beta() {
    let mut rng = thread_rng();
    // alpha=2, beta=2 => mean=0.5
    let beta = SimdBeta::new(2.0, 2.0);
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| beta.sample(&mut rng)).collect();
    let (xs, bins) = make_histogram(&samples, 200, 0.0, 1.0);
    plot_pdf_estimate("SimdBeta PDF", &xs, &bins);

    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    // for Beta(2,2), mean= 2/(2+2)=0.5
    println!("[Beta(2,2)] empirical mean: {:.4}", mean_sample);
    assert!((mean_sample - 0.5).abs() < 0.05);
  }

  #[test]
  fn test_inverse_gauss() {
    let mut rng = thread_rng();
    // pick mu=1, lambda=2
    let ig = SimdInverseGauss::new(1.0, 2.0);
    let n = 50_000;
    let samples: Vec<f32> = (0..n).map(|_| ig.sample(&mut rng)).collect();
    // we'll plot ~ [0..3]
    let (xs, bins) = make_histogram(&samples, 200, 0.0, 3.0);
    plot_pdf_estimate("SimdInverseGauss PDF", &xs, &bins);

    // Just display empirical mean
    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    println!(
      "[InverseGauss mu=1, lambda=2] empirical mean: {:.4}",
      mean_sample
    );
    // Theoretical mean ~ mu=1
    assert!((mean_sample - 1.0).abs() < 0.1);
  }

  #[test]
  fn test_normal_inverse_gauss() {
    let mut rng = thread_rng();
    // alpha=2, beta=0, delta=1, mu=0 => let's do a simpler NIG
    let nig = SimdNormalInverseGauss::new(2.0, 0.0, 1.0, 0.0);
    let n = 50_000;
    let samples: Vec<f32> = (0..n).map(|_| nig.sample(&mut rng)).collect();
    let (xs, bins) = make_histogram(&samples, 200, -3.0, 3.0);
    plot_pdf_estimate("SimdNIG PDF", &xs, &bins);

    // no easy closed-form, just check approximate location near 0
    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    println!("[NIG alpha=2, beta=0] empirical mean: {:.4}", mean_sample);
    assert!(mean_sample.abs() < 0.2);
  }

  #[test]
  fn test_student_t() {
    let mut rng = thread_rng();
    let st = SimdStudentT::new(5.0); // df=5
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| st.sample(&mut rng)).collect();
    let (xs, bins) = make_histogram(&samples, 200, -5.0, 5.0);
    plot_pdf_estimate("SimdStudentT PDF", &xs, &bins);

    // For t-distr with df=5, mean=0, var= 5/(5-2)=5/3 if df>2
    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    let var_sample = samples
      .iter()
      .map(|&x| (x - mean_sample) * (x - mean_sample))
      .sum::<f32>()
      / (n as f32 - 1.0);
    println!(
      "[StudentT df=5] mean ~ {:.4}, var ~ {:.4}",
      mean_sample, var_sample
    );
    assert!(mean_sample.abs() < 0.1);
    // theoretical var= 5/(5-2)= 5/3=1.6667
    assert!((var_sample - 1.6667).abs() < 0.5);
  }

  //-------------------------------------------------------------------------------
  // Discrete distribution tests
  //-------------------------------------------------------------------------------

  #[test]
  fn test_binomial() {
    let mut rng = thread_rng();
    // n=10, p=0.3 => mean=3
    let binom = SimdBinomial::new(10, 0.3);
    let n = 100_000;
    let samples: Vec<u32> = (0..n).map(|_| binom.sample(&mut rng)).collect();

    // We'll do a "histogram" in [0..10] for integer outcomes
    let mut bins = vec![0f32; 11];
    for &val in &samples {
      if val <= 10 {
        bins[val as usize] += 1.0;
      }
    }
    let total = samples.len() as f32;
    for b in bins.iter_mut() {
      *b /= total;
    }
    // x= 0..10
    let xs: Vec<f32> = (0..11).map(|i| i as f32).collect();

    // plot
    let trace = Scatter::new(xs.clone(), bins.clone())
      .mode(Mode::Lines)
      .line(Line::new().shape(LineShape::Linear))
      .name("Binomial PMF");
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    let mean_sample = samples.iter().copied().sum::<u32>() as f32 / n as f32;
    println!("[Binomial(10,0.3)] empirical mean: {:.4}", mean_sample);
    // theory = n*p=3
    assert!((mean_sample - 3.0).abs() < 0.2);
  }

  #[test]
  fn test_geometric() {
    let mut rng = thread_rng();
    // p=0.25 => mean=1/p=4 if we define Geometric with +1 offset
    let geom = SimdGeometric::new(0.25);
    let n = 100_000;
    let samples: Vec<u32> = (0..n).map(|_| geom.sample(&mut rng)).collect();

    // we do a histogram from 0..20?
    let max_range = 20;
    let mut bins = vec![0f32; max_range + 1];
    for &val in &samples {
      if val as usize <= max_range {
        bins[val as usize] += 1.0;
      }
    }
    let total = n as f32;
    for b in bins.iter_mut() {
      *b /= total;
    }
    let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
    let trace = Scatter::new(xs.clone(), bins.clone())
      .mode(Mode::Lines)
      .name("Geometric PMF");
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    let mean_sample = samples.iter().copied().sum::<u32>() as f32 / n as f32;
    println!("[Geometric(p=0.25)] empirical mean: {:.4}", mean_sample);
    // if "Geom" = 1 + floor(...) approach => mean=1/p=4
    assert!((mean_sample - 4.0).abs() < 0.5);
  }

  #[test]
  fn test_hypergeometric() {
    let mut rng = thread_rng();
    // N=20, K=5, n=6
    let hyper = SimdHypergeometric::new(20, 5, 6);
    // theoretical mean= n*(K/N)= 6*(5/20)= 1.5
    let n = 50_000;
    let samples: Vec<u32> = (0..n).map(|_| hyper.sample(&mut rng)).collect();

    let max_range = 6;
    let mut bins = vec![0f32; max_range + 1];
    for &val in &samples {
      if val <= 6 {
        bins[val as usize] += 1.0;
      }
    }
    let total = n as f32;
    for b in bins.iter_mut() {
      *b /= total;
    }
    let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
    let trace = Scatter::new(xs.clone(), bins.clone())
      .mode(Mode::Lines)
      .name("Hyper PMF");
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    let mean_sample = samples.iter().copied().sum::<u32>() as f32 / n as f32;
    println!(
      "[Hypergeo(20,5,6)] emp mean: {:.4}, theory=1.5",
      mean_sample
    );
    assert!((mean_sample - 1.5).abs() < 0.2);
  }

  #[test]
  fn test_poisson() {
    let mut rng = thread_rng();
    // lambda=4
    let pois = SimdPoisson::new(4.0);
    let n = 100_000;
    let samples: Vec<u32> = (0..n).map(|_| pois.sample(&mut rng)).collect();

    // let's do a histogram up to ~15
    let max_range = 15;
    let mut bins = vec![0f32; max_range + 1];
    for &val in &samples {
      if val as usize <= max_range {
        bins[val as usize] += 1.0;
      }
    }
    let total = n as f32;
    for b in bins.iter_mut() {
      *b /= total;
    }
    let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
    let trace = Scatter::new(xs.clone(), bins.clone())
      .mode(Mode::Lines)
      .name("Poisson PMF");
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    let mean_sample = samples.iter().sum::<u32>() as f32 / n as f32;
    println!("[Poisson(lambda=4)] empirical mean= {:.4}", mean_sample);
    // theory= 4
    assert!((mean_sample - 4.0).abs() < 0.3);
  }
}
