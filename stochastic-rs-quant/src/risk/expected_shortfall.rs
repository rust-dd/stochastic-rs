//! Expected Shortfall / Conditional VaR of the loss $L$ (Acerbi & Tasche 2002, (3.3)),
//! $$
//! \mathrm{ES}_{\alpha}(L)=\frac{1}{1-\alpha}\int_{\alpha}^{1}\mathrm{VaR}_{u}(L)\,du;
//! $$
//! the historical and Monte Carlo estimators take it on the sample's own law, its tail mean.
//!
//! Acerbi & Tasche (2002), J. Bank. Financ. 26(7), doi:10.1016/S0378-4266(02)00283-2.
//! Rockafellar & Uryasev (2002), J. Bank. Financ. 26(7), doi:10.1016/S0378-4266(02)00271-6.

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_pdf;

use super::var::PnlOrLoss;
use super::var::VarMethod;
use super::var::assert_confidence;
use super::var::gaussian_var;
use super::var::loss_moments;
use super::var::sorted_losses;
use crate::QuantError;
use crate::traits::RealExt;

/// Expected Shortfall by `method`, with that method's errors; panics unless `0 < confidence < 1`.
pub fn expected_shortfall<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  method: VarMethod,
) -> Result<T, QuantError> {
  assert_confidence(confidence);
  match method {
    VarMethod::Gaussian => gaussian_es(samples, confidence, orientation),
    VarMethod::Historical => historical_es(samples, confidence, orientation),
    VarMethod::MonteCarlo => monte_carlo_es(samples, confidence, orientation),
  }
}

/// Gaussian ES from the sample moments; `InsufficientData` below two samples, `InvalidInput`
/// for a non-finite one, `Numerical` if a moment overflows; panics unless `0 < confidence < 1`.
///
/// $$
/// \mathrm{ES}_{\alpha}=-\hat\mu+\hat\sigma\,\frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}.
/// $$
pub fn gaussian_es<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError> {
  assert_confidence(confidence);
  let (mean, sigma) = loss_moments(samples, orientation, "gaussian ES")?;
  let c = confidence.to_f64().unwrap();
  let q = ndtri(c);
  let phi_q = norm_pdf(q);
  let factor = T::from_f64_fast(phi_q / (1.0 - c));
  Ok(mean + sigma * factor)
}

/// Historical ES, the tail mean of the sample's loss law (Acerbi & Tasche 2002, Def. 2.6) over
/// its sorted losses `L₍₁₎ ≤ … ≤ L₍ₙ₎`; the errors and panic of `historical_var`.
///
/// $$
/// \mathrm{ES}_\alpha=\frac1w\Big[\sum_{i=n-m+1}^{n}L_{(i)}+(w-m)\,L_{(n-m)}\Big],\qquad
///   w=n(1-\alpha),\quad m=\min(\lfloor w\rfloor,\,n-1).
/// $$
pub fn historical_es<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError> {
  tail_es(samples, confidence, orientation, "historical ES")
}

/// Monte Carlo ES: [`historical_es`] over simulated samples.
pub fn monte_carlo_es<T: RealExt>(
  simulated_samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<T, QuantError> {
  tail_es(simulated_samples, confidence, orientation, "Monte Carlo ES")
}

/// Gaussian VaR and ES together.
pub fn gaussian_var_es<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
) -> Result<(T, T), QuantError> {
  Ok((
    gaussian_var(samples, confidence, orientation)?,
    gaussian_es(samples, confidence, orientation)?,
  ))
}

fn tail_es<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  what: &'static str,
) -> Result<T, QuantError> {
  assert_confidence(confidence);
  let sorted = sorted_losses(samples, orientation, what)?;
  let w = sorted.len() as f64 * (1.0 - confidence.to_f64().unwrap());
  Ok(tail_mean(sorted.iter().rev().copied(), w))
}

/// Tail mean of `w ∈ (0, n]` of the `n` losses given worst first: the `⌊w⌋` worst in full, then
/// `w − ⌊w⌋` of the next, over `w`.
pub(crate) fn tail_mean<T: RealExt>(worst_first: impl IntoIterator<Item = T>, w: f64) -> T {
  let mut losses = worst_first.into_iter();
  let m = w.floor() as usize;
  let top = losses.by_ref().take(m).fold(T::zero(), |acc, l| acc + l);
  let part = losses
    .next()
    .map_or(T::zero(), |l| l * T::from_f64_fast(w - m as f64));
  (top + part) / T::from_f64_fast(w)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;
  use crate::risk::var::historical_var;

  fn assert_close(got: f64, expected: f64) {
    let tol = 4.0 * f64::EPSILON * expected.abs();
    assert!(
      (got - expected).abs() <= tol,
      "got {got:?}, expected {expected:?}"
    );
  }

  fn one_to(k: i32) -> Array1<f64> {
    Array1::from_iter((1..=k).map(f64::from))
  }

  /// With `n = 10` and `α = 0.85` the tail holds 1.5 samples: the worst loss and half of the
  /// second-worst.
  #[test]
  fn the_tail_of_ten_losses_at_85_percent_is_the_worst_and_half_the_next() {
    let losses = array![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
    let es = historical_es(losses.view(), 0.85, PnlOrLoss::Loss).unwrap();
    assert_close(es, 25.0 / 3.0);
  }

  #[test]
  fn es_never_exceeds_the_worst_loss_where_the_tail_holds_a_fraction_of_a_sample() {
    for c in [0.9, 0.95, 0.975, 0.99] {
      assert_close(
        historical_es(one_to(10).view(), c, PnlOrLoss::Loss).unwrap(),
        10.0,
      );
      assert_close(
        monte_carlo_es(one_to(10).view(), c, PnlOrLoss::Loss).unwrap(),
        10.0,
      );
    }
    let es = historical_es(one_to(8).view(), 0.8125, PnlOrLoss::Loss).unwrap();
    assert_close(es, 23.0 / 3.0);
    let ties = array![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0, 10.0];
    assert_close(
      historical_es(ties.view(), 0.85, PnlOrLoss::Loss).unwrap(),
      29.0 / 3.0,
    );
  }

  /// Uryasev's six equally likely scenarios after Rockafellar & Uryasev (2002): at `α = 7/12`,
  /// `CVaR = f₄/5 + 2f₅/5 + 2f₆/5`.
  #[test]
  fn es_reads_the_six_scenario_cvar_example() {
    let es = historical_es(one_to(6).view(), 7.0 / 12.0, PnlOrLoss::Loss).unwrap();
    assert_close(es, 5.2);
  }

  #[test]
  fn constant_losses_have_that_constant_as_es() {
    let losses = Array1::from_elem(50, 1.5);
    assert_close(
      historical_es(losses.view(), 0.95, PnlOrLoss::Loss).unwrap(),
      1.5,
    );
  }

  /// CVaR as the minimum over `ζ` of `ζ + Σ(Lᵢ − ζ)⁺/w` (Acerbi & Tasche 2002, Def. 2.5 and
  /// Cor. 4.3); the function is convex and piecewise linear, so a sample attains it.
  fn cvar_by_minimisation(losses: &[f64], w: f64) -> f64 {
    let excess = |z: f64| losses.iter().map(|&l| (l - z).max(0.0)).sum::<f64>();
    losses
      .iter()
      .map(|&z| z + excess(z) / w)
      .fold(f64::INFINITY, f64::min)
  }

  #[test]
  fn es_is_the_minimised_cvar_on_random_samples_with_ties() {
    let mut fractional_tails = 0;
    for n in [1_usize, 2, 3, 5, 7, 10, 13, 20, 50, 101, 250] {
      for c in [0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999] {
        for seed in 0..10 {
          let normal = SimdNormal::<f64>::new(0.0, 1.0);
          let z = normal
            .seeded(&Deterministic::new(1_000 * seed + n as u64))
            .sample_n(n);
          let losses = z.mapv(|v| (v * 10.0).round() / 10.0);
          let w = n as f64 * (1.0 - c);
          fractional_tails += usize::from(w.fract() != 0.0);
          let oracle = cvar_by_minimisation(losses.as_slice().unwrap(), w);
          let tol = 1e-12 * oracle.abs().max(1.0);
          let es = historical_es(losses.view(), c, PnlOrLoss::Loss).unwrap();
          assert!(
            (es - oracle).abs() <= tol,
            "n {n}, c {c}: {es:?} vs {oracle:?}"
          );
          let var = historical_var(losses.view(), c, PnlOrLoss::Loss).unwrap();
          let worst = losses.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
          assert!(
            var <= es + tol && es <= worst + tol,
            "n {n}, c {c}: {var} {es} {worst}"
          );
        }
      }
    }
    assert_eq!(fractional_tails, 820);
  }
}
