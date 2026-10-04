use crate::traits::FloatExt;

/// Payoff types supported by the M-T engine.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MtPayoff<T: FloatExt> {
  /// Vanilla call `(S^{asset}_T − K)₊`.
  Call { asset: usize, strike: T },
  /// Vanilla put `(K − S^{asset}_T)₊`.
  Put { asset: usize, strike: T },
  /// Digital put on 2 assets: `1(0≤S₁≤K₁)·1(0≤S₂≤K₂)`.
  DigitalPut2D { strikes: [T; 2] },
  /// Basket call `(Σ wᵢ Sᵢ − K)₊`.
  BasketCall { weights: Vec<T>, strike: T },
  /// Worst-of put `(K − min Sᵢ)₊`.
  WorstOfPut { strike: T },
}

impl<T: FloatExt> MtPayoff<T> {
  /// Payoff at the terminal prices `st`; NaN when a price it reads is NaN.
  pub fn evaluate(&self, st: &[T]) -> T {
    match self {
      Self::Call { asset, strike } => (st[*asset] - *strike).max_or_nan(T::zero()),
      Self::Put { asset, strike } => (*strike - st[*asset]).max_or_nan(T::zero()),
      Self::DigitalPut2D { strikes } => {
        assert_eq!(st.len(), 2, "DigitalPut2D requires exactly two assets");
        if st[0].is_nan() || st[1].is_nan() {
          T::nan()
        } else if st[0] >= T::zero()
          && st[0] <= strikes[0]
          && st[1] >= T::zero()
          && st[1] <= strikes[1]
        {
          T::one()
        } else {
          T::zero()
        }
      }
      Self::BasketCall { weights, strike } => {
        let basket = weights
          .iter()
          .zip(st)
          .map(|(&w, &s)| w * s)
          .fold(T::zero(), |a, b| a + b);
        (basket - *strike).max_or_nan(T::zero())
      }
      Self::WorstOfPut { strike } => {
        let worst = st
          .iter()
          .copied()
          .fold(T::infinity(), |a, b| a.min_or_nan(b));
        (*strike - worst).max_or_nan(T::zero())
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pricing::malliavin_thalmaier::kernel::g_digital_put_2d;

  #[test]
  fn digital_put_support_is_closed_and_nonnegative() {
    let payoff = MtPayoff::DigitalPut2D {
      strikes: [100.0_f64, 100.0],
    };

    assert_eq!(payoff.evaluate(&[0.0, 0.0]), 1.0);
    assert_eq!(payoff.evaluate(&[100.0, 100.0]), 1.0);
    assert_eq!(payoff.evaluate(&[-1e-12, 50.0]), 0.0);
    assert_eq!(payoff.evaluate(&[50.0, -1e-12]), 0.0);
    assert_eq!(payoff.evaluate(&[100.0 + 1e-12, 50.0]), 0.0);
  }

  #[test]
  fn digital_put_support_matches_closed_form_kernel_trace() {
    let strikes = [100.0_f64, 80.0];
    let payoff = MtPayoff::DigitalPut2D { strikes };
    let points = [
      [0.0, 0.0],
      [100.0, 80.0],
      [0.0, 40.0],
      [50.0, 80.0],
      [-1e-12, 40.0],
      [50.0, 80.0 + 1e-12],
    ];

    for point in points {
      let kernel = g_digital_put_2d(point, strikes);
      let trace = kernel[0][0] + kernel[1][1];
      assert_eq!(trace, payoff.evaluate(&point), "point={point:?}");
    }
  }

  #[test]
  fn a_nan_terminal_price_is_a_nan_payoff_not_a_clamped_zero() {
    let worst_of = MtPayoff::WorstOfPut { strike: 100.0 };
    assert_eq!(worst_of.evaluate(&[97.0, 99.0, 101.0]), 3.0);
    let call = MtPayoff::Call {
      asset: 1,
      strike: 100.0,
    };
    let put = MtPayoff::Put {
      asset: 1,
      strike: 100.0,
    };
    let basket = MtPayoff::BasketCall {
      weights: vec![1.0 / 3.0; 3],
      strike: 100.0,
    };
    for payoff in [worst_of, call, put, basket] {
      assert!(
        payoff.evaluate(&[97.0, f64::NAN, 101.0]).is_nan(),
        "{payoff:?}"
      );
    }
    let digital = MtPayoff::DigitalPut2D {
      strikes: [100.0, 100.0],
    };
    assert!(digital.evaluate(&[f64::NAN, 50.0]).is_nan());
  }

  #[test]
  #[should_panic(expected = "DigitalPut2D requires exactly two assets")]
  fn digital_put_rejects_wrong_dimension() {
    let payoff = MtPayoff::DigitalPut2D {
      strikes: [100.0_f64, 100.0],
    };
    let _ = payoff.evaluate(&[50.0]);
  }
}
