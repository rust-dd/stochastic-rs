use crate::traits::FloatExt;

/// Payoff types supported by the M-T engine.
#[derive(Clone, Debug)]
pub enum MtPayoff<T: FloatExt> {
  /// Vanilla call `(S^{asset}_T − K)₊`.
  Call { asset: usize, strike: T },
  /// Vanilla put `(K − S^{asset}_T)₊`.
  Put { asset: usize, strike: T },
  /// Digital put on 2 assets: `1(S₁≤K₁)·1(S₂≤K₂)`.
  DigitalPut2D { strikes: [T; 2] },
  /// Basket call `(Σ wᵢ Sᵢ − K)₊`.
  BasketCall { weights: Vec<T>, strike: T },
  /// Worst-of put `(K − min Sᵢ)₊`.
  WorstOfPut { strike: T },
}

impl<T: FloatExt> MtPayoff<T> {
  pub fn evaluate(&self, st: &[T]) -> T {
    match self {
      Self::Call { asset, strike } => (st[*asset] - *strike).max(T::zero()),
      Self::Put { asset, strike } => (*strike - st[*asset]).max(T::zero()),
      Self::DigitalPut2D { strikes } => {
        if st[0] <= strikes[0] && st[1] <= strikes[1] {
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
        (basket - *strike).max(T::zero())
      }
      Self::WorstOfPut { strike } => {
        let worst = st.iter().copied().fold(T::infinity(), |a, b| a.min(b));
        (*strike - worst).max(T::zero())
      }
    }
  }
}
