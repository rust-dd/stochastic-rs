//! Strike-maturity grid of the shipped training sets (zero rates, `S0 = 1`), from
//! <https://github.com/amuguruza/NN-StochVol-Calibrations>; a surface is flat and maturity-major.
//!
//! Horvath, Muguruza, Tomas, "Deep Learning Volatility", arXiv:1901.09647 (§4.1.1 lists this grid).

/// The notebooks' strike axis, ascending: `K / S0` for the Bergomi sets, `S0 / K` for Heston
/// (see [`heston::STRIKES`](super::heston::STRIKES)).
pub const MONEYNESS: [f64; 11] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];

/// Maturities in years, ascending.
pub const MATURITIES: [f64; 8] = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0];

/// `OUTPUT_DIM` of every shipped surrogate.
pub const LEN: usize = MONEYNESS.len() * MATURITIES.len();

pub(crate) const fn inverse<const N: usize>(nodes: [f64; N]) -> [f64; N] {
  let mut out = [0.0; N];
  let mut i = 0;
  while i < N {
    out[i] = 1.0 / nodes[i];
    i += 1;
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn nodes_are_ascending_and_the_surface_has_88_points() {
    assert!(MONEYNESS.windows(2).all(|w| w[0] < w[1]));
    assert!(MATURITIES.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(LEN, 88);
  }

  #[test]
  fn inverse_descends_and_round_trips() {
    let inv = inverse(MONEYNESS);
    assert!(inv.windows(2).all(|w| w[0] > w[1]));
    assert!(
      inverse(inv)
        .iter()
        .zip(MONEYNESS)
        .all(|(a, b)| (a - b).abs() < 1e-15)
    );
  }
}
