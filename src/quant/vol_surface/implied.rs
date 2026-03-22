//! # Implied Volatility Surface
//!
//! Constructs an implied volatility surface from market option prices.
//!
//! Given a grid of call (or put) prices $C(K_i, T_j)$ over strikes $K$ and
//! maturities $T$, inverts the Black-Scholes formula to obtain
//! $\sigma_{\mathrm{imp}}(K_i, T_j)$.
//!
//! Uses the [`implied_vol`] crate (Jäckel's rational approximation) for
//! robust, high-precision inversion.
//!
//! Reference: Jäckel (2017), "Let's Be Rational"

use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;
use ndarray::Array2;

/// Market data for a single option quote.
#[derive(Clone, Debug)]
pub struct OptionQuote {
  /// Strike price
  pub strike: f64,
  /// Time to expiry in years
  pub tau: f64,
  /// Market price of the option
  pub price: f64,
  /// `true` for call, `false` for put
  pub is_call: bool,
}

/// Implied volatility surface built from market data.
#[derive(Clone, Debug)]
pub struct ImpliedVolSurface {
  /// Strikes (ascending), length N_K
  pub strikes: Vec<f64>,
  /// Maturities in years (ascending), length N_T
  pub maturities: Vec<f64>,
  /// Forward prices for each maturity, length N_T
  pub forwards: Vec<f64>,
  /// Implied volatility grid (N_T, N_K); NaN where inversion failed
  pub ivs: Array2<f64>,
  /// Total implied variance grid: $w(k, T) = \sigma^2 T$, shape (N_T, N_K)
  pub total_variance: Array2<f64>,
  /// Log-forward moneyness grid: $k = \ln(K / F)$, shape (N_T, N_K)
  pub log_moneyness: Array2<f64>,
}

impl ImpliedVolSurface {
  /// Build an implied volatility surface from a grid of option prices.
  ///
  /// # Arguments
  /// * `strikes` - Strike prices (ascending)
  /// * `maturities` - Maturities in years (ascending)
  /// * `forwards` - Forward prices for each maturity
  /// * `prices` - **Undiscounted** option price grid (N_T, N_K)
  /// * `is_call` - Whether prices are call (`true`) or put (`false`)
  #[must_use]
  pub fn from_prices(
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    forwards: Vec<f64>,
    prices: &Array2<f64>,
    is_call: bool,
  ) -> Self {
    let nt = maturities.len();
    let nk = strikes.len();
    assert_eq!(prices.dim(), (nt, nk), "prices shape must be (N_T, N_K)");
    assert_eq!(forwards.len(), nt, "forwards length must match maturities");

    let mut ivs = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut total_variance = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut log_moneyness = Array2::<f64>::zeros((nt, nk));

    for j in 0..nt {
      let f = forwards[j];
      let t = maturities[j];
      for i in 0..nk {
        let k = strikes[i];
        log_moneyness[[j, i]] = (k / f).ln();

        let iv = ImpliedBlackVolatility::builder()
          .option_price(prices[[j, i]])
          .forward(f)
          .strike(k)
          .expiry(t)
          .is_call(is_call)
          .build()
          .and_then(|v| v.calculate::<DefaultSpecialFn>())
          .unwrap_or(f64::NAN);

        if iv.is_finite() && iv > 0.0 {
          ivs[[j, i]] = iv;
          total_variance[[j, i]] = iv * iv * t;
        }
      }
    }

    Self {
      strikes,
      maturities,
      forwards,
      ivs,
      total_variance,
      log_moneyness,
    }
  }

  /// Build from a flat list of [`OptionQuote`]s.
  ///
  /// Quotes are sorted and grouped by maturity, then by strike.
  /// Forward prices are required for each unique maturity.
  #[must_use]
  pub fn from_quotes(quotes: &[OptionQuote], forwards: &[(f64, f64)]) -> Self {
    let mut tau_set: Vec<f64> = quotes.iter().map(|q| q.tau).collect();
    tau_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tau_set.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let mut strike_set: Vec<f64> = quotes.iter().map(|q| q.strike).collect();
    strike_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    strike_set.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let fwd_map: std::collections::HashMap<u64, f64> = forwards
      .iter()
      .map(|&(t, f)| (t.to_bits(), f))
      .collect();

    let nt = tau_set.len();
    let nk = strike_set.len();
    let mut prices = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut is_call_grid = vec![true; nt * nk];
    let fwd_vec: Vec<f64> = tau_set
      .iter()
      .map(|t| {
        *fwd_map
          .get(&t.to_bits())
          .unwrap_or_else(|| panic!("missing forward for tau={t}"))
      })
      .collect();

    for q in quotes {
      let j = tau_set
        .iter()
        .position(|t| (t - q.tau).abs() < 1e-12)
        .unwrap();
      let i = strike_set
        .iter()
        .position(|k| (k - q.strike).abs() < 1e-12)
        .unwrap();
      prices[[j, i]] = q.price;
      is_call_grid[j * nk + i] = q.is_call;
    }

    let mut ivs = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut total_variance = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut log_moneyness = Array2::<f64>::zeros((nt, nk));

    for j in 0..nt {
      let f = fwd_vec[j];
      let t = tau_set[j];
      for i in 0..nk {
        let p = prices[[j, i]];
        if p.is_nan() {
          continue;
        }
        let k = strike_set[i];
        log_moneyness[[j, i]] = (k / f).ln();

        let iv = ImpliedBlackVolatility::builder()
          .option_price(p)
          .forward(f)
          .strike(k)
          .expiry(t)
          .is_call(is_call_grid[j * nk + i])
          .build()
          .and_then(|v| v.calculate::<DefaultSpecialFn>())
          .unwrap_or(f64::NAN);

        if iv.is_finite() && iv > 0.0 {
          ivs[[j, i]] = iv;
          total_variance[[j, i]] = iv * iv * t;
        }
      }
    }

    Self {
      strikes: strike_set,
      maturities: tau_set,
      forwards: fwd_vec,
      ivs,
      total_variance,
      log_moneyness,
    }
  }

  /// Extract a single smile slice (implied vols for one maturity).
  #[must_use]
  pub fn smile_slice(&self, maturity_idx: usize) -> SmileSlice {
    assert!(maturity_idx < self.maturities.len());
    let nk = self.strikes.len();
    let mut ks = Vec::with_capacity(nk);
    let mut vols = Vec::with_capacity(nk);
    let mut ws = Vec::with_capacity(nk);
    let f = self.forwards[maturity_idx];
    let t = self.maturities[maturity_idx];

    for i in 0..nk {
      let iv = self.ivs[[maturity_idx, i]];
      if iv.is_finite() {
        ks.push(self.log_moneyness[[maturity_idx, i]]);
        vols.push(iv);
        ws.push(iv * iv * t);
      }
    }

    SmileSlice {
      log_moneyness: ks,
      implied_vols: vols,
      total_variance: ws,
      forward: f,
      tau: t,
    }
  }
}

/// A single maturity smile slice with market-observed data.
#[derive(Clone, Debug)]
pub struct SmileSlice {
  /// Log-forward moneyness $k = \ln(K/F)$
  pub log_moneyness: Vec<f64>,
  /// Implied volatilities
  pub implied_vols: Vec<f64>,
  /// Total implied variance $w = \sigma^2 T$
  pub total_variance: Vec<f64>,
  /// Forward price
  pub forward: f64,
  /// Time to expiry in years
  pub tau: f64,
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::array;

  #[test]
  fn from_prices_round_trip() {
    use statrs::distribution::{ContinuousCDF, Normal};

    let normal = Normal::new(0.0, 1.0).unwrap();
    let s = 100.0;
    let r = 0.05;
    let sigma = 0.20;

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.50, 1.0];
    let forwards: Vec<f64> = maturities.iter().map(|&t| s * f64::exp(r * t)).collect();

    let mut prices = Array2::<f64>::zeros((maturities.len(), strikes.len()));
    for (j, &t) in maturities.iter().enumerate() {
      let f = forwards[j];
      for (i, &k) in strikes.iter().enumerate() {
        let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        // Undiscounted Black price
        prices[[j, i]] = f * normal.cdf(d1) - k * normal.cdf(d2);
      }
    }

    let surface = ImpliedVolSurface::from_prices(
      strikes.clone(),
      maturities.clone(),
      forwards.clone(),
      &prices,
      true,
    );

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          (iv - sigma).abs() < 1e-6,
          "iv={iv} vs sigma={sigma} at T={}, K={}",
          maturities[j],
          strikes[i]
        );
      }
    }
  }

  #[test]
  fn smile_slice_filters_nans() {
    let surface = ImpliedVolSurface {
      strikes: vec![90.0, 100.0, 110.0],
      maturities: vec![0.5],
      forwards: vec![100.0],
      ivs: array![[0.22, f64::NAN, 0.20]],
      total_variance: array![[0.0242, f64::NAN, 0.02]],
      log_moneyness: array![[(90.0_f64 / 100.0).ln(), 0.0, (110.0_f64 / 100.0).ln()]],
    };

    let slice = surface.smile_slice(0);
    assert_eq!(slice.implied_vols.len(), 2);
    assert!((slice.implied_vols[0] - 0.22).abs() < 1e-12);
    assert!((slice.implied_vols[1] - 0.20).abs() < 1e-12);
  }
}
