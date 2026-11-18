//! # Implied Volatility Surface
//!
//! Constructs an implied volatility surface from market option prices.
//!
//! Given a grid of call (or put) prices $C(K_i, T_j)$ over strikes $K$ and
//! maturities $T$, inverts the Black-Scholes formula to obtain
//! $\sigma_{\mathrm{imp}}(K_i, T_j)$.
//!
//! Uses the [`implied_vol`] crate's `Hybrid` solver: Jäckel's rational approximation with FlashIV.
//!
//! Reference: Jäckel (2015), "Let's Be Rational", Wilmott 2015(75), 40–53

use implied_vol::ImpliedBlackVolatility;
use implied_vol::solver::Hybrid;
use ndarray::Array2;

use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::quote::validate_quotes;
use crate::total_order::ascending;

/// Black implied volatility of an undiscounted price on a forward: `0.0` at the intrinsic price,
/// `+∞` at the forward (the strike for a put), NaN outside them. Jäckel (2015), Wilmott 2015(75).
pub(crate) fn black_implied_vol(
  undiscounted_price: f64,
  forward: f64,
  strike: f64,
  tau: f64,
  is_call: bool,
) -> f64 {
  ImpliedBlackVolatility::builder()
    .option_price(undiscounted_price)
    .forward(forward)
    .strike(strike)
    .expiry(tau)
    .is_call(is_call)
    .build()
    .and_then(|iv| iv.calculate_with::<Hybrid>())
    .unwrap_or(f64::NAN)
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

/// Every grid lookup on the surface assumes both axes ascend strictly.
fn assert_ascending_axes(strikes: &[f64], maturities: &[f64]) {
  assert!(
    strikes.windows(2).all(|w| w[0] < w[1]),
    "strikes must satisfy `strikes.windows(2).all(|w| w[0] < w[1])`, got strikes = {strikes:?}"
  );
  assert!(
    maturities.windows(2).all(|w| w[0] < w[1]),
    "maturities must satisfy `maturities.windows(2).all(|w| w[0] < w[1])`, got maturities = {maturities:?}"
  );
}

impl ImpliedVolSurface {
  /// A surface from an **undiscounted** call (`is_call`) or put price grid of shape `(N_T, N_K)`, one forward per maturity; a cell is
  /// NaN where the price does not invert to a positive finite Black vol; panics unless both axes ascend strictly and shapes agree.
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
    assert_ascending_axes(&strikes, &maturities);

    let mut ivs = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut total_variance = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut log_moneyness = Array2::<f64>::zeros((nt, nk));

    for j in 0..nt {
      let f = forwards[j];
      let t = maturities[j];
      for i in 0..nk {
        let k = strikes[i];
        log_moneyness[[j, i]] = (k / f).ln();

        let iv = black_implied_vol(prices[[j, i]], f, k, t, is_call);

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

  /// A surface from an implied-vol grid `ivs` of shape `(N_T, N_K)` over `strikes` and `maturities`, one forward per maturity; a
  /// `total_variance` cell is NaN where the vol is not finite and positive; panics unless both axes ascend strictly and shapes agree.
  #[must_use]
  pub fn from_iv_grid(
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    forwards: Vec<f64>,
    ivs: Array2<f64>,
  ) -> Self {
    let nt = maturities.len();
    let nk = strikes.len();
    assert_eq!(ivs.dim(), (nt, nk), "ivs shape must be (N_T, N_K)");
    assert_eq!(forwards.len(), nt, "forwards length must match maturities");
    assert_ascending_axes(&strikes, &maturities);

    let mut total_variance = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut log_moneyness = Array2::<f64>::zeros((nt, nk));

    for j in 0..nt {
      let f = forwards[j];
      let t = maturities[j];
      for i in 0..nk {
        log_moneyness[[j, i]] = (strikes[i] / f).ln();
        let iv = ivs[[j, i]];
        if iv.is_finite() && iv > 0.0 {
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

  /// A surface from a surrogate's flat row-major `(maturity, strike)` output; panics unless the lengths agree (`N_T * N_K` vols, one
  /// forward per maturity) and both axes ascend strictly — a Heston surrogate's `STRIKES` descend: `predict_implied_vol_surface` sorts.
  ///
  /// # Example
  ///
  /// ```
  /// use stochastic_rs_quant::vol_surface::ImpliedVolSurface;
  ///
  /// let strikes = vec![90.0, 100.0, 110.0];
  /// let maturities = vec![0.5, 1.0];
  /// let forwards = vec![101.0, 102.0];
  /// // In practice a surrogate's `predict_surface(&params)?` with ascending `STRIKES`;
  /// // a literal grid keeps this example free of a trained surrogate.
  /// let flat: Vec<f32> = vec![0.22, 0.20, 0.21, 0.24, 0.22, 0.23];
  /// let surf = ImpliedVolSurface::from_flat_iv_grid(
  ///     strikes, maturities, forwards, &flat,
  /// );
  /// assert_eq!(surf.ivs.dim(), (2, 3));
  /// ```
  #[must_use]
  pub fn from_flat_iv_grid(
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    forwards: Vec<f64>,
    flat_ivs: &[f32],
  ) -> Self {
    let nt = maturities.len();
    let nk = strikes.len();
    assert_eq!(
      flat_ivs.len(),
      nt * nk,
      "flat_ivs length must equal N_T * N_K = {} * {} = {}",
      nt,
      nk,
      nt * nk,
    );
    let ivs = Array2::from_shape_vec((nt, nk), flat_ivs.iter().map(|&v| v as f64).collect())
      .expect("shape (N_T, N_K) is consistent with flat_ivs length");
    Self::from_iv_grid(strikes, maturities, forwards, ivs)
  }

  /// A surface from present-value quotes, inverted as `price · e^{rτ}` Black-76 prices; a cell with
  /// no quote or root is NaN; `InsufficientData`, `InvalidQuote`, `MissingInput`, `InvalidInput`.
  pub fn try_from_quotes(
    quotes: &[OptionQuote],
    forwards: &[(f64, f64)],
    r: f64,
  ) -> Result<Self, QuantError> {
    validate_quotes(quotes)?;
    if !r.is_finite() {
      return Err(QuantError::InvalidInput {
        what: "r",
        reason: format!("must be finite, got {r}"),
      });
    }

    let mut tau_set: Vec<f64> = quotes.iter().map(|q| q.tau).collect();
    tau_set.sort_by(ascending);
    tau_set.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let mut strike_set: Vec<f64> = quotes.iter().map(|q| q.strike).collect();
    strike_set.sort_by(ascending);
    strike_set.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let fwd_map: std::collections::HashMap<u64, f64> =
      forwards.iter().map(|&(t, f)| (t.to_bits(), f)).collect();

    let nt = tau_set.len();
    let nk = strike_set.len();
    let mut prices = Array2::<f64>::from_elem((nt, nk), f64::NAN);
    let mut is_call_grid = vec![true; nt * nk];
    let fwd_vec: Vec<f64> = tau_set
      .iter()
      .map(|t| {
        fwd_map
          .get(&t.to_bits())
          .copied()
          .ok_or_else(|| QuantError::MissingInput {
            what: "forward",
            detail: format!("tau = {t}"),
          })
      })
      .collect::<Result<_, _>>()?;

    for q in quotes {
      let j = tau_set
        .iter()
        .position(|t| (t - q.tau).abs() < 1e-12)
        .unwrap();
      let i = strike_set
        .iter()
        .position(|k| (k - q.strike).abs() < 1e-12)
        .unwrap();
      prices[[j, i]] = q.price * (r * q.tau).exp();
      is_call_grid[j * nk + i] = q.option_type == OptionType::Call;
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

        let iv = black_implied_vol(p, f, k, t, is_call_grid[j * nk + i]);

        if iv.is_finite() && iv > 0.0 {
          ivs[[j, i]] = iv;
          total_variance[[j, i]] = iv * iv * t;
        }
      }
    }

    Ok(Self {
      strikes: strike_set,
      maturities: tau_set,
      forwards: fwd_vec,
      ivs,
      total_variance,
      log_moneyness,
    })
  }

  /// Build an implied-vol surface directly from a market-data provider's
  /// option chain. Fetches the chain for `symbol`, converts the quotes to
  /// the `(OptionQuote, forwards)` inputs via
  /// [`OptionChain::to_surface_inputs`](crate::market::provider::OptionChain::to_surface_inputs)
  /// with carry $(r, q)$, then runs [`Self::try_from_quotes`].
  pub fn from_provider<P: crate::market::provider::MarketDataProvider>(
    provider: &P,
    symbol: &str,
    r: f64,
    q: f64,
  ) -> Result<Self, QuantError> {
    let chain = provider.option_chain(symbol)?;
    let (quotes, forwards) = chain.to_surface_inputs(r, q);
    if quotes.is_empty() {
      return Err(QuantError::MarketData {
        symbol: symbol.to_string(),
        reason: "the option chain yielded no usable quotes".to_string(),
      });
    }
    Self::try_from_quotes(&quotes, &forwards, r)
  }

  /// Extract a single smile slice (implied vols for one maturity).
  #[must_use]
  pub fn smile_slice(&self, maturity_idx: usize) -> SmileSlice {
    assert!(
      maturity_idx < self.maturities.len(),
      "maturity_idx must satisfy `maturity_idx < self.maturities.len()`, got maturity_idx = {maturity_idx:?}"
    );
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
  /// Black implied volatility at each `log_moneyness` point.
  pub implied_vols: Vec<f64>,
  /// Total implied variance $w = \sigma^2 T$
  pub total_variance: Vec<f64>,
  /// Forward price
  pub forward: f64,
  /// Time to maturity in years.
  pub tau: f64,
}

impl SmileSlice {
  /// Fit SVI raw parameters to this smile slice.
  pub fn fit_svi(
    &self,
    initial: Option<super::svi::SviRawParams<f64>>,
  ) -> super::svi::SviRawParams<f64> {
    super::svi::calibrate_svi(&self.log_moneyness, &self.total_variance, initial)
  }

  /// Convert to an SSVI slice using the ATM total variance from the data.
  ///
  /// ATM total variance $\theta$ is interpolated at $k = 0$.
  pub fn to_ssvi_slice(&self) -> super::ssvi::SsviSlice<f64> {
    let theta = self.atm_total_variance();
    super::ssvi::SsviSlice {
      log_moneyness: self.log_moneyness.clone(),
      total_variance: self.total_variance.clone(),
      theta,
    }
  }

  /// ATM total variance (`k = 0`): interpolated between the adjacent points, or extrapolated from
  /// the two innermost ones for a one-sided smile, since an endpoint would bias SSVI's $\theta_t$.
  fn atm_total_variance(&self) -> f64 {
    let n = self.log_moneyness.len();
    if n == 0 {
      return 0.0;
    }
    if n == 1 {
      return self.total_variance[0];
    }

    let idx = self.log_moneyness.partition_point(|&k| k < 0.0);

    if idx == 0 {
      // All strikes are k > 0 (one-sided OTM-call / ITM-put).
      // Extrapolate linearly to k=0 using the two innermost (smallest k) points.
      let k0 = self.log_moneyness[0];
      let k1 = self.log_moneyness[1];
      let w0 = self.total_variance[0];
      let w1 = self.total_variance[1];
      if (k1 - k0).abs() < 1e-14 {
        return w0;
      }
      let slope = (w1 - w0) / (k1 - k0);
      return w0 + slope * (0.0 - k0);
    }
    if idx >= n {
      // All strikes are k < 0 (one-sided ITM-call / OTM-put).
      // Extrapolate linearly to k=0 using the two innermost (largest k) points.
      let k0 = self.log_moneyness[n - 2];
      let k1 = self.log_moneyness[n - 1];
      let w0 = self.total_variance[n - 2];
      let w1 = self.total_variance[n - 1];
      if (k1 - k0).abs() < 1e-14 {
        return w1;
      }
      let slope = (w1 - w0) / (k1 - k0);
      return w1 + slope * (0.0 - k1);
    }

    let k0 = self.log_moneyness[idx - 1];
    let k1 = self.log_moneyness[idx];
    let w0 = self.total_variance[idx - 1];
    let w1 = self.total_variance[idx];

    if (k1 - k0).abs() < 1e-14 {
      return w0;
    }

    let alpha = (0.0 - k0) / (k1 - k0);
    w0 * (1.0 - alpha) + w1 * alpha
  }
}

impl ImpliedVolSurface {
  /// Fit SVI parameters to each maturity slice independently.
  pub fn fit_svi_slices(&self) -> Vec<super::svi::SviRawParams<f64>> {
    let nt = self.maturities.len();
    (0..nt).map(|j| self.smile_slice(j).fit_svi(None)).collect()
  }

  /// Fit SSVI surface to all maturity slices simultaneously.
  ///
  /// First extracts ATM total variance $\theta_t$ per slice, then
  /// calibrates global SSVI parameters $(\rho, \eta, \gamma)$.
  pub fn fit_ssvi(
    &self,
    initial: Option<super::ssvi::SsviParams<f64>>,
  ) -> super::ssvi::SsviSurface<f64> {
    let nt = self.maturities.len();
    let slices: Vec<super::ssvi::SsviSlice<f64>> = (0..nt)
      .map(|j| self.smile_slice(j).to_ssvi_slice())
      .collect();

    let params = super::ssvi::calibrate_ssvi(&slices, initial);

    let thetas: Vec<f64> = slices.iter().map(|s| s.theta).collect();

    super::ssvi::SsviSurface::new(params, thetas, self.maturities.clone())
  }
}

#[cfg(test)]
mod tests;
