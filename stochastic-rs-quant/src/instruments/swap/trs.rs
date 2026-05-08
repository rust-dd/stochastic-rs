//! Total Return Swap (TRS) — equity total-return leg vs floating-rate
//! funding leg.
//!
//! In a TRS the *total-return receiver* receives the dividend + capital
//! return on a reference asset and pays a floating funding leg
//! `funding_rate + spread`. The fair spread at inception equates the
//! present value of the two legs.
//!
//! # Pricing under continuous-time risk-neutrality
//!
//! The *total* return of an equity over a period — capital gain plus
//! dividends paid — has expected gross return $e^{r\,\Delta t}$ under
//! the risk-neutral measure (a self-financing position grows at the
//! short rate regardless of the dividend yield). The relevant pricing
//! forward is therefore the **total-return forward**:
//!
//! $$
//! F_{\mathrm{TR}}(t) = S_0\,e^{r\,t}
//! $$
//!
//! and **not** the price-only forward $S_0\,e^{(r-q)t}$. Per-period
//! expected total return is $R_i = F_{\mathrm{TR}}(t_i)/F_{\mathrm{TR}}(t_{i-1}) - 1
//! = e^{r\,\alpha_i} - 1$. Discounted equity-leg value:
//!
//! $$
//! \mathrm{PV}_{\mathrm{eq}} = N \sum_i D(t_i)\, R_i
//! $$
//!
//! Funding leg (simple interest over each accrual):
//!
//! $$
//! \mathrm{PV}_{\mathrm{fnd}} = N \sum_i D(t_i)\,\alpha_i\,(r_i + s)
//! $$
//!
//! Note: dividend yield does **not** enter the total-return cashflow —
//! it would only matter for a *price-return swap* (a different, less
//! common instrument). To model that, replace `equity_drift_rate` with
//! `r - q` and treat dividends as a separate leg.
//!
//! Reference: Hull, *Options, Futures, and Other Derivatives*, 11th ed.,
//! §35.8 — Total return swaps.
//! Reference: Brigo, Morini & Pallavicini, *Counterparty Credit Risk,
//! Collateral and Funding*, Wiley (2013), §13 — TRS pricing under
//! collateralised funding.

use crate::traits::FloatExt;

/// Direction in which the swap is held.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrsDirection {
  /// Receive equity total return, pay funding leg.
  #[default]
  ReceiveEquity,
  /// Pay equity total return, receive funding leg.
  PayEquity,
}

/// Schedule entry: accrual end time (in years) and day-count fraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrsPeriod<T: FloatExt> {
  pub end_time: T,
  pub accrual: T,
  pub funding_rate: T,
}

/// Total Return Swap on a single equity reference (or basket NAV).
///
/// The model assumes the reference asset's *total wealth* (price plus
/// reinvested dividends) follows $e^{r\,t}$ in expectation under the
/// risk-neutral measure, where $r$ is `equity_drift_rate`. Dividend
/// yield is therefore not a separate input — it is implicit in the
/// total-return forward.
#[derive(Debug, Clone)]
pub struct TotalReturnSwap<T: FloatExt> {
  pub notional: T,
  pub spot: T,
  /// Continuously-compounded drift of the *total wealth* of the
  /// reference asset (capital + dividends). For a stock referenced
  /// against an OIS-collateralised TRS this is the OIS rate;
  /// for a USD-funded equity TRS it is typically the funding rate.
  pub equity_drift_rate: T,
  /// Accrual schedule — one entry per period.
  pub schedule: Vec<TrsPeriod<T>>,
  /// Spread paid on top of the funding rate.
  pub spread: T,
  pub direction: TrsDirection,
}

/// Valuation summary for a TRS.
#[derive(Debug, Clone)]
pub struct TrsValuation<T: FloatExt> {
  /// PV of the equity total-return leg (positive value to the receiver).
  pub equity_leg_pv: T,
  /// PV of the funding leg (positive value to the receiver of the funding
  /// payments, negative to the payer).
  pub funding_leg_pv: T,
  /// Signed net PV under the swap direction.
  pub net_pv: T,
  /// Fair spread (added to funding rate) that makes `net_pv = 0` for the
  /// prevailing equity-drift / dividend / discount inputs.
  pub fair_spread: T,
  /// Annuity = $\sum_i D(t_i)\,\alpha_i$ — basis-point value of the spread.
  pub spread_annuity: T,
  /// Per-period equity cash flows (forward returns × notional).
  pub equity_cashflows: Vec<T>,
  /// Per-period funding cash flows ((rate + spread) × accrual × notional).
  pub funding_cashflows: Vec<T>,
}

impl<T: FloatExt> TotalReturnSwap<T> {
  /// Total-return forward $F_{\mathrm{TR}}(t) = S_0\,e^{r\,t}$ — the
  /// reference for per-period total-return calculations.
  pub fn total_return_forward(&self, t: T) -> T {
    self.spot * (self.equity_drift_rate * t).exp()
  }

  /// Value the swap given a discount-factor function `df(t)`.
  pub fn value<F: Fn(T) -> T>(&self, df: F) -> TrsValuation<T> {
    let one = T::one();
    let zero = T::zero();
    let n = self.schedule.len();
    let mut prev_fwd = self.spot;
    let mut equity_pv = zero;
    let mut funding_pv = zero;
    let mut annuity = zero;
    let mut equity_cf = Vec::with_capacity(n);
    let mut funding_cf = Vec::with_capacity(n);

    for period in &self.schedule {
      let fwd_now = self.total_return_forward(period.end_time);
      let r_eq = fwd_now / prev_fwd - one;
      let cf_eq = self.notional * r_eq;
      let cf_fund = self.notional * period.accrual * (period.funding_rate + self.spread);
      let disc = df(period.end_time);
      equity_pv += disc * cf_eq;
      funding_pv += disc * cf_fund;
      annuity += disc * period.accrual;
      equity_cf.push(cf_eq);
      funding_cf.push(cf_fund);
      prev_fwd = fwd_now;
    }

    let funding_no_spread: T = self
      .schedule
      .iter()
      .map(|p| df(p.end_time) * p.accrual * p.funding_rate)
      .fold(zero, |acc, x| acc + x);
    let fair_spread = if annuity != zero {
      (equity_pv - funding_no_spread * self.notional) / (annuity * self.notional)
    } else {
      T::nan()
    };
    let net_pv = match self.direction {
      TrsDirection::ReceiveEquity => equity_pv - funding_pv,
      TrsDirection::PayEquity => funding_pv - equity_pv,
    };

    TrsValuation {
      equity_leg_pv: equity_pv,
      funding_leg_pv: funding_pv,
      net_pv,
      fair_spread,
      spread_annuity: annuity * self.notional,
      equity_cashflows: equity_cf,
      funding_cashflows: funding_cf,
    }
  }

  /// Convenience: value at flat continuously-compounded discount rate.
  pub fn value_flat(&self, discount_rate: T) -> TrsValuation<T> {
    self.value(|t| (-discount_rate * t).exp())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn quarterly_schedule(maturity: f64, rate: f64) -> Vec<TrsPeriod<f64>> {
    let n = (maturity * 4.0).round() as usize;
    let dt = maturity / n as f64;
    (1..=n)
      .map(|i| TrsPeriod {
        end_time: dt * i as f64,
        accrual: dt,
        funding_rate: rate,
      })
      .collect()
  }

  #[test]
  fn fair_spread_small_for_self_financing_continuous() {
    // Self-financing: total wealth grows at the funding rate, so each
    // period the TR receiver gets e^{r α}-1 vs the funding payer's r α.
    // The gap is the simple-vs-continuous compounding correction
    // 0.5 r² α + O(α³) — small but not zero.
    let trs = TotalReturnSwap {
      notional: 1_000_000.0,
      spot: 100.0,
      equity_drift_rate: 0.04,
      schedule: quarterly_schedule(1.0, 0.04),
      spread: 0.0,
      direction: TrsDirection::ReceiveEquity,
    };
    let v = trs.value_flat(0.04);
    // For r=4%, quarterly: per-period gap ≈ 0.5·0.04²·0.25 = 2e-4.
    assert!(v.fair_spread.abs() < 5e-4, "fair_spread={}", v.fair_spread);
    assert!(
      v.fair_spread > 0.0,
      "TR ≥ simple-funding ⇒ fair spread positive"
    );
  }

  #[test]
  fn dividend_yield_does_not_enter_total_return() {
    // Total-return swap pays the *total* return: capital + dividends.
    // Dropping dividend_yield from the model is correct — under risk-
    // neutral measure E[total return] = e^{r·dt}-1 regardless of q.
    // (Smoke test: model has no `dividend_yield` field at all.)
    let trs = TotalReturnSwap {
      notional: 1.0,
      spot: 100.0,
      equity_drift_rate: 0.05,
      schedule: quarterly_schedule(1.0, 0.05),
      spread: 0.0,
      direction: TrsDirection::ReceiveEquity,
    };
    let v = trs.value_flat(0.05);
    assert!(v.equity_leg_pv > 0.0);
  }

  #[test]
  fn pay_vs_receive_have_opposite_signs() {
    let mut trs = TotalReturnSwap {
      notional: 1_000_000.0,
      spot: 100.0,
      equity_drift_rate: 0.06,
      schedule: quarterly_schedule(1.0, 0.04),
      spread: 0.005,
      direction: TrsDirection::ReceiveEquity,
    };
    let v_recv = trs.value_flat(0.04);
    trs.direction = TrsDirection::PayEquity;
    let v_pay = trs.value_flat(0.04);
    assert!((v_recv.net_pv + v_pay.net_pv).abs() < 1e-9);
  }

  #[test]
  fn fair_spread_zeroes_net_pv() {
    let mut trs = TotalReturnSwap {
      notional: 1_000_000.0,
      spot: 100.0,
      equity_drift_rate: 0.06,
      schedule: quarterly_schedule(2.0, 0.04),
      spread: 0.0,
      direction: TrsDirection::ReceiveEquity,
    };
    let v0 = trs.value_flat(0.04);
    trs.spread = v0.fair_spread;
    let v1 = trs.value_flat(0.04);
    assert!(
      v1.net_pv.abs() < 1e-7,
      "net_pv at fair spread = {}",
      v1.net_pv
    );
  }

  #[test]
  fn cashflows_match_period_count() {
    let trs = TotalReturnSwap {
      notional: 1.0,
      spot: 100.0,
      equity_drift_rate: 0.05,
      schedule: quarterly_schedule(1.0, 0.05),
      spread: 0.0,
      direction: TrsDirection::ReceiveEquity,
    };
    let v = trs.value_flat(0.05);
    assert_eq!(v.equity_cashflows.len(), 4);
    assert_eq!(v.funding_cashflows.len(), 4);
  }

  #[test]
  fn matches_textbook_one_period_formula() {
    // Single-period bullet TRS, total return = e^{r·T}-1, funding = r·T.
    // Fair spread × T ≈ 0.5 r² T (Taylor expansion of e^{rT}-1-rT).
    let r = 0.05;
    let t: f64 = 1.0;
    let trs = TotalReturnSwap {
      notional: 1.0,
      spot: 100.0,
      equity_drift_rate: r,
      schedule: vec![TrsPeriod {
        end_time: t,
        accrual: t,
        funding_rate: r,
      }],
      spread: 0.0,
      direction: TrsDirection::ReceiveEquity,
    };
    let v = trs.value_flat(r);
    let expected_eq_cf = (r * t).exp() - 1.0;
    let expected_fund_cf = r * t;
    assert!((v.equity_cashflows[0] - expected_eq_cf).abs() < 1e-12);
    assert!((v.funding_cashflows[0] - expected_fund_cf).abs() < 1e-12);
    let expected_spread = (expected_eq_cf - expected_fund_cf) / t;
    assert!((v.fair_spread - expected_spread).abs() < 1e-12);
  }
}
