use chrono::NaiveDate;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::curves::PyDiscountCurve;
use crate::credit::index::CdsIndex;
use crate::credit::index::IndexName;
use crate::credit::index::flat_survival;
use crate::credit::tranche::CdoTranche;
use crate::credit::tranche::PoolName;

fn parse_date(s: &str) -> PyResult<NaiveDate> {
  NaiveDate::parse_from_str(s, "%Y-%m-%d")
    .map_err(|e| PyValueError::new_err(format!("date '{s}' must be YYYY-MM-DD: {e}")))
}

/// Untranched CDS index of flat-hazard names on a running coupon.
#[pyclass(name = "CdsIndex", unsendable)]
pub struct PyCdsIndex {
  inner: CdsIndex,
}

#[pymethods]
impl PyCdsIndex {
  /// `names` are `(weight, recovery, hazard_rate)` triplets (weights sum to
  /// one); dates are `YYYY-MM-DD`.
  #[new]
  #[pyo3(signature = (names, coupon, notional, effective_date, maturity_date))]
  fn new(
    names: Vec<(f64, f64, f64)>,
    coupon: f64,
    notional: f64,
    effective_date: &str,
    maturity_date: &str,
  ) -> PyResult<Self> {
    let names = names
      .into_iter()
      .map(|(weight, recovery, hazard)| IndexName {
        weight,
        recovery,
        survival: flat_survival(hazard),
      })
      .collect();
    Ok(Self {
      inner: CdsIndex::new(
        names,
        coupon,
        notional,
        parse_date(effective_date)?,
        parse_date(maturity_date)?,
      ),
    })
  }

  /// `(protection_leg, premium_leg, risky_annuity, fair_spread, net_npv)` from the buyer's side.
  fn valuation(
    &self,
    valuation_date: &str,
    discount: &PyDiscountCurve,
  ) -> PyResult<(f64, f64, f64, f64, f64)> {
    let v = self
      .inner
      .valuation(parse_date(valuation_date)?, &discount.inner);
    Ok((
      v.protection_leg_npv,
      v.premium_leg_npv,
      v.risky_annuity,
      v.fair_spread,
      v.net_npv,
    ))
  }

  fn fair_spread(&self, valuation_date: &str, discount: &PyDiscountCurve) -> PyResult<f64> {
    Ok(
      self
        .inner
        .fair_spread(parse_date(valuation_date)?, &discount.inner),
    )
  }

  /// ISDA standard-model upfront for a quoted spread (recovery 40 % by default).
  #[pyo3(signature = (valuation_date, discount, quoted_spread, recovery=0.4))]
  fn isda_upfront(
    &self,
    valuation_date: &str,
    discount: &PyDiscountCurve,
    quoted_spread: f64,
    recovery: f64,
  ) -> PyResult<f64> {
    Ok(self.inner.isda_upfront(
      parse_date(valuation_date)?,
      &discount.inner,
      quoted_spread,
      recovery,
    ))
  }
}

/// Synthetic CDO tranche on a pool of flat-hazard names under the one-factor
/// Gaussian copula.
#[pyclass(name = "CdoTranche", unsendable)]
pub struct PyCdoTranche {
  inner: CdoTranche,
  pool: Vec<PoolName>,
}

#[pymethods]
impl PyCdoTranche {
  /// `pool` entries are `(weight, recovery, hazard_rate)`; `payment_times` in years.
  #[new]
  #[pyo3(signature = (pool, attachment, detachment, spread, payment_times, correlation, accrual=1.0, quadrature_nodes=40, loss_buckets=400))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    pool: Vec<(f64, f64, f64)>,
    attachment: f64,
    detachment: f64,
    spread: f64,
    payment_times: Vec<f64>,
    correlation: f64,
    accrual: f64,
    quadrature_nodes: usize,
    loss_buckets: usize,
  ) -> Self {
    let pool = pool
      .into_iter()
      .map(|(weight, recovery, hazard)| PoolName {
        weight,
        recovery,
        survival: flat_survival(hazard),
      })
      .collect();
    Self {
      inner: CdoTranche::new(
        attachment,
        detachment,
        spread,
        payment_times,
        accrual,
        correlation,
      )
      .with_resolution(quadrature_nodes, loss_buckets),
      pool,
    }
  }

  /// Expected tranche loss at `t` per unit of pool notional.
  fn expected_tranche_loss(&self, t: f64) -> f64 {
    self.inner.expected_tranche_loss(&self.pool, t)
  }

  /// Pool loss distribution on the grid `k / loss_buckets` at `t`.
  fn loss_distribution(&self, t: f64) -> Vec<f64> {
    self.inner.loss_distribution(&self.pool, t)
  }

  /// `(protection_leg, risky_annuity, premium_leg, fair_spread, upfront)` per unit of pool notional.
  fn valuation(&self, discount: &PyDiscountCurve) -> (f64, f64, f64, f64, f64) {
    let v = self.inner.valuation(&self.pool, &discount.inner);
    (
      v.protection_leg,
      v.risky_annuity,
      v.premium_leg,
      v.fair_spread,
      v.upfront,
    )
  }

  /// Vasicek large-pool expected tranche loss for a homogeneous pool with
  /// default probability `p` and loss-given-default `lgd`.
  fn large_pool_expected_tranche_loss(&self, p: f64, lgd: f64) -> f64 {
    self.inner.large_pool_expected_tranche_loss(p, lgd)
  }
}
