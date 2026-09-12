use super::*;

/// Golden-section minimisation of a unimodal function on `[lo, hi]`.
fn golden_section(f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, iters: usize) -> f64 {
  let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
  let mut c = hi - inv_phi * (hi - lo);
  let mut d = lo + inv_phi * (hi - lo);
  let (mut fc, mut fd) = (f(c), f(d));
  for _ in 0..iters {
    if fc < fd {
      hi = d;
      d = c;
      fd = fc;
      c = hi - inv_phi * (hi - lo);
      fc = f(c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + inv_phi * (hi - lo);
      fd = f(d);
    }
  }
  0.5 * (lo + hi)
}

/// Anchored slice data in `f64`.
struct AnchoredSlice {
  ks: Vec<f64>,
  ws: Vec<f64>,
  k_star: f64,
  theta_star: f64,
}

/// Admissible `ψ` interval for `ρ` on an anchored slice, given the previous
/// slice; `None` when the bounds cross.
fn psi_bounds(
  rho: f64,
  slice: &AnchoredSlice,
  previous: Option<&EssviSlice<f64>>,
) -> Option<(f64, f64)> {
  let abs = 1.0 + rho.abs();
  let (k, theta) = (slice.k_star, slice.theta_star);
  let psi_plus =
    -2.0 * rho * k / abs + (4.0 * rho * rho * k * k / (abs * abs) + 4.0 * theta / abs).sqrt();
  let mut hi = psi_plus.min(4.0 / abs);
  let mut lo = 1e-10_f64;
  let rk = rho * k;
  if rk > 0.0 {
    hi = hi.min(theta / rk * (1.0 - 1e-9));
  }
  if let Some(prev) = previous {
    let psi_minus = ((prev.psi - prev.rho * prev.psi) / (1.0 - rho))
      .max((prev.psi + prev.rho * prev.psi) / (1.0 + rho));
    lo = lo.max(psi_minus);
    if rk.abs() > 0.0 {
      let psi_hat = (theta - prev.theta) / rk;
      if rk > 0.0 {
        hi = hi.min(psi_hat);
      } else {
        lo = lo.max(psi_hat);
      }
    } else if theta < prev.theta {
      return None;
    }
  }
  if let Some(prev) = previous {
    if k == 0.0 && theta == prev.theta {
      let skew = prev.rho * prev.psi;
      if rho == 0.0 {
        if skew.abs() > 1e-14 {
          return None;
        }
      } else {
        let psi = skew / rho;
        return (psi >= lo && psi <= hi).then_some((psi, psi));
      }
    } else {
      // Section 2.1, R4: psi / theta must not increase between fitted slices.
      let denominator = prev.theta + prev.psi * rk;
      if denominator > 0.0 {
        hi = hi.min(prev.psi * theta / denominator);
      }
    }
  }
  (lo <= hi).then_some((lo, hi))
}

fn slice_sse(rho: f64, psi: f64, slice: &AnchoredSlice) -> f64 {
  let theta = slice.theta_star - rho * psi * slice.k_star;
  let model = EssviSlice::new(1.0, theta, rho, psi);
  slice
    .ks
    .iter()
    .zip(&slice.ws)
    .map(|(&k, &w)| (model.total_variance(k) - w).powi(2))
    .sum()
}

/// Best admissible `(ψ, sse)` for a given `ρ`.
fn best_psi(
  rho: f64,
  slice: &AnchoredSlice,
  previous: Option<&EssviSlice<f64>>,
) -> Option<(f64, f64)> {
  let (lo, hi) = psi_bounds(rho, slice, previous)?;
  let f = |psi: f64| slice_sse(rho, psi, slice);
  let psi = golden_section(&f, lo, hi, 80);
  Some((psi, f(psi)))
}

/// Calibrates one anchored slice going forward from `previous`.
fn calibrate_slice(
  slice: &AnchoredSlice,
  previous: Option<&EssviSlice<f64>>,
  maturity: f64,
) -> Option<EssviSlice<f64>> {
  let objective = |rho: f64| best_psi(rho, slice, previous).map_or(f64::INFINITY, |(_, sse)| sse);
  let grid = (0..81)
    .map(|i| -0.99 + 1.98 * i as f64 / 80.0)
    .chain(previous.map(|s| s.rho));
  let (mut best_rho, mut best_sse) = (0.0_f64, f64::INFINITY);
  for rho in grid {
    let sse = objective(rho);
    if sse < best_sse {
      best_sse = sse;
      best_rho = rho;
    }
  }
  let step = 1.98 / 80.0;
  let refined = golden_section(
    &objective,
    (best_rho - step).max(-0.999),
    (best_rho + step).min(0.999),
    60,
  );
  let rho = if objective(refined) <= best_sse {
    refined
  } else {
    best_rho
  };
  let (psi, sse) = best_psi(rho, slice, previous)?;
  if !sse.is_finite() {
    return None;
  }
  let calibrated = EssviSlice::new(
    maturity,
    slice.theta_star - rho * psi * slice.k_star,
    rho,
    psi,
  );
  if previous.is_some_and(|prev| !calendar_free_pair(prev, &calibrated)) {
    return None;
  }
  Some(calibrated)
}

/// Calibrates eSSVI slices to `slices` (ascending `maturities`, one per
/// slice) going forward in maturity, so that every slice is butterfly-free
/// and every consecutive pair calendar-spread-free by construction. Each
/// slice is anchored at its data point closest to the money.
///
/// # Panics
/// Panics for invalid quotes or an infeasible fit. Use [`try_calibrate_essvi`]
/// to handle these failures explicitly.
pub fn calibrate_essvi<T: RealExt>(slices: &[SsviSlice<T>], maturities: &[T]) -> EssviSurface<T> {
  try_calibrate_essvi(slices, maturities).expect("eSSVI calibration failed")
}

/// Calibrates an admissible eSSVI surface, returning an error for invalid quotes
/// or when an anchored slice has no feasible fit following the previous slice.
pub fn try_calibrate_essvi<T: RealExt>(
  slices: &[SsviSlice<T>],
  maturities: &[T],
) -> anyhow::Result<EssviSurface<T>> {
  anyhow::ensure!(slices.len() == maturities.len(), "one maturity per slice");
  anyhow::ensure!(
    !slices.is_empty(),
    "eSSVI calibration needs at least one slice"
  );
  anyhow::ensure!(
    maturities.iter().all(|t| t.is_finite() && *t > T::zero())
      && maturities.windows(2).all(|w| w[0] < w[1]),
    "maturities must be finite, positive and increasing"
  );
  let mut out = Vec::<EssviSlice<f64>>::with_capacity(slices.len());
  for (index, (slice, &maturity)) in slices.iter().zip(maturities).enumerate() {
    let ks = slice
      .log_moneyness
      .iter()
      .map(|k| k.to_f64().unwrap_or(f64::NAN))
      .collect::<Vec<_>>();
    let ws = slice
      .total_variance
      .iter()
      .map(|w| w.to_f64().unwrap_or(f64::NAN))
      .collect::<Vec<_>>();
    anyhow::ensure!(
      ks.len() >= 2 && ks.len() == ws.len(),
      "slice {index} needs at least two paired quotes"
    );
    anyhow::ensure!(
      ks.iter().all(|k| k.is_finite()) && ws.iter().all(|w| w.is_finite() && *w > 0.0),
      "slice {index} quotes must be finite with positive total variance"
    );
    let anchor = (0..ks.len())
      .min_by(|&i, &j| {
        ks[i]
          .abs()
          .partial_cmp(&ks[j].abs())
          .expect("finite log-moneyness")
      })
      .expect("non-empty");
    let anchored = AnchoredSlice {
      k_star: ks[anchor],
      theta_star: ws[anchor],
      ks,
      ws,
    };
    let previous = out.last();
    let calibrated = calibrate_slice(&anchored, previous, maturity.to_f64().unwrap_or(f64::NAN))
      .ok_or_else(|| anyhow::anyhow!("slice {index} has no admissible anchored eSSVI fit"))?;
    out.push(calibrated);
  }
  Ok(EssviSurface::new(
    out
      .into_iter()
      .map(|s| {
        EssviSlice::new(
          T::from_f64_fast(s.maturity),
          T::from_f64_fast(s.theta),
          T::from_f64_fast(s.rho),
          T::from_f64_fast(s.psi),
        )
      })
      .collect(),
  ))
}
