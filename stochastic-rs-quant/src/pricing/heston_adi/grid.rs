//! Non-uniform meshes and finite-difference coefficients of in 't Hout &
//! Foulon (2008), §2.2: the sinh-stretched `s` and `v` meshes (2.6), (2.8)
//! and the three-point first-derivative stencils (2.9a–c), the second
//! derivative (2.10) and the mixed derivative (2.11) on arbitrary meshes.

/// `s`-mesh `s_i = K + c sinh(ξ_i)` clustering around the strike (2.6);
/// `s_0 = lower`, `s_{m} = upper`, `c = K / 5`.
pub(super) fn strike_centred_mesh(lower: f64, upper: f64, strike: f64, m: usize) -> Vec<f64> {
  let c = strike / 5.0;
  let xi_lo = ((lower - strike) / c).asinh();
  let xi_hi = ((upper - strike) / c).asinh();
  let d_xi = (xi_hi - xi_lo) / m as f64;
  let mut s: Vec<f64> = (0..=m)
    .map(|i| strike + c * (xi_lo + i as f64 * d_xi).sinh())
    .collect();
  s[0] = lower;
  s[m] = upper;
  s
}

/// `v`-mesh `v_j = d sinh(j Δη)` clustering at `v = 0` (2.8); `v_m = upper`,
/// `d = upper / 500`.
pub(super) fn origin_centred_mesh(upper: f64, m: usize) -> Vec<f64> {
  let d = upper / 500.0;
  let d_eta = (upper / d).asinh() / m as f64;
  let mut v: Vec<f64> = (0..=m).map(|j| d * (j as f64 * d_eta).sinh()).collect();
  v[0] = 0.0;
  v[m] = upper;
  v
}

/// Backward three-point first derivative at `x_i` on `(x_{i−2}, x_{i−1}, x_i)`
/// (2.9a); returns `(α_{i,−2}, α_{i,−1}, α_{i,0})` for mesh widths
/// `dx_prev = Δx_{i−1}`, `dx = Δx_i`.
pub(super) fn backward_first(dx_prev: f64, dx: f64) -> (f64, f64, f64) {
  (
    dx / (dx_prev * (dx_prev + dx)),
    (-dx_prev - dx) / (dx_prev * dx),
    (dx_prev + 2.0 * dx) / (dx * (dx_prev + dx)),
  )
}

/// Central three-point first derivative (2.9b): `(β_{i,−1}, β_{i,0}, β_{i,1})`
/// for `dx = Δx_i`, `dx_next = Δx_{i+1}`.
pub(super) fn central_first(dx: f64, dx_next: f64) -> (f64, f64, f64) {
  (
    -dx_next / (dx * (dx + dx_next)),
    (dx_next - dx) / (dx * dx_next),
    dx / (dx_next * (dx + dx_next)),
  )
}

/// Forward three-point first derivative (2.9c): `(γ_{i,0}, γ_{i,1}, γ_{i,2})`
/// for `dx_next = Δx_{i+1}`, `dx_next2 = Δx_{i+2}`.
pub(super) fn forward_first(dx_next: f64, dx_next2: f64) -> (f64, f64, f64) {
  (
    (-2.0 * dx_next - dx_next2) / (dx_next * (dx_next + dx_next2)),
    (dx_next + dx_next2) / (dx_next * dx_next2),
    -dx_next / (dx_next2 * (dx_next + dx_next2)),
  )
}

/// Central second derivative (2.10): `(δ_{i,−1}, δ_{i,0}, δ_{i,1})`.
pub(super) fn central_second(dx: f64, dx_next: f64) -> (f64, f64, f64) {
  (
    2.0 / (dx * (dx + dx_next)),
    -2.0 / (dx * dx_next),
    2.0 / (dx_next * (dx + dx_next)),
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn meshes_hit_their_end_points_and_cluster_where_the_paper_says() {
    let s = strike_centred_mesh(0.0, 800.0, 100.0, 60);
    assert_eq!((s[0], s[60]), (0.0, 800.0));
    assert!(s.windows(2).all(|w| w[1] > w[0]));
    let near_k = s.iter().filter(|&&x| (x - 100.0).abs() < 20.0).count();
    let far = s.iter().filter(|&&x| x > 400.0).count();
    assert!(
      near_k > far,
      "more points near the strike than in the far tail"
    );
    let v = origin_centred_mesh(5.0, 30);
    assert_eq!((v[0], v[30]), (0.0, 5.0));
    assert!(v[1] < 0.05 && v.windows(2).all(|w| w[1] > w[0]));
  }

  /// The stencils reproduce derivatives of a quadratic exactly (second-order
  /// truncation), on an uneven mesh.
  #[test]
  fn stencils_are_exact_on_quadratics() {
    let x = [0.0, 0.3, 0.7, 1.5, 2.0];
    let f = |t: f64| 2.0 * t * t - 3.0 * t + 1.0;
    let df = |t: f64| 4.0 * t - 3.0;
    let dx = |i: usize| x[i] - x[i - 1];
    let (a2, a1, a0) = backward_first(dx(2), dx(3));
    assert!((a2 * f(x[1]) + a1 * f(x[2]) + a0 * f(x[3]) - df(x[3])).abs() < 1e-12);
    let (b1, b0, b2) = central_first(dx(2), dx(3));
    assert!((b1 * f(x[1]) + b0 * f(x[2]) + b2 * f(x[3]) - df(x[2])).abs() < 1e-12);
    let (g0, g1, g2) = forward_first(dx(2), dx(3));
    assert!((g0 * f(x[1]) + g1 * f(x[2]) + g2 * f(x[3]) - df(x[1])).abs() < 1e-12);
    let (d1, d0, d2) = central_second(dx(2), dx(3));
    assert!((d1 * f(x[1]) + d0 * f(x[2]) + d2 * f(x[3]) - 4.0).abs() < 1e-12);
  }
}
