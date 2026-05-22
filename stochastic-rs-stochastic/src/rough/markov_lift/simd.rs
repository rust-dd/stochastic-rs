use wide::f32x8;
use wide::f64x4;

use crate::traits::FloatExt;

/// Per-scalar SIMD kernel for the inner factor / path loops. Implemented for
/// `f64` with `f64x4` (4-wide) and `f32` with `f32x8` (8-wide).
pub trait RoughSimd: FloatExt {
  /// Single-path factor reduction: $\sum_l (w_l e_l)\,(H_l + J_l)$ with
  /// `we[l] = w_l * e_l` pre-merged.
  fn history_sum_fused(we: &[Self], h_state: &[Self], j_state: &[Self]) -> Self;

  /// Single-path fused state update using pre-computed
  /// $\mathrm{omx}_l = (1 - e_l)/x_l$.
  fn update_state_fused(
    h_state: &mut [Self],
    j_state: &mut [Self],
    exp_neg: &[Self],
    omx: &[Self],
    f_prev: Self,
    g_dw: Self,
  );

  /// Batch path reduction for a single factor $l$:
  /// `history[p] += we_l * (h_row[p] + j_row[p])` for all paths $p$.
  fn batch_history_accumulate(we_l: Self, h_row: &[Self], j_row: &[Self], history: &mut [Self]);

  /// Batch state update for a single factor $l$:
  /// - `h_row[p] = e_l * h_row[p] + f_prev[p] * omx_l`
  /// - `j_row[p] = e_l * (g_dw[p] + j_row[p])`
  fn batch_update_state(
    e_l: Self,
    omx_l: Self,
    h_row: &mut [Self],
    j_row: &mut [Self],
    f_prev: &[Self],
    g_dw: &[Self],
  );
}

impl RoughSimd for f64 {
  #[inline]
  fn history_sum_fused(we: &[f64], h_state: &[f64], j_state: &[f64]) -> f64 {
    let n = we.len();
    let chunks = n / 4;
    let mut acc = f64x4::splat(0.0);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let we_v = load_f64x4(we, base);
        let h_v = load_f64x4(h_state, base);
        let j_v = load_f64x4(j_state, base);
        acc = we_v.mul_add(h_v + j_v, acc);
      }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..n {
      sum += we[i] * (h_state[i] + j_state[i]);
    }
    sum
  }

  #[inline]
  fn update_state_fused(
    h_state: &mut [f64],
    j_state: &mut [f64],
    exp_neg: &[f64],
    omx: &[f64],
    f_prev: f64,
    g_dw: f64,
  ) {
    let n = h_state.len();
    let chunks = n / 4;
    let f_v = f64x4::splat(f_prev);
    let gdw_v = f64x4::splat(g_dw);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let e = load_f64x4(exp_neg, base);
        let o = load_f64x4(omx, base);
        let h = load_f64x4(h_state, base);
        let j = load_f64x4(j_state, base);
        let h_new = e.mul_add(h, f_v * o);
        let j_new = e * (gdw_v + j);
        store_f64x4(h_state, base, h_new);
        store_f64x4(j_state, base, j_new);
      }
    }
    for i in (chunks * 4)..n {
      let e = exp_neg[i];
      let o = omx[i];
      h_state[i] = f_prev * o + e * h_state[i];
      j_state[i] = e * (g_dw + j_state[i]);
    }
  }

  #[inline]
  fn batch_history_accumulate(we_l: f64, h_row: &[f64], j_row: &[f64], history: &mut [f64]) {
    let m = history.len();
    let chunks = m / 4;
    let we_v = f64x4::splat(we_l);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let h = load_f64x4(h_row, base);
        let j = load_f64x4(j_row, base);
        let hist = load_f64x4(history, base);
        let new_hist = we_v.mul_add(h + j, hist);
        store_f64x4(history, base, new_hist);
      }
    }
    for i in (chunks * 4)..m {
      history[i] += we_l * (h_row[i] + j_row[i]);
    }
  }

  #[inline]
  fn batch_update_state(
    e_l: f64,
    omx_l: f64,
    h_row: &mut [f64],
    j_row: &mut [f64],
    f_prev: &[f64],
    g_dw: &[f64],
  ) {
    let m = h_row.len();
    let chunks = m / 4;
    let e_v = f64x4::splat(e_l);
    let omx_v = f64x4::splat(omx_l);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let h = load_f64x4(h_row, base);
        let j = load_f64x4(j_row, base);
        let fp = load_f64x4(f_prev, base);
        let gdw = load_f64x4(g_dw, base);
        let h_new = e_v.mul_add(h, fp * omx_v);
        let j_new = e_v * (gdw + j);
        store_f64x4(h_row, base, h_new);
        store_f64x4(j_row, base, j_new);
      }
    }
    for i in (chunks * 4)..m {
      h_row[i] = e_l * h_row[i] + f_prev[i] * omx_l;
      j_row[i] = e_l * (g_dw[i] + j_row[i]);
    }
  }
}

impl RoughSimd for f32 {
  #[inline]
  fn history_sum_fused(we: &[f32], h_state: &[f32], j_state: &[f32]) -> f32 {
    let n = we.len();
    let chunks = n / 8;
    let mut acc = f32x8::splat(0.0);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let we_v = load_f32x8(we, base);
        let h_v = load_f32x8(h_state, base);
        let j_v = load_f32x8(j_state, base);
        acc = we_v.mul_add(h_v + j_v, acc);
      }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..n {
      sum += we[i] * (h_state[i] + j_state[i]);
    }
    sum
  }

  #[inline]
  fn update_state_fused(
    h_state: &mut [f32],
    j_state: &mut [f32],
    exp_neg: &[f32],
    omx: &[f32],
    f_prev: f32,
    g_dw: f32,
  ) {
    let n = h_state.len();
    let chunks = n / 8;
    let f_v = f32x8::splat(f_prev);
    let gdw_v = f32x8::splat(g_dw);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let e = load_f32x8(exp_neg, base);
        let o = load_f32x8(omx, base);
        let h = load_f32x8(h_state, base);
        let j = load_f32x8(j_state, base);
        let h_new = e.mul_add(h, f_v * o);
        let j_new = e * (gdw_v + j);
        store_f32x8(h_state, base, h_new);
        store_f32x8(j_state, base, j_new);
      }
    }
    for i in (chunks * 8)..n {
      let e = exp_neg[i];
      let o = omx[i];
      h_state[i] = f_prev * o + e * h_state[i];
      j_state[i] = e * (g_dw + j_state[i]);
    }
  }

  #[inline]
  fn batch_history_accumulate(we_l: f32, h_row: &[f32], j_row: &[f32], history: &mut [f32]) {
    let m = history.len();
    let chunks = m / 8;
    let we_v = f32x8::splat(we_l);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let h = load_f32x8(h_row, base);
        let j = load_f32x8(j_row, base);
        let hist = load_f32x8(history, base);
        let new_hist = we_v.mul_add(h + j, hist);
        store_f32x8(history, base, new_hist);
      }
    }
    for i in (chunks * 8)..m {
      history[i] += we_l * (h_row[i] + j_row[i]);
    }
  }

  #[inline]
  fn batch_update_state(
    e_l: f32,
    omx_l: f32,
    h_row: &mut [f32],
    j_row: &mut [f32],
    f_prev: &[f32],
    g_dw: &[f32],
  ) {
    let m = h_row.len();
    let chunks = m / 8;
    let e_v = f32x8::splat(e_l);
    let omx_v = f32x8::splat(omx_l);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let h = load_f32x8(h_row, base);
        let j = load_f32x8(j_row, base);
        let fp = load_f32x8(f_prev, base);
        let gdw = load_f32x8(g_dw, base);
        let h_new = e_v.mul_add(h, fp * omx_v);
        let j_new = e_v * (gdw + j);
        store_f32x8(h_row, base, h_new);
        store_f32x8(j_row, base, j_new);
      }
    }
    for i in (chunks * 8)..m {
      h_row[i] = e_l * h_row[i] + f_prev[i] * omx_l;
      j_row[i] = e_l * (g_dw[i] + j_row[i]);
    }
  }
}

#[inline(always)]
unsafe fn load_f64x4(src: &[f64], base: usize) -> f64x4 {
  debug_assert!(base + 4 <= src.len());
  let ptr = unsafe { src.as_ptr().add(base) as *const [f64; 4] };
  f64x4::from(unsafe { *ptr })
}

#[inline(always)]
unsafe fn store_f64x4(dst: &mut [f64], base: usize, v: f64x4) {
  debug_assert!(base + 4 <= dst.len());
  let ptr = unsafe { dst.as_mut_ptr().add(base) as *mut [f64; 4] };
  unsafe { *ptr = v.to_array() };
}

#[inline(always)]
unsafe fn load_f32x8(src: &[f32], base: usize) -> f32x8 {
  debug_assert!(base + 8 <= src.len());
  let ptr = unsafe { src.as_ptr().add(base) as *const [f32; 8] };
  f32x8::from(unsafe { *ptr })
}

#[inline(always)]
unsafe fn store_f32x8(dst: &mut [f32], base: usize, v: f32x8) {
  debug_assert!(base + 8 <= dst.len());
  let ptr = unsafe { dst.as_mut_ptr().add(base) as *mut [f32; 8] };
  unsafe { *ptr = v.to_array() };
}
