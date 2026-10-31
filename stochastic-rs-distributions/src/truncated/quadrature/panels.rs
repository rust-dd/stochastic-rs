//! The panels of an interval: each branch point's ladder of rungs, the slope and curvature budgets, and the nodes.

use std::ops::ControlFlow;

use super::Kernel;

/// How far the log-kernel may move across one panel through its slope.
const SLOPE_BUDGET: f64 = 30.0;

/// How far the log-kernel may move across one panel through its curvature.
const CURVATURE_BUDGET: f64 = 8.0;

/// The 32 positive nodes of the 64-point Gauss–Legendre rule on `[-1, 1]`, ascending (mpmath, 40 digits).
pub(super) const GL64_X: [f64; 32] = [
  0.024350292663424433,
  0.07299312178779904,
  0.12146281929612054,
  0.16964442042399283,
  0.21742364374000708,
  0.2646871622087674,
  0.31132287199021097,
  0.3572201583376681,
  0.4022701579639916,
  0.4463660172534641,
  0.48940314570705296,
  0.5312794640198946,
  0.571895646202634,
  0.6111553551723933,
  0.6489654712546573,
  0.6852363130542333,
  0.7198818501716109,
  0.7528199072605319,
  0.7839723589433414,
  0.8132653151227975,
  0.8406292962525803,
  0.8659993981540928,
  0.8893154459951141,
  0.9105221370785028,
  0.9295691721319396,
  0.9464113748584028,
  0.9610087996520538,
  0.973326827789911,
  0.983336253884626,
  0.9910133714767443,
  0.9963401167719553,
  0.9993050417357722,
];

/// The weights of `GL64_X`; with the mirrored nodes they sum to 2.
pub(super) const GL64_W: [f64; 32] = [
  0.04869095700913972,
  0.04857546744150343,
  0.048344762234802954,
  0.04799938859645831,
  0.04754016571483031,
  0.04696818281621002,
  0.046284796581314416,
  0.04549162792741814,
  0.044590558163756566,
  0.04358372452932345,
  0.04247351512365359,
  0.04126256324262353,
  0.03995374113272034,
  0.038550153178615626,
  0.03705512854024005,
  0.035472213256882386,
  0.033805161837141606,
  0.03205792835485156,
  0.030234657072402478,
  0.028339672614259483,
  0.02637746971505466,
  0.024352702568710874,
  0.022270173808383253,
  0.02013482315353021,
  0.017951715775697343,
  0.015726030476024718,
  0.013463047896718643,
  0.011168139460131128,
  0.008846759826363947,
  0.006504457968978363,
  0.004147033260562468,
  0.0017832807216964328,
];

/// `[origin + l, origin + r]`; a graded panel's origin is its branch point, so its offsets stay exact beside it.
#[derive(Clone, Copy)]
pub(super) struct Panel {
  pub(super) origin: f64,
  pub(super) l: f64,
  pub(super) r: f64,
  pub(super) graded: bool,
  /// `p` on the panel beside a singular branch point (`p < 0`), which integrates in `u = (t/h)^{p+1}`.
  pub(super) singular: Option<f64>,
}

/// A breakpoint and, beside a branch point `s`, its exact offset from `s`.
#[derive(Clone, Copy)]
struct Cut {
  x: f64,
  from: Option<(f64, f64)>,
  /// On an end within a tenth of the width of a singular branch point `(s, p)`: its piece is the innermost.
  inner: Option<(f64, f64)>,
}

impl Cut {
  fn offset(self, origin: f64) -> f64 {
    match self.from {
      Some((s, t)) if s == origin => t,
      Some((s, t)) => (s - origin) + t,
      None => self.x - origin,
    }
  }
}

/// Visits the panels of `[l, r]` in order, or in reverse: the breakpoints (each branch point's ladder within 20 widths,
/// then the slope and curvature budgets).
pub(super) fn panels<K: Kernel, B>(
  kernel: &K,
  c: f64,
  l: f64,
  r: f64,
  rev: bool,
  mut visit: impl FnMut(Panel) -> ControlFlow<B>,
) -> ControlFlow<B> {
  let width = r - l;
  let ladders = kernel
    .branch_points()
    .map(|point| point.filter(|&(s, _)| s >= l - 20.0 * width && s <= r + 20.0 * width));
  let rungs = ladders.into_iter().flatten().flat_map(move |(s, p)| {
    // Beside a vanishing kernel, deep enough for the mass there to fall below 1e-18 and to reach an end that stops
    // short of `s`; beside a singular one 19 normal decades, below which the innermost panel's `u` is exact.
    let steps = if p < 0.0 {
      19
    } else {
      let gap = (l - s).max(s - r).max(0.0);
      let mass = (18.0 / (p + 1.0)).ceil() + 1.0;
      let reach = if gap > 0.0 {
        (width / gap).log10().ceil() + 1.0
      } else {
        0.0
      };
      mass.max(reach).min(330.0) as i32
    };
    let h = move |j: i32| width / 10_f64.powi(j);
    (1..=steps)
      .map(move |j| -h(j))
      .chain((1..=steps).rev().map(h))
      .filter(move |&t| l - s < t && t < r - s && (p > 0.0 || t.abs() >= f64::MIN_POSITIVE))
      .map(move |t| Cut {
        x: s + t,
        from: Some((s, t)),
        inner: None,
      })
  });
  let end = |x: f64| Cut {
    x,
    from: ladders
      .into_iter()
      .flatten()
      .min_by(|p, q| (x - p.0).abs().total_cmp(&(x - q.0).abs()))
      .map(|(s, _)| (s, x - s)),
    inner: ladders
      .into_iter()
      .flatten()
      .find(|&(s, p)| p < 0.0 && 10.0 * (x - s).abs() < width),
  };
  let cuts = std::iter::once(end(l))
    .chain(rungs)
    .chain(std::iter::once(end(r)));
  if rev {
    pieces(kernel, c, cuts.rev(), true, &mut visit)
  } else {
    pieces(kernel, c, cuts, false, &mut visit)
  }
}

fn pieces<K: Kernel, B>(
  kernel: &K,
  c: f64,
  cuts: impl Iterator<Item = Cut>,
  rev: bool,
  visit: &mut impl FnMut(Panel) -> ControlFlow<B>,
) -> ControlFlow<B> {
  let mut prev: Option<Cut> = None;
  for cut in cuts {
    if let Some(last) = prev {
      let (cl, cr) = if rev { (cut, last) } else { (last, cut) };
      piece(kernel, c, cl, cr, rev, visit)?;
    }
    prev = Some(cut);
  }
  ControlFlow::Continue(())
}

/// One piece between two cuts, in equal panels by the budgets, from the branch point nearer than the centre; the
/// innermost piece beside a singular branch point is one panel from it.
fn piece<K: Kernel, B>(
  kernel: &K,
  c: f64,
  cl: Cut,
  cr: Cut,
  rev: bool,
  visit: &mut impl FnMut(Panel) -> ControlFlow<B>,
) -> ControlFlow<B> {
  if let Some((s, p)) = cl.inner.or(cr.inner) {
    let (tl, tr) = (cl.offset(s), cr.offset(s));
    if tr <= tl {
      return ControlFlow::Continue(());
    }
    return visit(Panel {
      origin: s,
      l: tl,
      r: tr,
      graded: true,
      singular: Some(p),
    });
  }
  let distance = |z: f64| {
    if (cl.x..=cr.x).contains(&z) {
      0.0
    } else {
      (cl.x - z).abs().min((cr.x - z).abs())
    }
  };
  let branch = kernel
    .branch_points()
    .into_iter()
    .flatten()
    .map(|(s, _)| s)
    .find(|&s| distance(s) < 0.5 * distance(c));
  let origin = branch.unwrap_or(c);
  let (tl, tr) = (cl.offset(origin), cr.offset(origin));
  if tr <= tl {
    return ControlFlow::Continue(());
  }
  // A rate that overflows beside a branch point says nothing: the ladder already grades that piece.
  let rate = |d: fn(&K, f64) -> f64| {
    let ends = [cl.x, cr.x]
      .into_iter()
      .filter(|&x| kernel.ln_k(x).is_finite());
    ends
      .map(|x| d(kernel, x).abs())
      .filter(|v| v.is_finite())
      .fold(0.0, f64::max)
  };
  let h = (tr - tl)
    .min(SLOPE_BUDGET / rate(K::d1))
    .min(CURVATURE_BUDGET / rate(K::d2).sqrt());
  let n = ((tr - tl) / h).ceil();
  let n = if n >= 1.0 && n.is_finite() {
    n as usize
  } else {
    1
  };
  let at = |i: usize| {
    if i == n {
      tr
    } else {
      tl + (tr - tl) * i as f64 / n as f64
    }
  };
  for k in 0..n {
    let i = if rev { n - 1 - k } else { k };
    visit(Panel {
      origin,
      l: at(i),
      r: at(i + 1),
      graded: branch.is_some(),
      singular: None,
    })?;
  }
  ControlFlow::Continue(())
}
