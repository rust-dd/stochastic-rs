use super::*;

/// The kernels count a step's arrivals up to 64, exact to `f32` while `intensity · dt` is at most
/// 25, where the mass past 64 is about 2e-11; a process expecting more samples on the host.
pub(crate) fn count_fallback<T: FloatExt>(intensity: Option<T>, dt: T) -> Option<&'static str> {
  intensity
    .is_some_and(|rate| rate * dt > T::from_usize_(25))
    .then_some("more than 25 expected events per step, past the kernels' 64-event count")
}

/// How big a jump is: a normal or stable sum is one draw of the aggregated law; a one-per-step law
/// is one draw, any other is summed or compounded in a bounded loop.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[doc(hidden)]
pub enum JumpSizes<T: FloatExt> {
  /// Normal jump sizes, as Merton's and Bates's models take them.
  Normal { mean: T, sd: T },
  /// Kou's double-exponential sizes: up with probability `p_up` at rate
  /// `eta_up`, down otherwise at rate `eta_down`.
  DoubleExponential { p_up: T, eta_up: T, eta_down: T },
  /// A tempered-stable subordinator's sizes: a candidate `ε u^{−1/α}` kept
  /// with probability `exp(−μ·candidate)`, so the sum is over the accepted
  /// ones. `neg_inv_alpha` is `−1/α`, which depends on `α` alone.
  TemperedStable { eps: T, neg_inv_alpha: T, mu: T },
  /// Normal sizes compounded multiplicatively, `∏ (1 + mean + sd·z_j) − 1`
  /// over the step's jumps: what a jump that multiplies the state needs, and
  /// what a single aggregated normal cannot express once a step sees two.
  NormalProduct { mean: T, sd: T },
  /// [`DoubleExponential`](JumpSizes::DoubleExponential) sizes compounded
  /// multiplicatively, `∏ (1 + y_j) − 1`.
  DoubleExponentialProduct { p_up: T, eta_up: T, eta_down: T },
  /// Exactly one normal size per step whatever the count: what an
  /// event-indexed compound Poisson path draws, one size per arrival.
  OneNormal { mean: T, sd: T },
  /// Exactly one double-exponential size per step, for the same use.
  OneDoubleExponential { p_up: T, eta_up: T, eta_down: T },
  /// Sizes of `±scale` with equal probability, summed over the step's jumps.
  Rademacher { scale: T },
  /// Symmetric α-stable sizes, summed over the step's jumps in one draw: the
  /// sum of `n` of them is the same law at scale `scale · n^{1/α}`.
  SymmetricStable { alpha: T, scale: T },
  /// Exponential sizes of rate `rate`, summed over the step's jumps.
  Exponential { rate: T },
  /// [`Exponential`](JumpSizes::Exponential) sizes compounded multiplicatively, `∏ (1 + y_j) − 1`.
  ExponentialProduct { rate: T },
  /// Exactly one exponential size per step, for an event-indexed path.
  OneExponential { rate: T },
}

impl<T: FloatExt> JumpSizes<T> {
  /// The law code and the three scalars the kernels read it through. Public
  /// because [`EulerKernel`] is: a device implemented outside this crate has
  /// to pass the same five values its launch takes.
  pub fn encode(&self) -> (u32, T, T, T) {
    match *self {
      JumpSizes::Normal { mean, sd } => (1, mean, sd, T::zero()),
      JumpSizes::DoubleExponential {
        p_up,
        eta_up,
        eta_down,
      } => (2, p_up, eta_up, eta_down),
      JumpSizes::TemperedStable {
        eps,
        neg_inv_alpha,
        mu,
      } => (3, eps, neg_inv_alpha, mu),
      JumpSizes::NormalProduct { mean, sd } => (4, mean, sd, T::zero()),
      JumpSizes::DoubleExponentialProduct {
        p_up,
        eta_up,
        eta_down,
      } => (5, p_up, eta_up, eta_down),
      JumpSizes::OneNormal { mean, sd } => (6, mean, sd, T::zero()),
      JumpSizes::OneDoubleExponential {
        p_up,
        eta_up,
        eta_down,
      } => (7, p_up, eta_up, eta_down),
      JumpSizes::Rademacher { scale } => (8, scale, T::zero(), T::zero()),
      JumpSizes::SymmetricStable { alpha, scale } => (9, alpha, scale, T::zero()),
      JumpSizes::Exponential { rate } => (10, rate, T::zero(), T::zero()),
      JumpSizes::ExponentialProduct { rate } => (11, rate, T::zero(), T::zero()),
      JumpSizes::OneExponential { rate } => (12, rate, T::zero(), T::zero()),
    }
  }

  /// The same sizes compounded multiplicatively, `∏ (1 + y_j) − 1`, for a step whose jump
  /// multiplies the state; only the normal, double-exponential and exponential laws have one.
  pub fn product(self) -> Option<Self> {
    match self {
      JumpSizes::Normal { mean, sd } => Some(JumpSizes::NormalProduct { mean, sd }),
      JumpSizes::DoubleExponential {
        p_up,
        eta_up,
        eta_down,
      } => Some(JumpSizes::DoubleExponentialProduct {
        p_up,
        eta_up,
        eta_down,
      }),
      JumpSizes::Exponential { rate } => Some(JumpSizes::ExponentialProduct { rate }),
      _ => None,
    }
  }

  /// One size of this law per step, for an event-indexed path that jumps at every step; only the
  /// normal, double-exponential and exponential laws have such a form.
  pub fn single(self) -> Option<Self> {
    match self {
      JumpSizes::Normal { mean, sd } => Some(JumpSizes::OneNormal { mean, sd }),
      JumpSizes::DoubleExponential {
        p_up,
        eta_up,
        eta_down,
      } => Some(JumpSizes::OneDoubleExponential {
        p_up,
        eta_up,
        eta_down,
      }),
      JumpSizes::Exponential { rate } => Some(JumpSizes::OneExponential { rate }),
      _ => None,
    }
  }
}

/// The Gamma draws a family reads as `gm` and `gm2`, when its increment is a
/// gamma variate rather than a Gaussian one.
///
/// Each is one Marsaglia-Tsang draw of `Gamma(shape, scale)`, with the
/// shape-below-one boost that method prescribes. A process that needs one
/// draw declares [`second`](Self::second) as `None`; the bilateral gamma
/// processes, whose increment is the difference of two, declare both.
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct GammaDraws<T: FloatExt> {
  /// The shape, scale and per-jump shape of `gm`. The shape the kernel uses
  /// is `shape + per_jump · nj`, which is what a compound sum of gamma jumps
  /// needs: the sum of `k` of them is one draw whose shape is `k` times the
  /// single jump's. A process whose shape does not depend on the jump count
  /// leaves the third term at zero.
  pub first: (T, T, T),
  /// The same three for `gm2`, when the family reads a second draw.
  pub second: Option<(T, T, T)>,
}

impl<T: FloatExt> GammaDraws<T> {
  /// How many draws to take, and the two shape/scale/per-jump triples the
  /// kernels read. Public for the same reason [`JumpSizes::encode`] is.
  pub fn encode(&self) -> (u32, T, T, T, T, T, T) {
    let (s1, c1, p1) = self.first;
    match self.second {
      Some((s2, c2, p2)) => (2, s1, c1, p1, s2, c2, p2),
      None => (1, s1, c1, p1, T::zero(), T::zero(), T::zero()),
    }
  }
}
