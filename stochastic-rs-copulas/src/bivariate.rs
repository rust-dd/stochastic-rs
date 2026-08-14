//! # Bivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
//! ## Choosing a copula
//!
//! Every family except [`independence::Independence`] is fit the same
//! way — [`BivariateExt::fit`] moment-matches Kendall's tau-b, then
//! inverts tau to the family's own shape parameter via `compute_theta` —
//! so *how* you fit is not a distinguishing axis. What actually
//! distinguishes these 13 is (1) which Kendall's tau values a family can
//! even represent, (2) the shape of its tail dependence, and (3) whether
//! fitting and sampling are closed-form or pay an iterative-solver cost.
//!
//! ### What Kendall's tau each family can represent
//!
//! | Reachable τ | Families |
//! |---|---|
//! | Full `(-1, 1)` | [`frank::Frank`], [`gaussian::GaussianCopula`], [`t_copula::TCopula`], [`plackett::Plackett`] (via an admittedly approximate τ ↔ Spearman's-ρ proxy — a few percent off, per that family's own module doc) |
//! | Positive only, `[0, 1)` | [`clayton::Clayton`], [`galambos::Galambos`], [`husler_reiss::HuslerReiss`], [`gumbel::Gumbel`], [`joe::Joe`] |
//! | Narrow, both signs | [`amh::Amh`] (`[-0.18, 0.33]`, asymmetric around 0), [`fgm::Fgm`] (`[-0.22, 0.22]`) |
//! | Degenerate (no free parameter in practice) | [`independence::Independence`] (`{0}` only); [`marshall_olkin::MarshallOlkin`] on its symmetric `theta` path (`[0, 1]`) — its separate `with_alpha_beta` constructor instead fits two independent shock-rate parameters directly from data, not from tau at all, and is the only family in this crate that is not exchangeable in general |
//!
//! Fitting a positive-only-τ family to negatively-dependent data is a real
//! failure mode, not a hypothetical: [`clayton::Clayton`]'s `compute_theta`
//! (`θ = 2τ/(1-τ)`) does not clamp, so negative τ silently produces a
//! negative, out-of-bounds `θ` that only surfaces as an error the first
//! time something calls [`BivariateExt::check_fit`] — not at `.fit()`
//! itself. Check your data's empirical τ sign before picking a family,
//! not after `.fit()` appears to succeed.
//!
//! ### Tail shape, and one heuristic to distrust
//!
//! Equal (here, zero) upper- and lower-tail dependence does **not** imply
//! radial symmetry. [`amh::Amh`] has zero tail dependence in both tails
//! yet is measurably not radially symmetric
//! (`C(u,v) ≠ u+v-1+C(1-u,1-v)`, confirmed numerically) — the in-tree
//! counterexample to that heuristic. Radially symmetric here:
//! [`fgm::Fgm`], [`frank::Frank`], [`gaussian::GaussianCopula`],
//! [`plackett::Plackett`], [`t_copula::TCopula`],
//! [`independence::Independence`]. Everything else — including
//! [`amh::Amh`] — is not.
//!
//! [`gumbel::Gumbel`] and [`joe::Joe`] have the identical upper-tail
//! formula ($\lambda_U = 2 - 2^{1/\theta}$, both requiring `θ ≥ 1`),
//! confirmed byte-for-byte the same expression in both source files. If
//! tail dependence is your only selection criterion, these two are
//! indistinguishable; they differ in Archimedean generator
//! ($(-\ln t)^\theta$ for Gumbel vs. $-\ln(1-(1-t)^\theta)$ for Joe) and
//! therefore in dependence shape away from the tail, but neither family's
//! own doc explains when that interior difference should drive a choice.
//! Treat "Gumbel or Joe" as a genuine tie unless an external reference or
//! a goodness-of-fit comparison points at one.
//!
//! ### Closed-form vs. iterative: fitting and sampling cost
//!
//! `compute_theta` (the tau → theta step inside `.fit()`) is closed-form
//! for [`clayton::Clayton`], [`fgm::Fgm`], [`gaussian::GaussianCopula`],
//! [`t_copula::TCopula`], [`marshall_olkin::MarshallOlkin`] and
//! [`gumbel::Gumbel`]; every other non-degenerate family
//! ([`amh::Amh`], [`frank::Frank`], [`galambos::Galambos`],
//! [`husler_reiss::HuslerReiss`], [`joe::Joe`], [`plackett::Plackett`])
//! root-finds it numerically. A one-time, cheap-in-absolute-terms cost
//! per `.fit()` call, but not free.
//!
//! [`BivariateExt::percent_point`] — the per-sample inversion
//! [`BivariateExt::sample`] relies on — is a separate cost, and fewer
//! families avoid it: only [`clayton::Clayton`],
//! [`gaussian::GaussianCopula`] and [`independence::Independence`] have a
//! real closed-form inverse. [`frank::Frank`] and [`gumbel::Gumbel`] each
//! override `percent_point`, but the override is closed-form only at the
//! degenerate boundary (`θ = 0` / `θ = 1`); every other `θ`, and every
//! other family — including [`t_copula::TCopula`] despite its closed-form
//! `partial_derivative` — falls through to the generic Brent-root
//! [`BivariateExt::percent_point_numerical`], one root-find per sampled
//! pair. For a Monte Carlo run sampling millions of pairs, that per-draw
//! cost is a real, practical reason to prefer Clayton or Gaussian when
//! either is otherwise an acceptable fit.
//!
//! ### The common data requirement, and one family that skips it
//!
//! `fit` requires both input columns already probability-integral-
//! transformed to `[0, 1]` — it runs a Kolmogorov-Smirnov uniformity
//! check (bound `1.627/√n`) on each column before touching Kendall's tau.
//! Feeding raw prices or returns fails this check; transform through an
//! empirical or fitted marginal CDF first (see
//! [`crate::univariate::gaussian::GaussianUnivariate`] or
//! [`crate::empirical`] for two ready-made transforms).
//! [`independence::Independence`]'s own `fit` is the one exception: it
//! hardcodes `tau = theta = 0.0` without reading the data at all, so it
//! never runs that check — relying on `.fit()` as an implicit
//! data-quality gate, even when independence is exactly what you expect,
//! does not work for this one family.
//!
//! ### A currently-open rough edge, reported rather than papered over
//!
//! [`frank::Frank`]'s `pdf` and `partial_derivative` build a denominator
//! as `g(u) + g(v) + g(1)` (a sum) where the closed form these methods'
//! own doc comments target needs `g(1) + g(u)·g(v)` (`g(z) = e^{-θz}-1`),
//! for every `θ` — not just at the `θ = 0` independence limit this
//! module's own tests cover, which is a disjoint special-cased path and
//! unaffected. Checked directly against a finite-difference/quadrature
//! probe of this family's own (correct) `cdf`: the gap is large — order
//! `0.1`-plus at `θ ∈ {0.5, 1, 2, 5, -3}` — not a rounding artifact.
//! `Frank::sample`/`percent_point` root-find through `partial_derivative`
//! and inherit the error. Not fixed here; needs its own test-driven pass
//! rather than a bundled edit.

pub use crate::traits::BivariateExt;

pub mod amh;
pub mod clayton;
pub mod fgm;
pub mod frank;
pub mod galambos;
pub mod gaussian;
pub mod gumbel;
pub mod husler_reiss;
pub mod independence;
pub mod joe;
pub mod marshall_olkin;
pub mod plackett;
pub mod t_copula;

#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
  Amh,
  Clayton,
  Fgm,
  Frank,
  Galambos,
  Gaussian,
  Gumbel,
  HuslerReiss,
  Independence,
  Joe,
  MarshallOlkin,
  Plackett,
  TCopula,
}

#[cfg(test)]
mod tests {
  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::frank::Frank;
  use crate::bivariate::gumbel::Gumbel;
  use crate::traits::BivariateExt;

  /// `BivariateExt::sample_with_seed` is a trait-default method — no
  /// bivariate family overrides it — so pin its determinism contract
  /// directly here, mirroring
  /// `multivariate::tests::every_multivariate_sampler_is_seedable_and_deterministic`.
  /// Same seed on the same object must replay bit-for-bit, and two
  /// independently-constructed, identically-seeded objects must agree.
  ///
  /// Frank and Gumbel are deliberately exercised at a non-boundary theta
  /// (not the `theta == 0.0` / `theta == 1.0` independence special case
  /// each family shortcuts around): both `percent_point` overrides used to
  /// call `BivariateExt::percent_point(self, ..)` via UFCS on their own
  /// non-boundary path, which — since each type overrides `percent_point`
  /// — resolved back to that same override instead of the trait's default
  /// body, recursing until the stack overflowed. Sampling at a real
  /// (non-degenerate) theta is what actually exercises that path; a test
  /// that only ever sampled at the independence boundary would never have
  /// caught either bug.
  #[test]
  fn every_bivariate_sampler_is_seedable_and_deterministic() {
    let mut clayton = Clayton::new();
    clayton.set_tau(0.5);
    clayton._compute_theta();
    assert_eq!(
      clayton.sample_with_seed(64, 42).unwrap(),
      clayton.sample_with_seed(64, 42).unwrap(),
      "Clayton replay"
    );
    let mut clayton_2 = Clayton::new();
    clayton_2.set_tau(0.5);
    clayton_2._compute_theta();
    assert_eq!(
      clayton.sample_with_seed(64, 42).unwrap(),
      clayton_2.sample_with_seed(64, 42).unwrap(),
      "Clayton cross-object"
    );

    let mut frank = Frank::new(None, Some(0.5));
    frank._compute_theta();
    assert_eq!(
      frank.sample_with_seed(64, 42).unwrap(),
      frank.sample_with_seed(64, 42).unwrap(),
      "Frank replay"
    );
    let mut frank_2 = Frank::new(None, Some(0.5));
    frank_2._compute_theta();
    assert_eq!(
      frank.sample_with_seed(64, 42).unwrap(),
      frank_2.sample_with_seed(64, 42).unwrap(),
      "Frank cross-object"
    );

    // theta = 1 / (1 - 0.5) = 2.0 — well past the theta == 1.0 independence
    // boundary that `Gumbel::percent_point` shortcuts around.
    let mut gumbel = Gumbel::new(None, Some(0.5));
    gumbel._compute_theta();
    assert_eq!(
      gumbel.sample_with_seed(64, 42).unwrap(),
      gumbel.sample_with_seed(64, 42).unwrap(),
      "Gumbel replay"
    );
    let mut gumbel_2 = Gumbel::new(None, Some(0.5));
    gumbel_2._compute_theta();
    assert_eq!(
      gumbel.sample_with_seed(64, 42).unwrap(),
      gumbel_2.sample_with_seed(64, 42).unwrap(),
      "Gumbel cross-object"
    );
  }
}
