//! Time-series hypothesis tests: unit-root/stationarity, structural-break,
//! random-walk, and functional-form misspecification. The OLS/GLS
//! regressions run on the pure-Rust `faer`; every
//! function here takes `ArrayView1<f64>` / `ArrayView2<f64>` directly, not
//! generic over [`crate::traits::FloatExt`] the way
//! [`crate::hurst::HurstEstimator`] is.
//!
//! # Nine tests, four different null hypotheses
//!
//! Every result type here implements [`crate::traits::HypothesisTest`], and
//! all nine live in this one module — but "stationarity" is an accurate
//! description of only five of them. Picking by module name instead of by
//! null hypothesis is how a reader ends up asking "which stationarity test
//! do I run" about [`reset::reset_test`], which does not examine a time
//! series at all.
//!
//! | Test | H₀ | Input | Min. obs. |
//! |---|---|---|---|
//! | [`adf::adf_test`] (ADF) | unit root | 1 series | 20 |
//! | [`kpss::kpss_test`] (KPSS) | (trend-)stationarity | 1 series | 20 |
//! | [`ers_dfgls::ers_dfgls_test`] (ERS/DF-GLS) | unit root | 1 series | 20 |
//! | [`phillips_perron::phillips_perron_test`] (PP) | unit root | 1 series | 20 |
//! | [`leybourne_mccabe::leybourne_mccabe_test`] (LM) | (trend-)stationarity | 1 series | 40 |
//! | [`andrews_ploberger::andrews_ploberger_test`] | regression-parameter constancy | series + design matrix | `3k+4` |
//! | [`cusum::cusum_test`] | regression-parameter constancy | series + design matrix | `k+2` |
//! | [`lo_mackinlay::lo_mackinlay_test`] | random walk (`VR(q)=1`) | 1 series | `4q` |
//! | [`reset::reset_test`] | correct linear functional form | series + design matrix | `k+q+2` |
//!
//! (`k` = design-matrix columns, `q` = aggregation horizon / extra powers.)
//!
//! ## Unit-root and stationarity tests on a single series
//!
//! The first five all ask some version of "does this series have a unit
//! root," but split on which answer is the null hypothesis:
//!
//! - **H₀ = unit root** — [`adf::adf_test`], [`ers_dfgls::ers_dfgls_test`],
//!   [`phillips_perron::phillips_perron_test`]. Failing to reject never
//!   proves a unit root; it only fails to rule one out.
//! - **H₀ = stationarity** — [`kpss::kpss_test`],
//!   [`leybourne_mccabe::leybourne_mccabe_test`]. Failing to reject never
//!   proves stationarity, for the same reason in reverse.
//!
//! Kwiatkowski, Phillips, Schmidt & Shin (1992) built KPSS specifically to
//! pair with ADF for this reason (*Journal of Econometrics* 54(1-3),
//! 159-178, DOI: 10.1016/0304-4076(92)90104-Y) — run both and read the
//! combination rather than trusting either alone:
//!
//! | ADF (H₀: unit root) | KPSS (H₀: stationary) | Reading |
//! |---|---|---|
//! | rejects | fails to reject | stationary — both agree |
//! | fails to reject | rejects | unit root — both agree |
//! | rejects | rejects | contradiction, often long-range dependence rather than a clean I(0)/I(1) split — see [`crate::hurst::HurstEstimator`] for estimating `H` directly instead of forcing a binary unit-root call |
//! | fails to reject | fails to reject | inconclusive, usually a power problem rather than evidence either way |
//!
//! Within the unit-root-null group, [`ers_dfgls::ers_dfgls_test`] is the
//! usual first choice, not [`adf::adf_test`]: Elliott, Rothenberg & Stock
//! (1996) GLS-detrend before testing, specifically to recover power that
//! OLS-detrended ADF loses against near-unit-root alternatives
//! (*Econometrica* 64(4), 813-836, DOI: 10.2307/2171846). It is the wrong
//! choice, though, whenever the series has no deterministic component at
//! all: [`ers_dfgls::ErsTrend`] offers only `Constant`/`ConstantTrend`,
//! while [`common::DeterministicTerm`]'s `None` variant lets
//! [`adf::adf_test`] test a zero-mean random walk directly.
//! [`phillips_perron::phillips_perron_test`] answers the same null with a
//! nonparametric (Newey-West) correction in place of ADF's lag
//! augmentation (Phillips & Perron 1988, *Biometrika* 75(2), 335-346, DOI:
//! 10.1093/biomet/75.2.335) — a genuine alternative, not a tuning variant
//! of ADF, but Schwert (1989) is the classic Monte Carlo study showing it
//! size-distorts under strong negative MA errors (*Journal of Business &
//! Economic Statistics* 7(2), 147-159, DOI: 10.1080/07350015.1989.10509723)
//! — precisely the regime ERS-DFGLS was designed to handle better. This
//! crate's own [`common::schwert_max_lags`] lag-length rule, the default
//! across ADF, ERS-DFGLS, KPSS and PP alike, is named after that same
//! paper.
//!
//! Within the stationarity-null group,
//! [`leybourne_mccabe::leybourne_mccabe_test`] swaps KPSS's nonparametric
//! long-run-variance correction for parametric AR prewhitening plus a
//! bootstrap p-value (Leybourne & McCabe 1994, *Journal of Business &
//! Economic Statistics* 12(2), 157-166) — more robust when residuals carry
//! strong autocorrelation that a Newey-West correction handles poorly, at
//! the cost of twice KPSS's minimum sample (40 vs. 20), an explicit AR
//! lag-order choice, and real compute for the bootstrap (400 resamples by
//! default).
//!
//! ADF's own statistic traces to Dickey & Fuller (1979), *Journal of the
//! American Statistical Association* 74(366), 427-431, DOI:
//! 10.2307/2286348; the augmentation lags that make it "Augmented" — so it
//! remains valid under general ARMA errors instead of only a clean AR(1) —
//! are due to Said & Dickey (1984), *Biometrika* 71(3), 599-607.
//!
//! ## Structural-break tests on a regression
//!
//! [`andrews_ploberger::andrews_ploberger_test`] and [`cusum::cusum_test`]
//! both require a design matrix `x`, not just a series — they test whether
//! a *regression's coefficients* stay constant, a different question from
//! "does this series have a unit root" even when `x` is only an intercept
//! column. Reaching for either to answer a unit-root question is a
//! category error, not a tuning choice.
//!
//! Between the two: [`andrews_ploberger::andrews_ploberger_test`] searches
//! for one unknown breakpoint and reports where it is (Andrews 1993,
//! *Econometrica* 61(4), 821-856; Andrews & Ploberger 1994, *Econometrica*
//! 62(6), 1383-1414) — the right tool when a single discrete regime change
//! is suspected. [`cusum::cusum_test`] instead walks a cumulative
//! recursive-residual path against a boundary (Brown, Durbin & Evans 1975,
//! *JRSS B* 37(2), 149-192) — older, cheaper, and better suited to
//! monitoring for gradual or contemporaneous drift than to pinpointing one
//! sharp break. Choosing CUSUM over CUSUMQ ([`cusum::CusumVariant`]) *is*
//! genuinely just a tuning choice within the same test: CUSUM catches mean
//! shifts, CUSUMQ catches variance shifts, both from the same
//! recursive-residual path.
//!
//! ## Random-walk and functional-form tests
//!
//! [`lo_mackinlay::lo_mackinlay_test`] tests something strictly stronger
//! than a unit root: that increments are *uncorrelated* at horizon `q`,
//! not merely that the level is `I(1)` (Lo & MacKinlay 1988, *Review of
//! Financial Studies* 1(1), 41-66). A series can carry a unit root and
//! still fail this test — an ARIMA(1,1,0) with a non-zero AR coefficient,
//! for instance. Prefer the heteroskedasticity-robust statistic
//! (`z_robust`, Lo-MacKinlay's `M2`) over the IID one (`z_iid`, `M1`)
//! whenever volatility clustering is plausible, which for financial
//! returns is essentially always;
//! [`lo_mackinlay::LoMacKinlayResult`]'s own [`crate::traits::HypothesisTest`]
//! impl already defaults to the robust statistic for exactly this reason.
//!
//! [`reset::reset_test`] is not a time-series test at all: Ramsey (1969)
//! designed it to catch omitted nonlinearity in a linear regression's
//! functional form (*JRSS B* 31(2), 350-371), not persistence or memory in
//! a series. It has no series-only entry point — it needs a design matrix
//! like the structural-break tests do — and reaching for RESET to answer
//! "which stationarity test should I run" is the clearest version of the
//! by-module-name mistake this doc exists to head off.

pub mod common;

pub use common::DeterministicTerm;
pub use common::LagSelection;

pub mod adf;
pub mod andrews_ploberger;
pub mod cusum;
pub mod ers_dfgls;
pub mod kpss;
pub mod leybourne_mccabe;
pub mod lo_mackinlay;
pub mod phillips_perron;
pub mod reset;
