//! # Symbol glossary
//!
//! **Design principle.** This crate keeps literature-faithful, per-model
//! parameter names instead of forcing one global naming convention across
//! 140+ process types. The same Greek letter legitimately plays different
//! roles in different models: `theta` is a mean-reversion **speed** in
//! [`Ou`](crate::diffusion::ou::Ou) / [`Cir`](crate::diffusion::cir::Cir) /
//! [`Vasicek`](crate::interest::vasicek::Vasicek), but a long-run **variance
//! level** in [`Heston`](crate::volatility::heston::Heston); `alpha` alone
//! has around seven unrelated meanings in this crate (see below). Renaming
//! every colliding field to remove the ambiguity would break positional
//! constructor calls, diverge from the paper each model transcribes, and
//! erase the very naming convention practitioners expect to find.
//!
//! So instead: **every field's own `///` doc states its role in that
//! specific model**, and this module is the cross-model index for the
//! symbols that recur most. Read a field's own doc comment for the
//! authoritative answer; this page is a map to help you find the right
//! model, not a replacement for the field doc.
//!
//! Coverage note: this table only surveys `stochastic-rs-stochastic`. The
//! same letters accumulate still more meanings elsewhere in the workspace
//! (e.g. `alpha` as a CVaR confidence level in `stochastic-rs-quant`, or as
//! a copula shape parameter in `stochastic-rs-copulas`) — out of scope here.
//!
//! ## θ (theta)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Mean-reversion **speed** (κ in the model's own SDE) | [`Ou`](crate::diffusion::ou::Ou), [`Cir`](crate::diffusion::cir::Cir), [`Vasicek`](crate::interest::vasicek::Vasicek), [`FellerLogistic`](crate::diffusion::feller::FellerLogistic), [`Fou`](crate::diffusion::fou::Fou), [`Fcir`](crate::diffusion::fcir::Fcir), `JumpFou`/`JumpFOUCustom` (`theta`) |
//! | Long-run **variance level** (θ in the model's own SDE) | [`Heston`](crate::volatility::heston::Heston), [`RlHeston`](crate::rough::RlHeston), and the wider Heston family (`theta`) |
//! | Time-dependent additive drift **target function** θ(t), fitted to the initial term structure | [`HullWhite`](crate::interest::hull_white::HullWhite), [`HullWhite2F`](crate::interest::hull_white_2f::HullWhite2F) (`theta: Fn1D<T>`); the same role is played by [`Adg`](crate::interest::adg::Adg)'s `k` field, which is named differently |
//! | Jump-size **compensator** (κ/E\[Y−1\] in the model's own SDE), subtracted from the drift, scaled by `lambda` | [`Merton`](crate::jump::merton::Merton), [`Kou`](crate::jump::kou::Kou) (`theta`); the same role is played by [`Bates1996`](crate::jump::bates::Bates1996)'s `k` field |
//! | Skewness-in-subordinated-time (β in the model's own SDE) — **not** mean reversion; NIG has none | [`Nig`](crate::jump::nig::Nig) (`theta`) |
//!
//! `Ckls`'s `theta1..theta4` are a separate, non-colliding convention: the
//! Chan–Karolyi–Longstaff–Sanders paper itself numbers its four coefficients
//! θ₁–θ₄ (drift intercept, drift slope, diffusion scale, diffusion
//! elasticity) — not an instance of the single-θ ambiguity above.
//!
//! ## α (alpha)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Lévy tail/stability index Y ∈ (0, 2) | [`Cgmy`](crate::jump::cgmy::Cgmy), [`Cts`](crate::jump::cts::Cts), [`Rdts`](crate::jump::rdts::Rdts), `KoBoL` (`alpha`) |
//! | Drift rate μ (despite the name) | [`Merton`](crate::jump::merton::Merton), [`Kou`](crate::jump::kou::Kou) (`alpha`) |
//! | Linear-drift **intercept** (κθ combined) in a reparametrized `drift = alpha − beta·X` form | [`Bates1996`](crate::jump::bates::Bates1996)'s variance factor, [`FJacobi`](crate::diffusion::fjacobi::FJacobi), [`Jacobi`](crate::diffusion::jacobi::Jacobi) (`alpha`) |
//! | Constant (zeroth-order) term of a polynomial drift | [`Quadratic`](crate::diffusion::quadratic::Quadratic) (`alpha`) |
//! | Self-excitation jump magnitude added to the intensity on each event | [`Hawkes`](crate::process::hawkes::Hawkes), [`MultivariateHawkes`](crate::process::multivariate_hawkes::MultivariateHawkes), [`HawkesJD`](crate::jump::hawkes_jd::HawkesJD) (`alpha`) |
//! | Affine diffusion-loading coefficient (multiplies a state variable inside a shared volatility loading) | [`DuffieKanJumpExp`](crate::interest::duffie_kan_jump_exp::DuffieKanJumpExp) (`alpha`) |
//!
//! ## σ (sigma)
//!
//! Mostly consistent: the diffusion scale of a Brownian driver. Two notes:
//! - In the Heston family, `sigma` is the **vol-of-vol** — it scales the
//!   *variance* process's own noise, one level removed from the asset.
//! - In jump models built as Brownian-subordinated Lévy processes
//!   ([`Vg`](crate::jump::vg::Vg), [`Nig`](crate::jump::nig::Nig)), `sigma`
//!   scales only the Gaussian half of the model, not overall dispersion.
//!
//! `rl_fou`/`rl_heston` used to misname this `sigma` (their own doc comment
//! already said "$\nu$"); both now use `nu`, so no `sigma`/`nu` collision
//! remains in those two types.
//!
//! ## ν (nu)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Vol-of-vol (diffusion scale of a stochastic-volatility state) | [`Sabr`](crate::volatility::sabr::Sabr), [`MultifactorSabr`](crate::volatility::multifactor_sabr::MultifactorSabr), [`RlFOU`](crate::rough::RlFOU), [`RlHeston`](crate::rough::RlHeston), [`Bergomi`](crate::volatility::bergomi::Bergomi) (`nu`) |
//! | Variance rate of a gamma time-change (kurtosis/tail-thickness control) — not a volatility at all | [`Vg`](crate::jump::vg::Vg) (`nu`) |
//! | Mean of a log-jump-size distribution (a **location**, not a scale) | [`MjdLog`](crate::jump::mjd_log::MjdLog) (`nu`) |
//!
//! ## β (beta)
//!
//! | Role | Models (field) |
//! |---|---|
//! | CEV exponent (elasticity of the state's own volatility) | [`Sabr`](crate::volatility::sabr::Sabr), [`MultifactorSabr`](crate::volatility::multifactor_sabr::MultifactorSabr) (`beta`) |
//! | Linear-drift **slope** (mean-reversion speed κ), paired with `alpha`'s intercept role | [`Bates1996`](crate::jump::bates::Bates1996)'s variance factor, [`FJacobi`](crate::diffusion::fjacobi::FJacobi), [`Jacobi`](crate::diffusion::jacobi::Jacobi) (`beta`) |
//! | Linear coefficient of a polynomial drift | [`Quadratic`](crate::diffusion::quadratic::Quadratic) (`beta`) |
//! | Excitation-kernel decay rate (how fast a jump's influence fades) | [`Hawkes`](crate::process::hawkes::Hawkes), [`MultivariateHawkes`](crate::process::multivariate_hawkes::MultivariateHawkes) (`beta`) |
//! | Displacement / shift applied to the driven state before GBM-style dynamics act on it (`S_t + beta`) | [`DisplacedDiffusion`](crate::diffusion::displaced_diffusion::DisplacedDiffusion) (`beta`) |
//!
//! ## δ (delta)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Dimension of a (squared) Bessel process; δ ≥ 2 keeps it strictly positive | [`SquaredBessel`](crate::diffusion::bessel::SquaredBessel), [`Bessel`](crate::diffusion::bessel::Bessel) (`delta`) |
//! | Curvature of a hyperbolic restoring drift term | [`Hyperbolic2`](crate::diffusion::hyperbolic2::Hyperbolic2) (`delta`) |
//! | Scale of an Inverse-Gaussian subordinator | [`IGSubordinator`](crate::process::subordinator::ig_subordinator::IGSubordinator) (`delta`) |
//! | Per-lag negative-shock (leverage) coefficient vector, unrelated to the scalar roles above | [`Agarch`](crate::autoregressive::agrach::Agarch) (`delta`) |
//!
//! ## μ (mu)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Constant proportional drift rate — **no** mean reversion | [`Gbm`](crate::diffusion::gbm::Gbm), [`Fgbm`](crate::diffusion::fgbm::Fgbm), [`GbmIh`](crate::diffusion::gbm_ih::GbmIh), [`DisplacedDiffusion`](crate::diffusion::displaced_diffusion::DisplacedDiffusion) (`mu`) |
//! | Long-run mean **level** (θ in the model's own SDE), reverted toward by a separate speed field | [`Ou`](crate::diffusion::ou::Ou), [`Cir`](crate::diffusion::cir::Cir), [`Vasicek`](crate::interest::vasicek::Vasicek), [`Fou`](crate::diffusion::fou::Fou), [`Fcir`](crate::diffusion::fcir::Fcir), `JumpFou`/`JumpFOUCustom` (`mu`) |
//! | Drift-in-subordinated-time (θ in the model's own SDE) — a skewness parameter, not a level | [`Vg`](crate::jump::vg::Vg) (`mu`) |
//! | Baseline Poisson intensity vector — not a drift or level | [`MultivariateHawkes`](crate::process::multivariate_hawkes::MultivariateHawkes) (`mu`) |
//! | Location parameter inside a rational drift term (not a rate) | [`Hyperbolic2`](crate::diffusion::hyperbolic2::Hyperbolic2) (`mu`) |
//!
//! ## λ (lambda)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Jump (Poisson) intensity — arrival rate | [`Merton`](crate::jump::merton::Merton), [`Kou`](crate::jump::kou::Kou), [`Bates1996`](crate::jump::bates::Bates1996), [`MjdLog`](crate::jump::mjd_log::MjdLog) (`lambda`) |
//! | Lévy-density tempering rate (`lambda_plus`/`lambda_minus`, i.e. G/M) — an exponential decay rate of the jump-size density, not an arrival rate | [`Cgmy`](crate::jump::cgmy::Cgmy), [`Cts`](crate::jump::cts::Cts), [`Rdts`](crate::jump::rdts::Rdts), `KoBoL`, [`BilateralGamma`](crate::jump::bilateral_gamma::BilateralGamma) (`lambda_plus`/`lambda_minus` or `lambda_p`/`lambda_m`) |
//!
//! In the Hawkes family, `lambda_t` is the modeled self-exciting intensity
//! **state**, not an input parameter; the input baseline is a separate `mu`
//! field.
//!
//! ## κ (kappa)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Mean-reversion speed | [`Cir`](crate::diffusion::cir::Cir) (as `theta`), [`FellerLogistic`](crate::diffusion::feller::FellerLogistic), [`ThreeHalf`](crate::diffusion::three_half::ThreeHalf), [`Pearson`](crate::diffusion::pearson::Pearson) (`kappa`) |
//! | Jump-size compensator (E\[Y−1\]-like term), unrelated to speed | [`Bates1996`](crate::jump::bates::Bates1996) (`k`) |
//!
//! ## ρ (rho)
//!
//! | Role | Models (field) |
//! |---|---|
//! | Instantaneous correlation between two driving Brownian motions (an input constant) | [`Heston`](crate::volatility::heston::Heston), [`Sabr`](crate::volatility::sabr::Sabr), [`Bates1996`](crate::jump::bates::Bates1996), [`HullWhite2F`](crate::interest::hull_white_2f::HullWhite2F), [`DuffieKanJumpExp`](crate::interest::duffie_kan_jump_exp::DuffieKanJumpExp) (`rho`) |
//! | The modeled correlation **process** itself (an output, not an input) | [`TransformedOU`](crate::correlation::TransformedOU), [`VanEmmerich`](crate::correlation::VanEmmerich), [`TengSCP`](crate::correlation::TengSCP) (implicit; their `rho0` is the level it starts from / reverts to) |
//!
//! ## The x₀ family (x0 / s0 / v0 / f0 / r0 / alpha0 / …)
//!
//! All name an **initial value**, but of different state variables:
//!
//! | Field | Initial value of |
//! |---|---|
//! | `x0` | The generic primary state variable — used across most non-financially-typed processes |
//! | `s0` | Spot / asset price ([`Gbm`](crate::diffusion::gbm::Gbm), [`Heston`](crate::volatility::heston::Heston), [`Bates1996`](crate::jump::bates::Bates1996)) |
//! | `v0` | Variance / volatility state ([`Heston`](crate::volatility::heston::Heston), [`Bergomi`](crate::volatility::bergomi::Bergomi)) |
//! | `f0` | Forward rate/price, for models built around a forward rather than a spot ([`Sabr`](crate::volatility::sabr::Sabr), [`MultifactorSabr`](crate::volatility::multifactor_sabr::MultifactorSabr)) |
//! | `alpha0` | Volatility state, specifically in [`Sabr`](crate::volatility::sabr::Sabr) / [`MultifactorSabr`](crate::volatility::multifactor_sabr::MultifactorSabr) — named for the module's own α_t, not `v0`, after the Task 3 rename that resolved the `v0`-really-means-α₀ contradiction |
//! | `r0` | Short rate ([`Hjm`](crate::interest::hjm::Hjm), [`DuffieKanJumpExp`](crate::interest::duffie_kan_jump_exp::DuffieKanJumpExp)) |
//! | `p0` | Bond price / auxiliary level ([`Hjm`](crate::interest::hjm::Hjm)) |
//! | `y0` | Secondary state variable, paired with a primary `x0` ([`FouqueOU2D`](crate::diffusion::fouque::FouqueOU2D)) |
//! | `rho0` | Correlation level a stochastic-correlation process starts from / reverts to — the odd one out, since it anchors a correlation, not a price or rate ([`TransformedOU`](crate::correlation::TransformedOU), [`VanEmmerich`](crate::correlation::VanEmmerich)) |
//!
//! ## Numbered-coefficient conventions (not collisions)
//!
//! A handful of models transcribe a paper that itself indexes its
//! coefficients rather than naming them: [`Ckls`](crate::diffusion::ckls::Ckls)
//! (`theta1..theta4`), [`AitSahalia`](crate::diffusion::ait_sahalia::AitSahalia)
//! / `NonLinearSDE` (`am1, a0, a1, a2, b0..b3`), [`Pearson`](crate::diffusion::pearson::Pearson)
//! (`kappa, mu, a, b, c`), [`FellerRoot`](crate::diffusion::feller_root::FellerRoot)
//! (`theta1, theta2, theta3`). These are not instances of the cross-model
//! ambiguity documented above — each field's own doc states which term of
//! the model's own formula it multiplies.
