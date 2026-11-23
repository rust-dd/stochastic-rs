//! `compile_fail` checks of names outside the public API: an acronym's old spelling (RFC 430) and
//! the SABR smile's internal pricing helpers.

/// ```compile_fail,E0432
/// use stochastic_rs_quant::pricing::bsm::BSMPricer;
/// ```
pub struct AcronymsAreOneWord;

/// ```compile_fail,E0603
/// use stochastic_rs_quant::pricing::sabr::bs_price_fx;
/// ```
pub struct BsPriceFxIsInternal;

/// ```compile_fail,E0603
/// use stochastic_rs_quant::pricing::sabr::model_price_hagan;
/// ```
pub struct ModelPriceHaganIsInternal;
