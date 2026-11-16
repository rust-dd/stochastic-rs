//! `compile_fail` checks: an acronym in a public type name is one word (RFC 430).

/// ```compile_fail,E0432
/// use stochastic_rs_quant::pricing::bsm::BSMPricer;
/// ```
pub struct AcronymsAreOneWord;
