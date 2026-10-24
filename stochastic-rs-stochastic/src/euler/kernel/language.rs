//! CUDA scalar types and math intrinsics for single and double precision.

pub(crate) struct Language<'a> {
  pub real: &'a str,
  pub sqrt: &'a str,
  pub log: &'a str,
  pub cos: &'a str,
  pub fast_log: &'a str,
  pub fast_cos: &'a str,
  pub sin: &'a str,
  pub exp: &'a str,
  pub pow: &'a str,
  pub abs: &'a str,
  pub tanh: &'a str,
  pub atan: &'a str,
  pub index: &'a str,
  pub wide: &'a str,
}

pub(crate) fn cuda_language(real: &'static str) -> Language<'static> {
  if real == "float" {
    Language {
      real,
      sqrt: "sqrtf",
      log: "logf",
      cos: "cosf",
      fast_log: "__logf",
      fast_cos: "__cosf",
      sin: "sinf",
      exp: "expf",
      pow: "powf",
      abs: "fabsf",
      tanh: "tanhf",
      atan: "atanf",
      index: "unsigned long long",
      wide: "unsigned long long",
    }
  } else {
    Language {
      real,
      sqrt: "sqrt",
      log: "log",
      fast_log: "log",
      fast_cos: "cos",
      cos: "cos",
      sin: "sin",
      exp: "exp",
      pow: "pow",
      abs: "fabs",
      tanh: "tanh",
      atan: "atan",
      index: "unsigned long long",
      wide: "unsigned long long",
    }
  }
}

pub(crate) fn prelude(lang: &Language<'_>) -> String {
  substitute(crate::euler::families::vocabulary::C_PRELUDE, lang)
}

pub(super) fn substitute(text: &str, lang: &Language<'_>) -> String {
  text
    .replace("INDEX", lang.index)
    .replace("U64", lang.wide)
    .replace("STOCH_SQRT", lang.sqrt)
    // Random draws use CUDA fast intrinsics; family formulas keep the accurate ones.
    .replace("STOCH_FAST_LOG", lang.fast_log)
    .replace("STOCH_FAST_COS", lang.fast_cos)
    .replace("STOCH_LOG", lang.log)
    .replace("STOCH_COS", lang.cos)
    .replace("STOCH_SIN", lang.sin)
    .replace("STOCH_EXP", lang.exp)
    .replace("STOCH_POW", lang.pow)
    .replace("STOCH_ABS", lang.abs)
    .replace("STOCH_TANH", lang.tanh)
    .replace("STOCH_ATAN", lang.atan)
    .replace("REAL", lang.real)
}
