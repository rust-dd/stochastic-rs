//! `autoregressive/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 129-type list, and shared methodology notes.

use ndarray::Array1;
use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::autoregressive::arch::Arch;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::ma::MAq;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::autoregressive::tgarch::GjrGarch;

use crate::common::N;
use crate::common::guard;

guard!(agarch, "Agarch", |s| Agarch::new(
  0.05,
  Array1::from(vec![0.1, 0.05]),
  Array1::from(vec![0.02, 0.01]),
  Array1::from(vec![0.8]),
  N,
  s
));

guard!(arp, "ARp", |s| ARp::new(
  Array1::from(vec![0.5, -0.2]),
  0.1,
  N,
  None,
  s
));

guard!(arch, "Arch", |s| Arch::new(
  0.05,
  Array1::from(vec![0.1, 0.05]),
  N,
  s
));

guard!(arima, "Arima", |s| Arima::new(
  Array1::from(vec![0.5]),
  Array1::from(vec![0.3]),
  1,
  0.1,
  N,
  s
));

guard!(egarch, "Egarch", |s| Egarch::new(
  0.05,
  Array1::from(vec![0.1, 0.05]),
  Array1::from(vec![0.02, 0.01]),
  Array1::from(vec![0.8]),
  N,
  s
));

guard!(garch, "Garch", |s| Garch::new(
  0.05,
  Array1::from(vec![0.1]),
  Array1::from(vec![0.8]),
  N,
  s
));

guard!(maq, "MAq", |s| MAq::new(
  Array1::from(vec![0.4, 0.2]),
  0.1,
  N,
  s
));

guard!(sarima, "Sarima", |s| Sarima::new(
  Array1::from(vec![0.4]),
  Array1::from(vec![0.2]),
  Array1::from(vec![0.1]),
  Array1::from(vec![0.1]),
  0,
  0,
  4,
  0.1,
  N,
  s
));

guard!(gjr_garch, "GjrGarch", |s| GjrGarch::new(
  0.05,
  Array1::from(vec![0.1, 0.05]),
  Array1::from(vec![0.02, 0.01]),
  Array1::from(vec![0.8]),
  N,
  s
));
