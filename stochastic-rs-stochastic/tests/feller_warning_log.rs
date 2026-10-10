//! Boundary-condition warnings go through the `log` facade, not stderr.

use std::hint::black_box;
use std::sync::Mutex;

use log::Level;
use log::LevelFilter;
use log::Log;
use log::Metadata;
use log::Record;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;

struct Capture(Mutex<Vec<(Level, String)>>);

impl Log for Capture {
  fn enabled(&self, _: &Metadata) -> bool {
    true
  }

  fn log(&self, record: &Record) {
    let mut records = self.0.lock().unwrap();
    records.push((record.level(), record.args().to_string()));
  }

  fn flush(&self) {}
}

static CAPTURE: Capture = Capture(Mutex::new(Vec::new()));

fn zero(_t: f64) -> f64 {
  0.0
}

fn short_rate(_t: f64) -> f64 {
  0.05
}

fn drain() -> Vec<(Level, String)> {
  std::mem::take(&mut *CAPTURE.0.lock().unwrap())
}

#[test]
fn boundary_violations_warn_through_log_unless_reflected() {
  log::set_logger(&CAPTURE).unwrap();
  log::set_max_level(LevelFilter::Warn);

  let build = |use_sym: Option<bool>| {
    black_box(Cir::<f64>::new(
      0.5, 0.04, 1.0, 16, None, None, use_sym, Unseeded,
    ));
    black_box(Fcir::<f64>::new(
      0.7, 0.5, 0.04, 1.0, 16, None, None, use_sym, Unseeded,
    ));
    black_box(CirPlusPlus::<f64>::new(
      0.5,
      0.04,
      1.0,
      zero as fn(f64) -> f64,
      16,
      None,
      None,
      use_sym,
      Unseeded,
    ));
    black_box(SquaredBessel::<f64>::new(
      1.0, 16, None, None, use_sym, Unseeded,
    ));
    black_box(Bessel::<f64>::new(1.0, 16, None, None, use_sym, Unseeded));
    black_box(Heston2D::<f64>::new(
      [Some(100.0); 2],
      [Some(0.04); 2],
      [0.05; 2],
      [0.04; 2],
      [0.5; 2],
      [1.0; 2],
      [0.0; 6],
      16,
      None,
      use_sym,
      Unseeded,
    ));
  };

  build(Some(true));
  assert!(drain().is_empty());

  build(None);
  let records = drain();
  assert_eq!(records.len(), 7, "{records:?}");
  assert!(records.iter().all(|(level, _)| *level == Level::Warn));

  black_box(BlackKarasinski::<f64>::new(
    short_rate as fn(f64) -> f64,
    0.0,
    0.2,
    16,
    None,
    None,
    Unseeded,
  ));
  assert_eq!(drain().len(), 1);

  black_box(Cir::<f64>::new(
    2.0, 0.04, 0.2, 16, None, None, None, Unseeded,
  ));
  assert!(drain().is_empty());
}
