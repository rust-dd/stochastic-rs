//! Every family answers `tail_dependence` through the `Result` channel, `NotFitted` before its parameters are set
//! (Independence has none), and no family stores one outside its range, a NaN included.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::BivariateFamily;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::TailDependence;

fn boxed(copula: impl BivariateExt + 'static) -> Box<dyn BivariateExt> {
  Box::new(copula)
}

fn unfitted() -> Vec<Box<dyn BivariateExt>> {
  vec![
    boxed(Amh::default()),
    boxed(Bb1::default()),
    boxed(Bb7::default()),
    boxed(Clayton::default()),
    boxed(Fgm::default()),
    boxed(Frank::default()),
    boxed(Galambos::default()),
    boxed(GaussianCopula::default()),
    boxed(Gumbel::default()),
    boxed(HuslerReiss::default()),
    boxed(Independence::new()),
    boxed(Joe::default()),
    boxed(MarshallOlkin::default()),
    boxed(Plackett::default()),
    boxed(TCopula::default()),
  ]
}

#[test]
fn every_family_gates_tail_dependence_on_its_parameters() {
  let families = unfitted();
  let covered = families.iter().map(|c| c.family()).collect::<HashSet<_>>();
  assert!(BivariateFamily::ALL.iter().all(|f| covered.contains(f)));
  for c in &families {
    let want = if c.family() == BivariateFamily::Independence {
      Ok(TailDependence {
        lower: 0.0,
        upper: 0.0,
      })
    } else {
      Err(CopulaError::NotFitted)
    };
    assert_eq!(c.tail_dependence(), want, "{:?}", c.family());
  }
}

/// Every setter panics on a NaN `theta` with the parameter's message and keeps its state, so no gate ever sees one.
#[test]
fn no_family_stores_a_nan_theta() {
  for mut c in unfitted() {
    let (family, before) = (c.family(), c.theta());
    let refused = catch_unwind(AssertUnwindSafe(|| c.set_theta(f64::NAN))).unwrap_err();
    let message = refused
      .downcast_ref::<String>()
      .unwrap_or_else(|| panic!("{family:?}: not a formatted panic"));
    assert!(
      message.starts_with("theta must satisfy `") && message.ends_with("`, got theta = NaN"),
      "{family:?}: {message}"
    );
    assert_eq!(c.theta(), before, "{family:?} stored a theta");
  }
}
