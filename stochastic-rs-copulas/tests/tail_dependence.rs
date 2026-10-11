//! Every family answers `tail_dependence` through the `Result` channel: `NotFitted` before a parameter is set and
//! `InvalidParameter` for a stored one outside its range (a NaN is outside every range), never a coefficient.

use std::collections::HashSet;

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

fn boxed(copula: impl BivariateExt + 'static) -> Box<dyn BivariateExt> {
  Box::new(copula)
}

#[test]
fn every_family_gates_tail_dependence_on_its_parameters() {
  let mut families = vec![
    boxed(Amh::new()),
    boxed(Bb1::new(None, None)),
    boxed(Bb7::new(None, None)),
    boxed(Clayton::new()),
    boxed(Fgm::new()),
    boxed(Frank::new(None)),
    boxed(Galambos::new()),
    boxed(GaussianCopula::new()),
    boxed(Gumbel::new(None)),
    boxed(HuslerReiss::new()),
    boxed(Independence::new()),
    boxed(Joe::new()),
    boxed(MarshallOlkin::new()),
    boxed(Plackett::new()),
    boxed(TCopula::new()),
  ];
  let covered = families.iter().map(|c| c.family()).collect::<HashSet<_>>();
  assert!(BivariateFamily::ALL.iter().all(|f| covered.contains(f)));
  for c in &mut families {
    assert_eq!(
      c.tail_dependence(),
      Err(CopulaError::NotFitted),
      "{:?}",
      c.family()
    );
    c.set_theta(f64::NAN);
    let got = c.tail_dependence();
    assert!(
      matches!(
        got,
        Err(CopulaError::InvalidParameter { name: "theta", .. })
      ),
      "{:?}: {got:?}",
      c.family()
    );
  }
}
