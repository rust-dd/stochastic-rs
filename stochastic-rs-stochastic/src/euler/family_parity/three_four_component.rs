use super::*;

/// One probe per three-component family.
pub(super) fn every_three_component_family() -> Vec<SystemProbe<3>> {
  let hjm = SystemProbe {
    spec: EulerSpec::HeathJarrowMorton,
    x0: [0.03, 1.0, 0.04],
    lift: None,
    program: None,
  };
  let garch = SystemProbe {
    spec: EulerSpec::Garch {
      omega: 0.00001,
      alpha: 0.1,
      beta: 0.85,
    },
    x0: [0.0, 0.0002, 0.0],
    lift: None,
    program: None,
  };
  let threshold = SystemProbe {
    spec: EulerSpec::ThresholdGarch {
      omega: 0.00001,
      alpha: 0.05,
      gamma: 0.1,
      beta: 0.85,
    },
    x0: [0.0, 0.0002, 0.0],
    lift: None,
    program: None,
  };
  let compound = SystemProbe {
    spec: EulerSpec::CompoundPoissonEvents { lambda: 5.0 },
    x0: [0.0, 0.0, 0.0],
    lift: None,
    program: None,
  };
  let wishart = SystemProbe {
    spec: EulerSpec::WishartTwo {
      m: [0.99, 0.01, 0.0, 0.98],
      theta: [1.0, 0.0, 0.3, 0.9],
      theta_inv: [1.0, 0.0, -1.0 / 3.0, 1.0 / 0.9],
      two: 1.0,
    },
    x0: [1.0, 0.2, 0.5],
    lift: None,
    program: None,
  };
  let double = |sym| {
    let spec = if sym {
      EulerSpec::DoubleHestonReflected {
        mu: 0.03,
        kappa1: 2.0,
        theta1: 0.04,
        sigma1: 0.3,
        rho1: -0.7,
        kappa2: 1.0,
        theta2: 0.02,
        sigma2: 0.2,
        rho2: -0.3,
      }
    } else {
      EulerSpec::DoubleHeston {
        mu: 0.03,
        kappa1: 2.0,
        theta1: 0.04,
        sigma1: 0.3,
        rho1: -0.7,
        kappa2: 1.0,
        theta2: 0.02,
        sigma2: 0.2,
        rho2: -0.3,
      }
    };
    SystemProbe {
      spec,
      x0: [100.0, 0.04, 0.02],
      lift: None,
      program: None,
    }
  };
  let exponential = SystemProbe {
    spec: EulerSpec::ExponentialGarch {
      omega: -0.2,
      alpha: 0.1,
      gamma: -0.05,
      beta: 0.95,
      e_abs_z: std::f32::consts::FRAC_2_PI.sqrt(),
    },
    x0: [0.0, -0.2, 0.0],
    lift: None,
    program: None,
  };
  vec![
    hjm,
    garch,
    threshold,
    exponential,
    compound,
    wishart,
    double(false),
    double(true),
    SystemProbe {
      spec: EulerSpec::StochasticCorrelationHeston {
        kappa_r: 1.0,
        mu_r: -0.3,
        sigma_r: 0.2,
        kappa_v: 2.0,
        mu_v: 0.04,
        sigma_v: 0.3,
        r: 0.02,
        rho2: 0.1,
      },
      x0: [100.0, 0.04, -0.3],
      lift: None,
      program: None,
    },
    // Sub-Feller, so both signed variances go below zero.
    SystemProbe {
      spec: EulerSpec::DoubleHestonFullTruncation([
        0.03, 0.5, 0.04, 1.0, -0.7, 0.4, 0.02, 0.8, -0.3,
      ]),
      x0: [100.0, 0.04, 0.02],
      lift: None,
      program: None,
    },
  ]
}

/// One probe per four-component family. The Bergomi variance is a function of the running sum
/// of its increments, so that sum and the elapsed time travel as two further components.
pub(super) fn every_four_component_family() -> Vec<SystemProbe<4>> {
  let two_asset = |sym| {
    let (mu1, mu2) = (0.03, 0.02);
    let (kappa1, theta1, sigma1) = (2.0, 0.04, 0.3);
    let (kappa2, theta2, sigma2) = (1.5, 0.03, 0.25);
    let (l11, l21, l22) = (1.0, -0.6, 0.8);
    let (l31, l32, l33) = (0.2, 0.1, 0.97);
    let (l41, l42, l43, l44) = (0.1, -0.2, 0.3, 0.92);
    let spec = if sym {
      EulerSpec::TwoAssetHestonReflected {
        mu1,
        mu2,
        kappa1,
        theta1,
        sigma1,
        kappa2,
        theta2,
        sigma2,
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        l41,
        l42,
        l43,
        l44,
      }
    } else {
      EulerSpec::TwoAssetHeston {
        mu1,
        mu2,
        kappa1,
        theta1,
        sigma1,
        kappa2,
        theta2,
        sigma2,
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        l41,
        l42,
        l43,
        l44,
      }
    };
    SystemProbe {
      spec,
      x0: [4.6, 0.04, 4.6, 0.03],
      lift: None,
      program: None,
    }
  };
  vec![
    SystemProbe {
      spec: EulerSpec::LiborMarket4 {
        sigma: [0.2, 0.25, 0.3, 0.35],
        delta: [0.5, 0.5, 0.5, 0.5],
        l: [1.0, 0.6, 0.8, 0.3, 0.2, 0.93, 0.1, 0.1, 0.2, 0.97],
      },
      x0: [0.03, 0.035, 0.04, 0.045],
      lift: None,
      program: None,
    },
    SystemProbe {
      spec: EulerSpec::CorrelatedGeometric4 {
        mu: [0.03, 0.02, 0.01, 0.04],
        sigma: [0.2, 0.3, 0.25, 0.15],
        l: [1.0, -0.6, 0.8, 0.2, 0.1, 0.97, 0.1, -0.2, 0.3, 0.92],
      },
      x0: [100.0, 50.0, 80.0, 60.0],
      lift: None,
      program: None,
    },
    SystemProbe {
      spec: EulerSpec::CorrelatedNoises4 {
        l: [1.0, -0.6, 0.8, 0.2, 0.1, 0.97, 0.1, -0.2, 0.3, 0.92],
      },
      x0: [0.0, 0.0, 0.0, 0.0],
      lift: None,
      program: None,
    },
    two_asset(false),
    two_asset(true),
    SystemProbe {
      spec: EulerSpec::HestonHullWhite2F {
        kappa: 0.8,
        theta: 0.2,
        sigma: 0.2,
        a: 1.1,
        b: 0.8,
        sigma1: 0.01,
        sigma2: 0.015,
        l: [1.0, -0.3, 0.95, 0.35, 0.37, 0.86, 0.08, -0.13, -0.44, 0.88],
      },
      x0: [0.0, 0.2, 0.03, 0.0],
      lift: None,
      program: None,
    },
    SystemProbe {
      spec: EulerSpec::HestonHullWhite2FQe {
        theta: 0.2,
        e_kd: 0.9968,
        c1: 0.0498,
        c2: 0.005,
        k0: 0.00095,
        k1: 1.4969,
        k2: -1.5031,
        a: 1.1,
        b: 0.8,
        sigma1: 0.01,
        sigma2: 0.015,
        l: [1.0, -0.4, 0.92, 0.35, 0.23, 0.86],
      },
      x0: [0.0, 0.2, 0.03, 0.0],
      lift: None,
      program: None,
    },
    SystemProbe {
      spec: EulerSpec::Bergomi {
        r: 0.02,
        nu: 0.5,
        half_nu_sq: 0.125,
        v0_sq: 0.04,
        rho: -0.6,
      },
      x0: [100.0, 0.04, 0.0, 0.0],
      lift: None,
      program: None,
    },
    // Sub-Feller, under a lower Cholesky factor with unit rows.
    SystemProbe {
      spec: EulerSpec::TwoAssetHestonFullTruncation([
        0.03, 0.02, 0.5, 0.04, 1.0, 0.4, 0.03, 0.8, 1.0, -0.6, 0.8, 0.3, 0.2, 0.93, 0.1, 0.1, 0.2,
        0.97,
      ]),
      x0: [4.6, 0.04, 4.6, 0.03],
      lift: None,
      program: None,
    },
  ]
}
