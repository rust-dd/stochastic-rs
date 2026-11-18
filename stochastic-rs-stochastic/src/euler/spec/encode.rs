use super::*;

impl<T: FloatExt> EulerSpec<T> {
  /// Family code and the parameter slots the device kernels read, the kernels' ABI: it stays
  /// inside the crate, so it can widen. A build without a device feature has no caller.
  #[cfg_attr(not(device_backend), allow(dead_code))]
  pub(crate) fn encode(&self) -> (u32, [T; PARAM_SLOTS]) {
    use families::Family;
    match *self {
      EulerSpec::GeometricBrownian { mu, sigma } => {
        (Family::GeometricBrownian.code(), pad([mu, sigma]))
      }
      EulerSpec::OrnsteinUhlenbeck { theta, mu, sigma } => {
        (Family::OrnsteinUhlenbeck.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::SquareRoot {
        kappa,
        theta,
        sigma,
      } => (Family::SquareRoot.code(), pad([kappa, theta, sigma])),
      EulerSpec::Additive => (Family::Additive.code(), pad([])),
      EulerSpec::ReflectedSquareRoot { theta, mu, sigma } => {
        (Family::ReflectedSquareRoot.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::MirroredSquareRoot { theta, mu, sigma } => {
        (Family::MirroredSquareRoot.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::Jacobi { alpha, beta, sigma } => {
        (Family::Jacobi.code(), pad([alpha, beta, sigma]))
      }
      EulerSpec::ConstantElasticity { mu, sigma, gamma } => {
        (Family::ConstantElasticity.code(), pad([mu, sigma, gamma]))
      }
      EulerSpec::Ckls {
        theta1,
        theta2,
        theta3,
        theta4,
      } => (Family::Ckls.code(), pad([theta1, theta2, theta3, theta4])),
      EulerSpec::Logistic { a, b } => (Family::Logistic.code(), pad([a, b])),
      EulerSpec::ThreeHalf { kappa, mu, sigma } => {
        (Family::ThreeHalf.code(), pad([kappa, mu, sigma]))
      }
      EulerSpec::LogGeometric { drift_ln, sigma } => {
        (Family::LogGeometric.code(), pad([drift_ln, sigma]))
      }
      EulerSpec::RadialOrnsteinUhlenbeck { kappa, sigma } => {
        (Family::RadialOrnsteinUhlenbeck.code(), pad([kappa, sigma]))
      }
      EulerSpec::LinearSde { a, b, c } => (Family::LinearSde.code(), pad([a, b, c])),
      EulerSpec::Hyperbolic { kappa, sigma } => (Family::Hyperbolic.code(), pad([kappa, sigma])),
      EulerSpec::ModifiedSquareRoot { kappa, sigma } => {
        (Family::ModifiedSquareRoot.code(), pad([kappa, sigma]))
      }
      EulerSpec::FellerRoot {
        theta1,
        decay,
        theta3,
      } => (Family::FellerRoot.code(), pad([theta1, decay, theta3])),
      EulerSpec::AitSahalia {
        am1,
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
        b3,
      } => (
        Family::AitSahalia.code(),
        pad([am1, a0, a1, a2, b0, b1, b2, b3]),
      ),
      EulerSpec::Gompertz { a, b, sigma } => (Family::Gompertz.code(), pad([a, b, sigma])),
      EulerSpec::Kimura { a, sigma } => (Family::Kimura.code(), pad([a, sigma])),
      EulerSpec::Quadratic {
        alpha,
        beta,
        gamma,
        sigma,
      } => (Family::Quadratic.code(), pad([alpha, beta, gamma, sigma])),
      EulerSpec::Pearson {
        kappa,
        mu,
        a,
        b,
        c,
        two_kappa,
      } => (Family::Pearson.code(), pad([kappa, mu, a, b, c, two_kappa])),
      EulerSpec::Verhulst { r, k, sigma } => (Family::Verhulst.code(), pad([r, k, sigma])),
      EulerSpec::VerhulstClamped { r, k, sigma } => {
        (Family::VerhulstClamped.code(), pad([r, k, sigma]))
      }
      EulerSpec::FellerLogistic {
        kappa,
        theta,
        sigma,
      } => (Family::FellerLogistic.code(), pad([kappa, theta, sigma])),
      EulerSpec::FellerLogisticReflected {
        kappa,
        theta,
        sigma,
      } => (
        Family::FellerLogisticReflected.code(),
        pad([kappa, theta, sigma]),
      ),
      EulerSpec::SquaredBesselState { delta, two } => {
        (Family::SquaredBesselState.code(), pad([delta, two]))
      }
      EulerSpec::SquaredBesselStateReflected { delta, two } => (
        Family::SquaredBesselStateReflected.code(),
        pad([delta, two]),
      ),
      EulerSpec::BesselFromSquared { delta, two } => {
        (Family::BesselFromSquared.code(), pad([delta, two]))
      }
      EulerSpec::BesselFromSquaredReflected { delta, two } => {
        (Family::BesselFromSquaredReflected.code(), pad([delta, two]))
      }
      EulerSpec::HyperbolicDiffusion {
        beta,
        gamma,
        delta,
        mu,
        sigma,
        half_var,
      } => (
        Family::HyperbolicDiffusion.code(),
        pad([beta, gamma, delta, mu, sigma, half_var]),
      ),
      EulerSpec::NonLinear {
        am1,
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
        b3,
      } => (
        Family::NonLinear.code(),
        pad([am1, a0, a1, a2, b0, b1, b2, b3]),
      ),
      EulerSpec::Displaced { mu, sigma, beta } => {
        (Family::Displaced.code(), pad([mu, sigma, beta]))
      }
      EulerSpec::TanhOrnsteinUhlenbeck { kappa, mu, sigma } => (
        Family::TanhOrnsteinUhlenbeck.code(),
        pad([kappa, mu, sigma]),
      ),
      EulerSpec::BoundedCorrelation { kappa, mu, sigma } => {
        (Family::BoundedCorrelation.code(), pad([kappa, mu, sigma]))
      }
      EulerSpec::Heston {
        mu,
        kappa,
        theta,
        sigma,
        rho,
        pow_v,
      } => (
        Family::Heston.code(),
        pad([mu, kappa, theta, sigma, rho, pow_v]),
      ),
      EulerSpec::HestonReflected {
        mu,
        kappa,
        theta,
        sigma,
        rho,
        pow_v,
      } => (
        Family::HestonReflected.code(),
        pad([mu, kappa, theta, sigma, rho, pow_v]),
      ),
      EulerSpec::Sabr {
        beta,
        rho,
        nu,
        half_nu_sq,
      } => (Family::Sabr.code(), pad([beta, rho, nu, half_nu_sq])),
      EulerSpec::Bergomi {
        r,
        nu,
        half_nu_sq,
        v0_sq,
        rho,
      } => (Family::Bergomi.code(), pad([r, nu, half_nu_sq, v0_sq, rho])),
      EulerSpec::TwoScaleOrnsteinUhlenbeck {
        kappa,
        theta,
        eps,
        alpha,
        eps_inv,
        sqrt_eps_inv,
      } => (
        Family::TwoScaleOrnsteinUhlenbeck.code(),
        pad([kappa, theta, eps, alpha, eps_inv, sqrt_eps_inv]),
      ),
      EulerSpec::LogHeston {
        drift,
        kappa,
        theta,
        xi,
        rho,
      } => (
        Family::LogHeston.code(),
        pad([drift, kappa, theta, xi, rho]),
      ),
      EulerSpec::LogHestonReflected {
        drift,
        kappa,
        theta,
        xi,
        rho,
      } => (
        Family::LogHestonReflected.code(),
        pad([drift, kappa, theta, xi, rho]),
      ),
      EulerSpec::DoubleHeston {
        mu,
        kappa1,
        theta1,
        sigma1,
        rho1,
        kappa2,
        theta2,
        sigma2,
        rho2,
      } => (
        Family::DoubleHeston.code(),
        pad([
          mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2,
        ]),
      ),
      EulerSpec::DoubleHestonReflected {
        mu,
        kappa1,
        theta1,
        sigma1,
        rho1,
        kappa2,
        theta2,
        sigma2,
        rho2,
      } => (
        Family::DoubleHestonReflected.code(),
        pad([
          mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2,
        ]),
      ),
      EulerSpec::StochasticCorrelationHeston {
        kappa_r,
        mu_r,
        sigma_r,
        kappa_v,
        mu_v,
        sigma_v,
        r,
        rho2,
      } => (
        Family::StochasticCorrelationHeston.code(),
        pad([kappa_r, mu_r, sigma_r, kappa_v, mu_v, sigma_v, r, rho2]),
      ),
      EulerSpec::HullWhite { alpha, sigma } => (Family::HullWhite.code(), pad([alpha, sigma])),
      EulerSpec::CurveDrift { sigma } => (Family::CurveDrift.code(), pad([sigma])),
      EulerSpec::LogMeanReverting {
        decay,
        pull,
        sigma_eff,
      } => (
        Family::LogMeanReverting.code(),
        pad([decay, pull, sigma_eff]),
      ),
      EulerSpec::ShiftedSquareRoot { theta, mu, sigma } => {
        (Family::ShiftedSquareRoot.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::ShiftedSquareRootMirrored { theta, mu, sigma } => (
        Family::ShiftedSquareRootMirrored.code(),
        pad([theta, mu, sigma]),
      ),
      EulerSpec::TimeVaryingGeometricBrownian { mu } => {
        (Family::TimeVaryingGeometricBrownian.code(), pad([mu]))
      }
      EulerSpec::CorrelatedBrownian { rho } => (Family::CorrelatedBrownian.code(), pad([rho])),
      EulerSpec::BrownianBridge { xt, sigma } => (Family::BrownianBridge.code(), pad([xt, sigma])),
      EulerSpec::TwoFactorHullWhite {
        a,
        b,
        sigma1,
        sigma2,
        rho,
      } => (
        Family::TwoFactorHullWhite.code(),
        pad([a, b, sigma1, sigma2, rho]),
      ),
      EulerSpec::TwoFactorSquareRoot {
        theta1,
        mu1,
        sigma1,
        theta2,
        mu2,
        sigma2,
        sym1,
        sym2,
      } => (
        Family::TwoFactorSquareRoot.code(),
        pad([theta1, mu1, sigma1, theta2, mu2, sigma2, sym1, sym2]),
      ),
      EulerSpec::DuffieKan {
        a1,
        b1,
        c1,
        sigma1,
        a2,
        b2,
        c2,
        sigma2,
        alpha,
        beta,
        gamma,
        rho,
      } => (
        Family::DuffieKan.code(),
        pad([
          a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho,
        ]),
      ),
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
      } => (
        Family::TwoAssetHeston.code(),
        pad([
          mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2, l11, l21, l22, l31, l32, l33,
          l41, l42, l43, l44,
        ]),
      ),
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
      } => (
        Family::TwoAssetHestonReflected.code(),
        pad([
          mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2, l11, l21, l22, l31, l32, l33,
          l41, l42, l43, l44,
        ]),
      ),
      EulerSpec::MertonJumpLog { drift_ln, sigma } => {
        (Family::MertonJumpLog.code(), pad([drift_ln, sigma]))
      }
      EulerSpec::BatesJump {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      } => (
        Family::BatesJump.code(),
        pad([drift_c, alpha, beta, sigma, rho]),
      ),
      EulerSpec::BatesJumpReflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      } => (
        Family::BatesJumpReflected.code(),
        pad([drift_c, alpha, beta, sigma, rho]),
      ),
      EulerSpec::AndersenQe {
        theta,
        e_kd,
        c1,
        c2,
        k0,
        k1,
        k2,
        k34,
        mu,
      } => (
        Family::AndersenQe.code(),
        pad([theta, e_kd, c1, c2, k0, k1, k2, k34, mu]),
      ),
      EulerSpec::CountingProcess => (Family::CountingProcess.code(), pad([])),
      EulerSpec::InverseGaussianSubordinator {
        mu_ig,
        two_lam,
        four_mu_lam,
      } => (
        Family::InverseGaussianSubordinator.code(),
        pad([mu_ig, two_lam, four_mu_lam]),
      ),
      EulerSpec::NormalInverseGaussian {
        theta,
        sigma,
        mu_ig,
        two_lam,
        four_mu_lam,
      } => (
        Family::NormalInverseGaussian.code(),
        pad([theta, sigma, mu_ig, two_lam, four_mu_lam]),
      ),
      EulerSpec::StableSubordinator {
        alpha,
        inv_alpha,
        one_minus_alpha,
        tail_exp,
        log_scale,
        pi,
      } => (
        Family::StableSubordinator.code(),
        pad([alpha, inv_alpha, one_minus_alpha, tail_exp, log_scale, pi]),
      ),
      EulerSpec::KouJumpHeston {
        drift_c,
        kappa,
        theta,
        sigma_v,
        rho,
      } => (
        Family::KouJumpHeston.code(),
        pad([drift_c, kappa, theta, sigma_v, rho]),
      ),
      EulerSpec::KouJumpHestonReflected {
        drift_c,
        kappa,
        theta,
        sigma_v,
        rho,
      } => (
        Family::KouJumpHestonReflected.code(),
        pad([drift_c, kappa, theta, sigma_v, rho]),
      ),
      EulerSpec::DuffieKanJump {
        a1,
        b1,
        c1,
        sigma1,
        a2,
        b2,
        c2,
        sigma2,
        alpha,
        beta,
        gamma,
        rho,
      } => (
        Family::DuffieKanJump.code(),
        pad([
          a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho,
        ]),
      ),
      EulerSpec::HawkesJumpDiffusion {
        drift_c,
        sigma,
        alpha,
        beta,
        mu_lambda,
        jump_mu,
        jump_sigma,
      } => (
        Family::HawkesJumpDiffusion.code(),
        pad([drift_c, sigma, alpha, beta, mu_lambda, jump_mu, jump_sigma]),
      ),
      EulerSpec::Garch { omega, alpha, beta } => (Family::Garch.code(), pad([omega, alpha, beta])),
      EulerSpec::ThresholdGarch {
        omega,
        alpha,
        gamma,
        beta,
      } => (
        Family::ThresholdGarch.code(),
        pad([omega, alpha, gamma, beta]),
      ),
      EulerSpec::ExponentialGarch {
        omega,
        alpha,
        gamma,
        beta,
        e_abs_z,
      } => (
        Family::ExponentialGarch.code(),
        pad([omega, alpha, gamma, beta, e_abs_z]),
      ),
      EulerSpec::Innovation { mean, sd } => (Family::Innovation.code(), pad([mean, sd])),
      EulerSpec::CorrelatedInnovation { rho } => (Family::CorrelatedInnovation.code(), pad([rho])),
      EulerSpec::Autoregressive { phi, sigma } => {
        (Family::Autoregressive.code(), pad([phi, sigma]))
      }
      EulerSpec::MovingAverage { theta, sigma } => {
        (Family::MovingAverage.code(), pad([theta, sigma]))
      }
      EulerSpec::GammaSubordinator => (Family::GammaSubordinator.code(), pad([])),
      EulerSpec::VarianceGamma { mu, sigma } => (Family::VarianceGamma.code(), pad([mu, sigma])),
      EulerSpec::BilateralGamma => (Family::BilateralGamma.code(), pad([])),
      EulerSpec::BilateralGammaMotion { sigma } => {
        (Family::BilateralGammaMotion.code(), pad([sigma]))
      }
      EulerSpec::TemperedStableSubordinator { drift } => {
        (Family::TemperedStableSubordinator.code(), pad([drift]))
      }
      EulerSpec::BarndorffNielsenShephard { decay, mu } => {
        (Family::BarndorffNielsenShephard.code(), pad([decay, mu]))
      }
      EulerSpec::CorrelatedFractionalMotion { rho } => {
        (Family::CorrelatedFractionalMotion.code(), pad([rho]))
      }
      EulerSpec::ComplexFractionalOu {
        lambda,
        omega,
        scale,
      } => (
        Family::ComplexFractionalOu.code(),
        pad([lambda, omega, scale]),
      ),
      EulerSpec::TransformedOrnsteinUhlenbeck {
        kappa,
        mu,
        sigma,
        arctan,
        half_pi,
      } => (
        Family::TransformedOrnsteinUhlenbeck.code(),
        pad([kappa, mu, sigma, arctan, half_pi]),
      ),
      EulerSpec::PoissonArrivals { lambda } => (Family::PoissonArrivals.code(), pad([lambda])),
      EulerSpec::DynamicSabr => (Family::DynamicSabr.code(), pad([])),
      EulerSpec::HeathJarrowMorton => (Family::HeathJarrowMorton.code(), pad([])),
      EulerSpec::AffineDiffusionGaussian { sigma } => {
        (Family::AffineDiffusionGaussian.code(), pad([sigma]))
      }
      EulerSpec::WuZhang {
        alpha,
        beta,
        nu,
        lambda,
      } => (Family::WuZhang.code(), pad([alpha, beta, nu, lambda])),
      EulerSpec::CorrelatedGeometric4 { mu, sigma, l } => {
        let mut values = [T::zero(); 18];
        values[..4].copy_from_slice(&mu);
        values[4..8].copy_from_slice(&sigma);
        values[8..].copy_from_slice(&l);
        (Family::CorrelatedGeometric4.code(), pad(values))
      }
      EulerSpec::CorrelatedNoises4 { l } => (Family::CorrelatedNoises4.code(), pad(l)),
      EulerSpec::RegimeSwitching {
        mu,
        sigma,
        thresholds,
      } => {
        let mut values = [T::zero(); 17];
        values[0] = mu;
        values[1..5].copy_from_slice(&sigma);
        for (row, cuts) in thresholds.iter().enumerate() {
          values[5 + 3 * row..8 + 3 * row].copy_from_slice(cuts);
        }
        (Family::RegimeSwitching.code(), pad(values))
      }
      EulerSpec::RiemannLiouville => (Family::RiemannLiouville.code(), pad([])),
      EulerSpec::RiemannLiouvilleHeston {
        mu,
        kappa,
        theta,
        nu,
        rho,
      } => (
        Family::RiemannLiouvilleHeston.code(),
        pad([mu, kappa, theta, nu, rho]),
      ),
      EulerSpec::RiemannLiouvilleOu { kappa, mu, nu } => {
        (Family::RiemannLiouvilleOu.code(), pad([kappa, mu, nu]))
      }
      EulerSpec::RiemannLiouvilleBlackScholes { s0, sigma } => (
        Family::RiemannLiouvilleBlackScholes.code(),
        pad([s0, sigma]),
      ),
      EulerSpec::AdditiveJumpDiffusion { drift_dt, sigma } => {
        (Family::AdditiveJumpDiffusion.code(), pad([drift_dt, sigma]))
      }
      EulerSpec::Bates1996 {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      } => (
        Family::Bates1996.code(),
        pad([drift_c, alpha, beta, sigma, rho]),
      ),
      EulerSpec::Bates1996Reflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      } => (
        Family::Bates1996Reflected.code(),
        pad([drift_c, alpha, beta, sigma, rho]),
      ),
      EulerSpec::CompoundPoissonEvents { lambda } => {
        (Family::CompoundPoissonEvents.code(), pad([lambda]))
      }
      EulerSpec::JumpFractionalOu { theta, mu, sigma } => {
        (Family::JumpFractionalOu.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::VolterraSquareRoot { kappa, theta, nu } => {
        (Family::VolterraSquareRoot.code(), pad([kappa, theta, nu]))
      }
      EulerSpec::GaussianPolynomialVolatility { coefficients } => (
        Family::GaussianPolynomialVolatility.code(),
        pad(coefficients),
      ),
      EulerSpec::RoughHestonMemory {
        mu,
        theta,
        ek,
        nu,
        c1,
        c2,
        inv_g,
        rho,
      } => (
        Family::RoughHestonMemory.code(),
        pad([mu, theta, ek, nu, c1, c2, inv_g, rho]),
      ),
      EulerSpec::FractionalBatesMemory {
        mu_c,
        theta,
        ek,
        xi,
        inv_g,
        rho,
      } => (
        Family::FractionalBatesMemory.code(),
        pad([mu_c, theta, ek, xi, inv_g, rho]),
      ),
      EulerSpec::RoughBergomiMemory {
        r,
        a,
        v0sq,
        residual_sd,
        rho,
      } => (
        Family::RoughBergomiMemory.code(),
        pad([r, a, v0sq, residual_sd, rho]),
      ),
      EulerSpec::LinearFractionalStable {
        alpha,
        inv_alpha,
        tail_exp,
        b,
        scale_s,
        pi,
        half_pi,
      } => (
        Family::LinearFractionalStable.code(),
        pad([alpha, inv_alpha, tail_exp, b, scale_s, pi, half_pi]),
      ),
      EulerSpec::MovingAverageFilter { sigma } => {
        (Family::MovingAverageFilter.code(), pad([sigma]))
      }
      EulerSpec::TemperedStableSeries {
        b_t,
        rate,
        inv_alpha,
        e_scale,
        e_pow,
        w_plus,
        lambda_plus,
        lambda_minus,
      } => (
        Family::TemperedStableSeries.code(),
        pad([
          b_t,
          rate,
          inv_alpha,
          e_scale,
          e_pow,
          w_plus,
          lambda_plus,
          lambda_minus,
        ]),
      ),
      EulerSpec::VolterraReference => (Family::VolterraReference.code(), pad([])),
      EulerSpec::HawkesEvents { mu, alpha, beta } => {
        (Family::HawkesEvents.code(), pad([mu, alpha, beta]))
      }
      EulerSpec::HawkesEvents2 { mu, alpha, beta } => (
        Family::HawkesEvents2.code(),
        pad([
          mu[0], mu[1], alpha[0], alpha[1], alpha[2], alpha[3], beta[0], beta[1],
        ]),
      ),
      EulerSpec::CheyetteLocalVol { kappa } => (Family::CheyetteLocalVol.code(), pad([kappa])),
      EulerSpec::VolterraProgram => (Family::VolterraProgram.code(), pad([])),
      EulerSpec::HestonSlv {
        mu,
        kappa,
        theta,
        sigma,
        rho,
      } => (
        Family::HestonSlv.code(),
        pad([mu, kappa, theta, sigma, rho]),
      ),
      EulerSpec::InverseStableSubordinator {
        alpha,
        c,
        inv_alpha,
        one_minus_alpha,
        tail_exp,
        pi,
      } => (
        Family::InverseStableSubordinator.code(),
        pad([alpha, c, inv_alpha, one_minus_alpha, tail_exp, pi]),
      ),
      EulerSpec::StochasticVolatilityCgmy {
        rate0,
        inv_alpha,
        lambda_plus,
        lambda_minus,
        bcoef,
        twoc,
        ek,
        rho,
      } => (
        Family::StochasticVolatilityCgmy.code(),
        pad([
          rate0,
          inv_alpha,
          lambda_plus,
          lambda_minus,
          bcoef,
          twoc,
          ek,
          rho,
        ]),
      ),
      EulerSpec::WishartTwo {
        m,
        theta,
        theta_inv,
        two,
      } => {
        let mut values = [T::zero(); 13];
        values[..4].copy_from_slice(&m);
        values[4..8].copy_from_slice(&theta);
        values[8..12].copy_from_slice(&theta_inv);
        values[12] = two;
        (Family::WishartTwo.code(), pad(values))
      }
      EulerSpec::LiborMarket4 { sigma, delta, l } => {
        let mut values = [T::zero(); 18];
        values[..4].copy_from_slice(&sigma);
        values[4..8].copy_from_slice(&delta);
        values[8..].copy_from_slice(&l);
        (Family::LiborMarket4.code(), pad(values))
      }
    }
  }
}
