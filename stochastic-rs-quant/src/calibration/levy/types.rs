use crate::calibration::Calibrated;
use crate::traits::ToModel;

/// Supported Lévy model types for calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LevyModelType {
  /// Variance Gamma: $\psi(\xi)=-\frac{1}{\nu}\ln\!\bigl(1-i\theta\nu\xi+\tfrac12\sigma^2\nu\xi^2\bigr)$
  VarianceGamma,
  /// Normal Inverse Gaussian: $\psi(\xi)=\delta\bigl(\sqrt{\alpha^2-\beta^2}-\sqrt{\alpha^2-(\beta+i\xi)^2}\bigr)$
  Nig,
  /// Cgmy: $\psi(\xi)=C\,\Gamma(-Y)\bigl[(M-i\xi)^Y-M^Y+(G+i\xi)^Y-G^Y\bigr]$
  Cgmy,
  /// Merton Jump-Diffusion: $\psi(\xi)=-\tfrac12\sigma^2\xi^2+\lambda\bigl(e^{i\mu_J\xi-\frac12\sigma_J^2\xi^2}-1\bigr)$
  MertonJd,
  /// Kou Double-Exponential: $\psi(\xi)=-\tfrac12\sigma^2\xi^2+\lambda\bigl(\frac{p\eta_1}{\eta_1-i\xi}+\frac{(1-p)\eta_2}{\eta_2+i\xi}-1\bigr)$
  Kou,
}

/// Calibrated parameter set for a Lévy model — the parameter vector together
/// with its [`LevyModelType`] tag (the vector layout depends on the model).
#[derive(Clone, Debug, PartialEq)]
pub struct LevyParams {
  /// Calibrated parameter vector. Layout is model-specific.
  pub values: Vec<f64>,
  /// Lévy model variant the vector belongs to.
  pub model_type: LevyModelType,
}

/// The five Lévy Fourier models behind one enum, so a Lévy fit's
/// [`ToModel::to_model`](crate::traits::ToModel::to_model) stays a concrete `ModelPricer`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LevyModel {
  VarianceGamma(crate::pricing::fourier::VarianceGammaFourier),
  Nig(crate::pricing::fourier::NigFourier),
  Cgmy(crate::pricing::fourier::CgmyFourier),
  MertonJd(crate::pricing::fourier::MertonJdFourier),
  Kou(crate::pricing::fourier::KouFourier),
}

impl crate::traits::ModelPricer for LevyModel {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    match self {
      LevyModel::VarianceGamma(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Nig(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Cgmy(m) => m.price_call(s, k, r, q, tau),
      LevyModel::MertonJd(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Kou(m) => m.price_call(s, k, r, q, tau),
    }
  }

  /// The member's own put, not parity on the enum's call: the payoff at expiry where its is.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    match self {
      LevyModel::VarianceGamma(m) => m.price_put(s, k, r, q, tau),
      LevyModel::Nig(m) => m.price_put(s, k, r, q, tau),
      LevyModel::Cgmy(m) => m.price_put(s, k, r, q, tau),
      LevyModel::MertonJd(m) => m.price_put(s, k, r, q, tau),
      LevyModel::Kou(m) => m.price_put(s, k, r, q, tau),
    }
  }
}

/// Each variant forwards to a [`FourierModelExt`](crate::pricing::fourier::FourierModelExt)
/// model, so the enum is a European vanilla call pricer exactly as its
/// members are.
impl crate::traits::VanillaEuropeanCall for LevyModel {}

impl LevyModel {
  /// The Fourier model of `p`, laid out as `model_type` reads it.
  pub(super) fn of(model_type: LevyModelType, p: &[f64]) -> Self {
    use crate::pricing::fourier::*;
    match model_type {
      LevyModelType::VarianceGamma => LevyModel::VarianceGamma(VarianceGammaFourier {
        sigma: p[0],
        theta: p[1],
        nu: p[2],
      }),
      LevyModelType::Nig => LevyModel::Nig(NigFourier {
        alpha: p[0],
        beta: p[1],
        delta: p[2],
      }),
      LevyModelType::Cgmy => LevyModel::Cgmy(CgmyFourier {
        c: p[0],
        g: p[1],
        m: p[2],
        y: p[3],
      }),
      LevyModelType::MertonJd => LevyModel::MertonJd(MertonJdFourier {
        sigma: p[0],
        lambda: p[1],
        mu_j: p[2],
        sigma_j: p[3],
      }),
      LevyModelType::Kou => LevyModel::Kou(KouFourier {
        sigma: p[0],
        lambda: p[1],
        p_up: p[2],
        eta1: p[3],
        eta2: p[4],
      }),
    }
  }
}

impl ToModel for Calibrated<LevyParams> {
  type Model = LevyModel;

  fn to_model(&self) -> LevyModel {
    LevyModel::of(self.params.model_type, &self.params.values)
  }
}

#[cfg(test)]
mod tests {
  use super::super::loss::default_params;
  use super::*;
  use crate::traits::ModelPricer;

  /// At expiry it is the payoff, where parity on a call rounds `(1.1, 0.3)`'s 0 to `-5.6e-17`.
  #[test]
  fn a_levy_models_put_is_its_members_to_the_bit() {
    let types = [
      LevyModelType::VarianceGamma,
      LevyModelType::Nig,
      LevyModelType::Cgmy,
      LevyModelType::MertonJd,
      LevyModelType::Kou,
    ];
    for model in types.map(|model| LevyModel::of(model, &default_params(model))) {
      for (s, k, tau) in [(1.1, 0.3, 0.0), (100.0, 110.0, 0.0), (100.0, 110.0, 1.0)] {
        let member = match &model {
          LevyModel::VarianceGamma(m) => m.price_put(s, k, 0.05, 0.01, tau),
          LevyModel::Nig(m) => m.price_put(s, k, 0.05, 0.01, tau),
          LevyModel::Cgmy(m) => m.price_put(s, k, 0.05, 0.01, tau),
          LevyModel::MertonJd(m) => m.price_put(s, k, 0.05, 0.01, tau),
          LevyModel::Kou(m) => m.price_put(s, k, 0.05, 0.01, tau),
        };
        let own = model.price_put(s, k, 0.05, 0.01, tau);
        assert_eq!(
          own.to_bits(),
          member.to_bits(),
          "{model:?} at ({s}, {k}, {tau})"
        );
        if tau == 0.0 {
          assert_eq!(own, f64::max(k - s, 0.0), "{model:?} at ({s}, {k})");
        }
      }
    }
  }
}
