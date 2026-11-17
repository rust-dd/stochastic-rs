use super::*;
use crate::malliavin::Malliavin;

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Heston<T, Euler, S, B> {
  /// The path and `D_{t_i} v_T`, by Alòs and Ewald (2008, Cor. 4.2) under `2κθ ≥ σ²` for `Sqrt`,
  /// by Nualart and Itô on `ln σ(v)` for `ThreeHalves`; a zero in `v` zeroes the entries up to it.
  /// $$
  /// D_{t_i} v_T = \sigma v_T^{p} \exp\Bigl(\sum_{k=i}^{n-2} r(v_{t_k}) \Delta t\Bigr), \quad r(v) = \begin{cases} -\frac{\kappa}{2} - \frac{\kappa\theta/2 - \sigma^2/8}{v} & p = \frac12 \\ \frac{\kappa}{2} - \frac{3\kappa\theta}{2v} - \frac{3\sigma^2 v}{8} & p = \frac32 \end{cases}
  /// $$
  pub fn malliavin(&self) -> Malliavin<T, [Array1<T>; 2]> {
    let [s, v] = self.sample();
    let dt = self.cgns.dt();
    let half = T::from_f64_fast(0.5);
    let (kappa_theta, sigma2) = (self.kappa * self.theta, self.sigma * self.sigma);
    let v_t = v[self.n - 1];
    let (a, b, c, scale) = match self.pow {
      HestonPow::Sqrt => (
        -half * self.kappa,
        half * kappa_theta - T::from_f64_fast(0.125) * sigma2,
        T::zero(),
        self.sigma * v_t.sqrt(),
      ),
      HestonPow::ThreeHalves => (
        half * self.kappa,
        T::from_f64_fast(1.5) * kappa_theta,
        T::from_f64_fast(0.375) * sigma2,
        self.sigma * v_t * v_t.sqrt(),
      ),
    };
    let mut derivative = Array1::<T>::zeros(self.n);
    let mut tail = T::zero();
    for i in (0..self.n).rev() {
      if i + 1 < self.n {
        tail += (a - b / v[i] - c * v[i]) * dt;
      }
      derivative[i] = scale * tail.exp();
    }
    Malliavin {
      paths: [s, v],
      derivative,
    }
  }
}
