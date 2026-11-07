//! The moments and the entropy: sums of the kernel's powers about the centre, in units of the clipped width.

use std::ops::ControlFlow;
use std::sync::OnceLock;

use super::Kernel;
use super::Truncation;
use super::panels::Cut;
use super::panels::Panel;
use super::panels::panels;
use super::panels::span;
use crate::special::ldexp;

impl<K: Kernel> Truncation<K> {
  /// `(mean, (scale·2^p)²·variance, skewness, excess kurtosis)`, `2^p` the law's unit, in the square before it rounds.
  pub(in crate::truncated) fn moments(
    &self,
    scale: f64,
    p: i32,
    summed: &OnceLock<[f64; 4]>,
  ) -> [f64; 4] {
    let (frame, [unit, offset], [variance, skewness, kurtosis]) =
      self.located_moments(scale, p, summed);
    [frame.plus(unit * offset), variance, skewness, kurtosis]
  }

  /// The mean as its frame and `unit·offset` (in widths, where an offset would underflow against the width), then the
  /// other three; an interval cutting neither end reads the base's closed forms (2/√k at k = 1e300).
  pub(super) fn located_moments(
    &self,
    scale: f64,
    p: i32,
    summed: &OnceLock<[f64; 4]>,
  ) -> (Cut, [f64; 2], [f64; 3]) {
    let (lo, up) = self.kernel.support();
    // By exact offsets: a clip within an ulp of its bound still cuts nothing.
    if (self.lo.x <= lo || span(self.lo, self.a) > 0.0)
      && (self.up.x >= up || span(self.up, self.b) < 0.0)
    {
      let [mean, sd, skewness, kurtosis] = self.kernel.base_moments();
      let unit = ldexp(scale * sd, p);
      return (Cut::at(mean), [1.0, 0.0], [unit * unit, skewness, kurtosis]);
    }
    if self.narrow {
      return (self.centre, [1.0, 0.0], [0.0, f64::NAN, f64::NAN]);
    }
    self.summed_moments(scale, p, summed)
  }

  /// Whether the reference point is the clipped interval's midpoint, where a flat kernel's moments are summed.
  pub(in crate::truncated) fn about_midpoint(&self) -> bool {
    self.centre == self.a.shifted(0.5 * span(self.a, self.b))
  }

  /// The moments from sums in units of `width`: the mean's offset `d`, then sums about it, so no binomial shift by a
  /// far reference point cancels; `summed` keeps what the scale leaves untouched from the first call.
  pub(super) fn summed_moments(
    &self,
    scale: f64,
    p: i32,
    summed: &OnceLock<[f64; 4]>,
  ) -> (Cut, [f64; 2], [f64; 3]) {
    let [offset, variance, skewness, kurtosis] = *summed.get_or_init(|| {
      let reach = self.reach();
      let mean = self.moment_sums::<2>(reach, 0.0);
      let d = mean[1] / mean[0];
      let m = self.moment_sums::<5>(reach, d);
      let e = m[1] / m[0];
      // The central sums before they are divided by `m[0]`: beside a near-point mass `μ₂` is subnormal, its sums are not.
      let c2 = m[2] - e * m[1];
      let c3 = m[3] - 3.0 * e * m[2] + 2.0 * e * e * m[1];
      let c4 = m[4] - 4.0 * e * m[3] + 6.0 * e * e * m[2] - 3.0 * e * e * e * m[1];
      let inv_sd = m[0].sqrt() / c2.sqrt();
      [
        self.flat_mean().unwrap_or(d + e),
        c2 / m[0],
        (c3 / c2) * inv_sd,
        (c4 / c2) * inv_sd * inv_sd - 3.0,
      ]
    });
    let unit = ldexp(scale * self.width, p);
    (
      self.frame,
      [self.width, offset],
      [(unit * variance) * unit, skewness, kurtosis],
    )
  }

  fn moment_sums<const N: usize>(&self, reach: (Cut, Cut), about: f64) -> [f64; N] {
    let mut m = [0.0; N];
    let _ = self.walk_about_centre(reach, |panel| {
      if let Some(e) = panel
        .singular
        .filter(|_| !Self::powers_on_mass_nodes(&panel, N))
      {
        self.singular_moments(&panel, e, about, &mut m);
      } else {
        self.nodes(&panel, |w, ln_v, offset, _| {
          let (g, z) = (w * (ln_v - self.scale).exp(), offset / self.width - about);
          let mut zj = 1.0;
          for slot in &mut m {
            *slot += g * zj;
            zj *= z;
          }
        });
      }
      ControlFlow::<()>::Continue(())
    });
    m
  }

  /// `−∫ f ln f` as `ln M₀ − E[ln k(X) − ln k(c)]`, so no large `ln k(c)` cancels against `E[ln k(X)]`; `kept` holds it
  /// from the first call.
  pub(super) fn entropy(&self, kept: &OnceLock<f64>) -> f64 {
    if self.narrow {
      return f64::NAN;
    }
    *kept.get_or_init(|| {
      // Relative to `m0`: past 1e154 a mass and its log's own size overflow in one product (`β = 4.9e-165` at 1).
      let shift = self.m0.ln() + self.scale;
      let (mut mass, mut log) = (0.0, 0.0);
      // Each singular panel's `Σ g·ln u` and exponent, scaled by `(e − 1)/e` once the mass is known.
      let mut apart = [(0.0, 1.0); 2];
      let _ = self.walk_about_centre(self.reach(), |panel| {
        if let Some(e) = panel.singular {
          let (m, l, u) = self.singular_entropy(&panel, e, shift);
          (mass, log) = (mass + m, log + l);
          if let Some(slot) = apart.iter_mut().find(|(v, _)| *v == 0.0) {
            *slot = (u, e);
          }
        } else {
          self.nodes(&panel, |w, ln_v, _, ln_g| {
            let g = w * (ln_v - shift).exp();
            if g > 0.0 {
              mass += g;
              log += g * ln_g;
            }
          });
        }
        ControlFlow::<()>::Continue(())
      });
      let apart: f64 = apart.iter().map(|&(u, e)| (u / mass) * (e - 1.0) / e).sum();
      (self.width.ln() + shift + mass.ln()) - log / mass - apart
    })
  }

  /// The moments' and entropy's walk: about the centre, where a near-symmetric law's odd powers keep their digits.
  fn walk_about_centre<B>(
    &self,
    reach: (Cut, Cut),
    visit: impl FnMut(Panel) -> ControlFlow<B>,
  ) -> ControlFlow<B> {
    panels(&self.kernel, self.centre, reach, false, false, visit)
  }
}
