use super::LogRealizedVarianceParameterBoundaryFlags;
use super::LogRealizedVarianceParameterUncertainty;
use super::LogRealizedVarianceParameters;
use super::LogRealizedVarianceQmlBounds;
use super::filter::filter_centered_log_observations;

const BOUNDARY_FRACTION: f64 = 1e-4;
const CONDITION_NUMBER_CAP: f64 = 1e8;
const DIFFERENCE_RELATIVE_STEP: f64 = 1e-4;
const DIFFERENCE_SPAN_STEP: f64 = 1e-6;

pub(super) fn estimate_parameter_uncertainty(
  observations: &[f64],
  measurement_variance: f64,
  parameters: LogRealizedVarianceParameters,
  bounds: LogRealizedVarianceQmlBounds,
) -> LogRealizedVarianceParameterUncertainty {
  let boundary = parameter_boundary_flags(parameters, bounds);
  let Some((information, score_outer_product)) =
    numerical_derivatives(observations, measurement_variance, parameters, bounds)
  else {
    return unavailable(boundary, None, None);
  };
  let Some((scaled_inverse, condition_number, scales)) = invert_scaled_information(information)
  else {
    return unavailable(boundary, Some(information), Some(score_outer_product));
  };
  let mut observed_information_covariance = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      observed_information_covariance[row][column] =
        scaled_inverse[row][column] / (scales[row] * scales[column]);
    }
  }
  if !observed_information_covariance
    .iter()
    .flatten()
    .all(|value| value.is_finite())
    || !(observed_information_covariance[0][0] > 0.0
      && observed_information_covariance[1][1] > 0.0
      && observed_information_covariance[2][2] > 0.0)
  {
    return unavailable(boundary, Some(information), Some(score_outer_product));
  }
  let robust_sandwich_covariance =
    symmetric_product(observed_information_covariance, score_outer_product);
  let robust_covariance_usable = positive_definite_scaled(robust_sandwich_covariance);
  let robust_standard_errors = robust_covariance_usable.then(|| {
    [
      robust_sandwich_covariance[0][0].sqrt(),
      robust_sandwich_covariance[1][1].sqrt(),
      robust_sandwich_covariance[2][2].sqrt(),
    ]
  });
  let ill_conditioned = !condition_number.is_finite() || condition_number > CONDITION_NUMBER_CAP;
  LogRealizedVarianceParameterUncertainty {
    observed_information: Some(information),
    score_outer_product: Some(score_outer_product),
    observed_information_covariance: Some(observed_information_covariance),
    robust_sandwich_covariance: robust_covariance_usable.then_some(robust_sandwich_covariance),
    robust_standard_errors,
    scaled_condition_number: condition_number,
    singular: false,
    ill_conditioned,
    robust_covariance_usable,
    boundary,
  }
}

pub(super) fn parameter_boundary_flags(
  parameters: LogRealizedVarianceParameters,
  bounds: LogRealizedVarianceQmlBounds,
) -> LogRealizedVarianceParameterBoundaryFlags {
  let mu_position = normalized_position(parameters.mu, bounds.min_mu, bounds.max_mu);
  let phi_position = normalized_position(parameters.phi, bounds.min_phi, bounds.max_phi);
  let q_position = normalized_position(parameters.q.ln(), bounds.min_q.ln(), bounds.max_q.ln());
  LogRealizedVarianceParameterBoundaryFlags {
    mu_at_lower_bound: mu_position <= BOUNDARY_FRACTION,
    mu_at_upper_bound: mu_position >= 1.0 - BOUNDARY_FRACTION,
    phi_at_lower_bound: phi_position <= BOUNDARY_FRACTION,
    phi_at_upper_bound: phi_position >= 1.0 - BOUNDARY_FRACTION,
    q_at_lower_bound: q_position <= BOUNDARY_FRACTION,
    q_at_upper_bound: q_position >= 1.0 - BOUNDARY_FRACTION,
  }
}

fn numerical_derivatives(
  observations: &[f64],
  measurement_variance: f64,
  parameters: LogRealizedVarianceParameters,
  bounds: LogRealizedVarianceQmlBounds,
) -> Option<([[f64; 3]; 3], [[f64; 3]; 3])> {
  let point = [parameters.mu, parameters.phi, parameters.q];
  let lower = [bounds.min_mu, bounds.min_phi, bounds.min_q];
  let upper = [bounds.max_mu, bounds.max_phi, bounds.max_q];
  let mut steps = [0.0; 3];
  for index in 0..3 {
    steps[index] = difference_step(point[index], lower[index], upper[index])?;
  }
  let objective = |values: [f64; 3]| {
    let candidate = LogRealizedVarianceParameters {
      mu: values[0],
      phi: values[1],
      q: values[2],
    };
    filter_centered_log_observations(observations, measurement_variance, candidate)
      .ok()
      .map(|filtered| -filtered.log_likelihood)
      .filter(|value| value.is_finite())
  };
  let center = objective(point)?;
  let mut information = [[0.0; 3]; 3];
  for index in 0..3 {
    let mut above = point;
    let mut below = point;
    above[index] += steps[index];
    below[index] -= steps[index];
    information[index][index] =
      (objective(above)? - 2.0 * center + objective(below)?) / steps[index].powi(2);
  }
  for row in 0..3 {
    for column in row + 1..3 {
      let mut above_above = point;
      let mut above_below = point;
      let mut below_above = point;
      let mut below_below = point;
      above_above[row] += steps[row];
      above_above[column] += steps[column];
      above_below[row] += steps[row];
      above_below[column] -= steps[column];
      below_above[row] -= steps[row];
      below_above[column] += steps[column];
      below_below[row] -= steps[row];
      below_below[column] -= steps[column];
      let value = (objective(above_above)? - objective(above_below)? - objective(below_above)?
        + objective(below_below)?)
        / (4.0 * steps[row] * steps[column]);
      information[row][column] = value;
      information[column][row] = value;
    }
  }
  if !information.iter().flatten().all(|value| value.is_finite()) {
    return None;
  }
  let mut scores = vec![[0.0; 3]; observations.len()];
  for parameter_index in 0..3 {
    let mut above = point;
    let mut below = point;
    above[parameter_index] += steps[parameter_index];
    below[parameter_index] -= steps[parameter_index];
    let above_contributions = likelihood_contributions(observations, measurement_variance, above)?;
    let below_contributions = likelihood_contributions(observations, measurement_variance, below)?;
    for observation_index in 0..observations.len() {
      scores[observation_index][parameter_index] = (above_contributions[observation_index]
        - below_contributions[observation_index])
        / (2.0 * steps[parameter_index]);
    }
  }
  let mut score_outer_product = [[0.0; 3]; 3];
  for score in scores {
    if !score.iter().all(|value| value.is_finite()) {
      return None;
    }
    for row in 0..3 {
      for column in 0..3 {
        score_outer_product[row][column] += score[row] * score[column];
      }
    }
  }
  score_outer_product
    .iter()
    .flatten()
    .all(|value| value.is_finite())
    .then_some((information, score_outer_product))
}

fn likelihood_contributions(
  observations: &[f64],
  measurement_variance: f64,
  values: [f64; 3],
) -> Option<Vec<f64>> {
  let parameters = LogRealizedVarianceParameters {
    mu: values[0],
    phi: values[1],
    q: values[2],
  };
  filter_centered_log_observations(observations, measurement_variance, parameters)
    .ok()
    .map(|filtered| filtered.log_likelihood_contribution_path)
    .filter(|contributions| {
      contributions.len() == observations.len()
        && contributions.iter().all(|value| value.is_finite())
    })
}

fn difference_step(value: f64, lower: f64, upper: f64) -> Option<f64> {
  let distance = (value - lower).min(upper - value);
  if !(distance.is_finite() && distance > 0.0) {
    return None;
  }
  let requested =
    (value.abs() * DIFFERENCE_RELATIVE_STEP).max((upper - lower) * DIFFERENCE_SPAN_STEP);
  let step = requested.min(0.25 * distance);
  (step.is_finite() && step > 100.0 * f64::EPSILON * value.abs().max(1.0)).then_some(step)
}

fn invert_scaled_information(information: [[f64; 3]; 3]) -> Option<([[f64; 3]; 3], f64, [f64; 3])> {
  if !(information[0][0] > 0.0 && information[1][1] > 0.0 && information[2][2] > 0.0) {
    return None;
  }
  let scales = [
    information[0][0].sqrt(),
    information[1][1].sqrt(),
    information[2][2].sqrt(),
  ];
  let mut scaled = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      scaled[row][column] = information[row][column] / (scales[row] * scales[column]);
    }
  }
  positive_definite(scaled)?;
  let inverse = inverse_symmetric(scaled)?;
  let condition_number = infinity_norm(scaled) * infinity_norm(inverse);
  Some((inverse, condition_number, scales))
}

fn positive_definite(matrix: [[f64; 3]; 3]) -> Option<()> {
  let first = matrix[0][0];
  if !(first.is_finite() && first > 1e-12) {
    return None;
  }
  let l00 = first.sqrt();
  let l10 = matrix[1][0] / l00;
  let l20 = matrix[2][0] / l00;
  let second = matrix[1][1] - l10 * l10;
  if !(second.is_finite() && second > 1e-12) {
    return None;
  }
  let l11 = second.sqrt();
  let l21 = (matrix[2][1] - l20 * l10) / l11;
  let third = matrix[2][2] - l20 * l20 - l21 * l21;
  (third.is_finite() && third > 1e-12).then_some(())
}

fn inverse_symmetric(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
  let [a, b, c] = matrix[0];
  let d = matrix[1][1];
  let e = matrix[1][2];
  let f = matrix[2][2];
  let cofactors = [
    [d * f - e * e, c * e - b * f, b * e - c * d],
    [c * e - b * f, a * f - c * c, b * c - a * e],
    [b * e - c * d, b * c - a * e, a * d - b * b],
  ];
  let determinant = a * cofactors[0][0] + b * cofactors[0][1] + c * cofactors[0][2];
  if !(determinant.is_finite() && determinant > 1e-14) {
    return None;
  }
  let mut inverse = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      inverse[row][column] = cofactors[row][column] / determinant;
    }
  }
  inverse
    .iter()
    .flatten()
    .all(|value| value.is_finite())
    .then_some(inverse)
}

fn symmetric_product(
  inverse_information: [[f64; 3]; 3],
  score_outer_product: [[f64; 3]; 3],
) -> [[f64; 3]; 3] {
  let left = matrix_product(inverse_information, score_outer_product);
  let raw = matrix_product(left, inverse_information);
  let mut symmetric = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      symmetric[row][column] = 0.5 * (raw[row][column] + raw[column][row]);
    }
  }
  symmetric
}

fn matrix_product(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
  let mut product = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      for inner in 0..3 {
        product[row][column] += left[row][inner] * right[inner][column];
      }
    }
  }
  product
}

fn positive_definite_scaled(matrix: [[f64; 3]; 3]) -> bool {
  if !matrix.iter().flatten().all(|value| value.is_finite())
    || !(matrix[0][0] > 0.0 && matrix[1][1] > 0.0 && matrix[2][2] > 0.0)
  {
    return false;
  }
  let scales = [
    matrix[0][0].sqrt(),
    matrix[1][1].sqrt(),
    matrix[2][2].sqrt(),
  ];
  let mut scaled = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      scaled[row][column] = matrix[row][column] / (scales[row] * scales[column]);
    }
  }
  positive_definite(scaled).is_some()
}

fn infinity_norm(matrix: [[f64; 3]; 3]) -> f64 {
  matrix
    .iter()
    .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
    .fold(0.0, f64::max)
}

fn normalized_position(value: f64, lower: f64, upper: f64) -> f64 {
  (value - lower) / (upper - lower)
}

fn unavailable(
  boundary: LogRealizedVarianceParameterBoundaryFlags,
  observed_information: Option<[[f64; 3]; 3]>,
  score_outer_product: Option<[[f64; 3]; 3]>,
) -> LogRealizedVarianceParameterUncertainty {
  LogRealizedVarianceParameterUncertainty {
    observed_information,
    score_outer_product,
    observed_information_covariance: None,
    robust_sandwich_covariance: None,
    robust_standard_errors: None,
    scaled_condition_number: f64::INFINITY,
    singular: true,
    ill_conditioned: true,
    robust_covariance_usable: false,
    boundary,
  }
}
