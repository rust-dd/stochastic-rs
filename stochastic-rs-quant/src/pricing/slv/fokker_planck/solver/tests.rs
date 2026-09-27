use super::*;

/// Wyns & Du Toit (2016), §3: the Rannacher half-step solves the full
/// new-time operator. A Douglas split leaves a nonzero equation residual.
#[test]
fn rannacher_solves_the_unsplit_equation_at_the_new_time_level() {
  let mesh = Mesh::new(100.0_f64.ln(), 1.0, 0.2, 11, 0.05, 0.6, 0.01, 9);
  let along_v = Direction::along_v(&mesh, 2.0, 0.04, 0.4);
  let p = (0..mesh.len())
    .map(|i| 0.5 + (i as f64 * 0.7).sin().powi(2))
    .collect::<Vec<_>>();
  let old_leverage = vec![1.4; mesh.m1()];
  for attainable in [false, true] {
    let mut buffers = StepBuffers::new(mesh.len());
    for (rho, dt) in [(-0.6, 0.0025), (0.4, 0.02), (0.0, 0.01)] {
      let leverage = (0..mesh.m1())
        .map(|i| 0.6 + dt * i as f64)
        .collect::<Vec<_>>();
      let mut prev = Level::new(&mesh, 0.02, rho, 0.4, &old_leverage, attainable);
      let mut next = Level::new(&mesh, 0.02, rho, 0.4, &leverage, attainable);
      let mut out = vec![0.0; mesh.len()];
      step(
        &mesh,
        &along_v,
        &mut prev,
        &mut next,
        dt,
        0.5 + 3.0_f64.sqrt() / 6.0,
        true,
        &p,
        &mut out,
        &mut buffers,
      )
      .unwrap();
      let (mut a0, mut a1, mut a2) = (
        vec![0.0; mesh.len()],
        vec![0.0; mesh.len()],
        vec![0.0; mesh.len()],
      );
      next.mixed.apply(&out, &mut a0);
      next.x.apply_x(&mesh, &out, &mut a1);
      along_v.apply_v(&mesh, &out, &mut a2);
      for i in 0..mesh.len() {
        let residual = out[i] - dt * (a0[i] + a1[i] + a2[i]) - p[i];
        assert!(residual.abs() < 1e-11, "row {i}: residual {residual}");
      }
    }
  }
}

/// The forward corner mean excludes the singular zero-variance density
/// only when zero is attainable; the central mean retains it otherwise.
#[test]
fn the_mixed_boundary_flux_depends_on_attainability() {
  let mesh = Mesh::new(0.0, 1.0, 0.2, 9, 0.05, 0.6, 0.01, 9);
  let leverage = vec![1.0; mesh.m1()];
  let central = Mixed::new(&mesh, -0.6, 0.4, &leverage, false);
  let forward = Mixed::new(&mesh, -0.6, 0.4, &leverage, true);
  let expected = -0.6 * 0.4 * 0.5 * mesh.v[1];
  let central_entries = central.corner_entries(4, 1);
  let forward_entries = forward.corner_entries(4, 1);
  assert!(central_entries.iter().any(|(index, _)| *index < mesh.m1()));
  assert!(forward_entries.iter().all(|(index, _)| *index >= mesh.m1()));
  assert!((central_entries.iter().map(|(_, c)| c).sum::<f64>() - expected).abs() < 1e-15);
  assert!((forward_entries.iter().map(|(_, c)| c).sum::<f64>() - expected).abs() < 1e-15);
}

/// A Dirac start excites modes stiff in both directions. Four full Euler
/// half-steps bound the negative mass below 0.1%; the Douglas start gives
/// 2.1% on this mesh. Independently checked with SciPy's sparse LU applied
/// to the Wyns–Du Toit finite-volume operator: negative mass 0.000341566.
#[test]
fn rannacher_damps_the_dirac_start_in_both_directions() {
  let mesh = Mesh::new(
    100.0_f64.ln(),
    30.0_f64.ln(),
    0.2,
    121,
    0.05,
    0.75,
    0.01,
    60,
  );
  let leverage = vec![1.0; mesh.m1()];
  let along_v = Direction::along_v(&mesh, 2.0, 0.04, 0.4);
  let mut prev = Level::new(&mesh, 0.02, -0.6, 0.4, &leverage, true);
  let mut next = Level::new(&mesh, 0.02, -0.6, 0.4, &leverage, true);
  let mut buffers = StepBuffers::new(mesh.len());
  let mut p = mesh.dirac();
  let mut out = vec![0.0; mesh.len()];
  for _ in 0..4 {
    step(
      &mesh,
      &along_v,
      &mut prev,
      &mut next,
      0.0025,
      0.5 + 3.0_f64.sqrt() / 6.0,
      true,
      &p,
      &mut out,
      &mut buffers,
    )
    .unwrap();
    std::mem::swap(&mut p, &mut out);
  }
  let negative = p.iter().map(|value| (-value).max(0.0)).collect::<Vec<_>>();
  let negative_mass = mesh.mass(&negative);
  assert!(negative_mass < 1e-3, "negative mass {negative_mass}");
  assert!((negative_mass - 0.000_341_565_983_631_511_4).abs() < 1e-10);
  assert!((mesh.mass(&p) - 1.0).abs() < 1e-10);
}
