use super::*;
use crate::volatility::common::synthetic_surface_dataset;

const N_T: usize = 2;
const N_K: usize = 5;

fn trained() -> Result<StochVolNn> {
  let (lb, ub) = ([0.0_f32; 2], [1.0_f32; 2]);
  let spec = StochVolModelSpec::new(
    "bridge",
    8,
    lb.to_vec(),
    ub.to_vec(),
    vec![1.1, 0.8, 1.2, 0.9, 1.0],
    vec![0.5, 1.0],
  )?;
  let (params, surfaces) = synthetic_surface_dataset(&lb, &ub, 64, N_T * N_K, 5);
  let mut nn = StochVolNn::new(spec, &Device::Cpu, 5)?;
  let cfg = TrainConfig {
    epochs: 3,
    ..TrainConfig::default()
  };
  nn.train(&params, &surfaces, &cfg)?;
  Ok(nn)
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
  std::env::temp_dir().join(format!("stochastic_rs_model_{tag}_{}", std::process::id()))
}

#[test]
fn surface_bridge_pairs_each_strike_with_its_own_column() -> Result<()> {
  let nn = trained()?;
  let params = [0.3_f32, 0.6];
  let flat = nn.predict_surface(&params)?;
  let surface = nn.predict_implied_vol_surface(&params, 100.0, &[100.0; N_T])?;
  let want = [0.8, 0.9, 1.0, 1.1, 1.2].map(|k| k * 100.0);
  assert_eq!(surface.strikes, want);
  for t in 0..N_T {
    for (j, strike) in nn.spec().strikes.iter().enumerate() {
      let k = surface
        .strikes
        .iter()
        .position(|s| *s == strike * 100.0)
        .unwrap();
      assert_eq!(surface.ivs[[t, k]], f64::from(flat[t * N_K + j]));
    }
  }
  Ok(())
}

#[test]
fn a_wrong_forward_count_or_spot_is_an_error() -> Result<()> {
  let nn = trained()?;
  let params = [0.3_f32, 0.6];
  assert!(
    nn.predict_implied_vol_surface(&params, 100.0, &[100.0])
      .is_err()
  );
  for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
    assert!(
      nn.predict_implied_vol_surface(&params, 100.0, &[100.0, bad])
        .is_err()
    );
  }
  assert!(
    nn.predict_implied_vol_surface(&params, 0.0, &[1.0; N_T])
      .is_err()
  );
  assert!(
    nn.predict_implied_vol_surface(&params, f64::NAN, &[1.0; N_T])
      .is_err()
  );
  Ok(())
}

#[test]
fn the_grid_survives_save_and_load() -> Result<()> {
  let nn = trained()?;
  let dir = temp_dir("grid");
  nn.save(&dir)?;
  let back = StochVolNn::load("bridge", &dir, &Device::Cpu);
  let _ = fs::remove_dir_all(&dir);
  let back = back?;
  assert_eq!(back.spec().strikes, nn.spec().strikes);
  assert_eq!(back.spec().maturities, nn.spec().maturities);
  Ok(())
}

#[test]
fn a_format_1_directory_is_rejected_with_a_retrain_message() -> Result<()> {
  let nn = trained()?;
  let dir = temp_dir("v1");
  nn.save(&dir)?;
  let meta = dir.join(META_FILE);
  let old = fs::read_to_string(&meta)?.replacen("version=2", "version=1", 1);
  fs::write(&meta, old)?;
  let result = StochVolNn::load("bridge", &dir, &Device::Cpu);
  let _ = fs::remove_dir_all(&dir);
  let message = format!(
    "{:#}",
    result.err().expect("a format 1 directory must not load")
  );
  assert!(
    message.contains("format 1")
      && message.contains("only format 2")
      && message.contains("no grid"),
    "{message}"
  );
  Ok(())
}

#[test]
fn a_hand_built_bad_grid_or_an_overflowing_spot_is_an_error_not_a_panic() -> Result<()> {
  let mut nn = trained()?;
  let params = [0.3_f32, 0.6];
  assert!(
    nn.predict_implied_vol_surface(&params, f64::MAX, &[1.0; N_T])
      .is_err()
  );
  nn.spec.maturities = vec![1.0, 0.5];
  assert!(
    nn.predict_implied_vol_surface(&params, 100.0, &[1.0; N_T])
      .is_err()
  );
  nn.spec.maturities = vec![0.5, 1.0];
  nn.spec.strikes[1] = nn.spec.strikes[0];
  assert!(
    nn.predict_implied_vol_surface(&params, 100.0, &[1.0; N_T])
      .is_err()
  );
  Ok(())
}
