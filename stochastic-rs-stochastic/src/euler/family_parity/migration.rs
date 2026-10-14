//! Exact Metal Euler paths across every declared family and launch shape.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use ndarray::Array3;

use super::*;
use crate::device::Metal;

mod live_series;

const MAGIC: &[u8] = b"STOEULR1";
const FIRST: usize = 3;
const ROWS: usize = 3;
const SEEDS: [u64; 3] = [11, 37, 0x0000_0001_0000_000b];

struct ShapeProbe<'a, const D: usize> {
  base: &'a SystemProbe<D>,
  jumps: bool,
  gamma: bool,
  step_first: bool,
  terms: Option<u32>,
}

impl<const D: usize> crate::traits::Sealed for ShapeProbe<'_, D> {}

impl<const D: usize> ProcessExt<f32> for ShapeProbe<'_, D> {
  type Output = [Array1<f32>; D];
  type Sampler<'s>
    = SystemProbeSampler<D>
  where
    Self: 's;

  fn sampler(&self) -> Self::Sampler<'_> {
    self.base.sampler()
  }
}

impl<const D: usize> EulerSystem<f32, D> for ShapeProbe<'_, D> {
  fn euler_spec(&self) -> EulerSpec<f32> {
    self.base.euler_spec()
  }

  fn initial_state(&self) -> [f32; 4] {
    self.base.initial_state()
  }

  fn grid_points(&self) -> usize {
    self.base.grid_points()
  }

  fn horizon(&self) -> f32 {
    self.base.horizon()
  }

  fn time_step(&self) -> f32 {
    self.base.time_step()
  }

  fn device_seed(&self) -> u64 {
    self.base.device_seed()
  }

  fn curves(&self) -> Option<Vec<Vec<f32>>> {
    self.base.curves()
  }

  fn lift_spec(&self) -> Option<LiftSpec<'_, f32>> {
    self.base.lift_spec()
  }

  fn series_terms(&self) -> Option<u32> {
    self.terms.or_else(|| self.base.series_terms())
  }

  fn table_spec(&self) -> Option<TableSpec<f32>> {
    self.base.table_spec()
  }

  fn program_spec(&self) -> Option<ProgramSpec<'_>> {
    self.base.program_spec()
  }

  fn jump_intensity(&self) -> Option<f32> {
    self.jumps.then_some(PROBE_INTENSITY)
  }

  fn jump_sizes(&self) -> Option<JumpSizes<f32>> {
    self.jumps.then(|| probe_sizes(&self.base.spec))
  }

  fn gamma_draws(&self) -> Option<GammaDraws<f32>> {
    self.gamma.then_some(PROBE_GAMMAS)
  }

  fn step_first(&self) -> bool {
    self.step_first
  }

  fn host_sample(&self) -> Self::Output {
    self.base.host_sample()
  }
}

struct Golden {
  bytes: Vec<u8>,
  offset: usize,
  recording: bool,
  cases: usize,
}

impl Golden {
  fn new(recording: bool) -> Self {
    let mut golden = Self {
      bytes: if recording {
        Vec::new()
      } else {
        std::fs::read(fixture_path()).expect("record the Euler fixture before comparison")
      },
      offset: 0,
      recording,
      cases: 0,
    };
    golden.field(MAGIC, "fixture format");
    golden.field(
      &(super::super::families::Family::ALL.len() as u32).to_le_bytes(),
      "family count",
    );
    golden
  }

  fn field(&mut self, actual: &[u8], label: &str) {
    if self.recording {
      self.bytes.extend_from_slice(actual);
    } else {
      let end = self.offset + actual.len();
      let expected = self
        .bytes
        .get(self.offset..end)
        .unwrap_or_else(|| panic!("truncated Euler fixture at {label}"));
      assert_eq!(actual, expected, "Euler fixture layout: {label}");
      self.offset = end;
    }
  }

  fn paths(&mut self, key: &str, paths: &Array3<f32>) {
    self.field(&(key.len() as u32).to_le_bytes(), key);
    self.field(key.as_bytes(), key);
    self.field(&(paths.len() as u32).to_le_bytes(), key);
    for ((component, path, step), &value) in paths.indexed_iter() {
      let bits = value.to_bits().to_le_bytes();
      if self.recording {
        self.bytes.extend_from_slice(&bits);
      } else {
        let expected = self
          .bytes
          .get(self.offset..self.offset + 4)
          .unwrap_or_else(|| panic!("truncated Euler fixture: {key}"));
        let want = u32::from_le_bytes(expected.try_into().unwrap());
        assert_eq!(
          value.to_bits(),
          want,
          "{key}, component={component}, path={path}, step={step}: {value:?} != {:?}",
          f32::from_bits(want),
        );
        self.offset += 4;
      }
    }
    self.cases += 1;
  }

  fn finish(self) {
    let families = super::super::families::Family::ALL.len();
    assert_eq!(self.cases, families * (4 * SEEDS.len() + 1) + 6);
    if self.recording {
      let label = std::env::var("STOCHASTIC_RS_EULER_BASELINE_LABEL")
        .expect("set baseline revision, GPU, OS, Xcode/SDK and math mode in the baseline label");
      assert!(
        !label.contains(['\n', '\r']),
        "baseline label must be one line"
      );
      let path = fixture_path();
      let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("fixture destination must not already exist");
      file.write_all(&self.bytes).unwrap();
      let metadata = format!(
        "{label}\nformat=STOEULR1\nfamilies={families}\ncases={}\nbytes={}\n",
        self.cases,
        self.bytes.len(),
      );
      let mut provenance = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path.with_extension("meta"))
        .expect("fixture provenance destination must not already exist");
      provenance.write_all(metadata.as_bytes()).unwrap();
    } else {
      assert_eq!(self.offset, self.bytes.len(), "unread Euler fixture data");
    }
  }
}

fn fixture_path() -> PathBuf {
  std::env::var_os("STOCHASTIC_RS_EULER_FIXTURE")
    .map(PathBuf::from)
    .unwrap_or_else(|| {
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metal_euler.bin")
    })
}

fn launch<const D: usize>(
  device: &Metal,
  probe: &ShapeProbe<'_, D>,
  first: usize,
  rows: usize,
  seed: u64,
) -> Array3<f32> {
  let values = device
    .euler_system_kernel(probe, first, rows, seed)
    .unwrap_or_else(|error| panic!("{}: {error}", family_name(&probe.base.spec)));
  assert_eq!(values.dim(), (D, rows, N));
  values
}

fn assert_same(actual: &Array3<f32>, expected: &Array3<f32>, label: &str) {
  assert_eq!(actual.dim(), expected.dim(), "{label}");
  for ((component, path, step), &value) in actual.indexed_iter() {
    assert_eq!(
      value.to_bits(),
      expected[[component, path, step]].to_bits(),
      "{label}: component={component}, path={path}, step={step}",
    );
  }
}

fn check_chunking<const D: usize>(
  device: &Metal,
  probe: &ShapeProbe<'_, D>,
  expected: &Array3<f32>,
  first: usize,
  seed: u64,
  label: &str,
) {
  let mut joined = Array3::<f32>::zeros((D, ROWS, N));
  for (offset, rows) in [(0, 1), (1, 2)] {
    let part = launch(device, probe, first + offset, rows, seed);
    joined
      .slice_mut(ndarray::s![.., offset..offset + rows, ..])
      .assign(&part);
  }
  assert_same(&joined, expected, label);
}

fn case<const D: usize>(
  device: &Metal,
  probe: &ShapeProbe<'_, D>,
  first: usize,
  seed: u64,
  golden: &mut Golden,
  check: bool,
) {
  let family = probe.base.spec.encode().0;
  let label = format!(
    "family={family},name={},jumps={},gamma={},step_first={},terms={},first={first},rows={ROWS},seed={seed},planes={D},steps={N}",
    family_name(&probe.base.spec),
    probe.jumps,
    probe.gamma,
    probe.step_first,
    probe.series_terms().unwrap_or(0),
  );
  let paths = launch(device, probe, first, ROWS, seed);
  if check {
    let repeated = launch(device, probe, first, ROWS, seed);
    assert_same(&repeated, &paths, &format!("repeat: {label}"));
    check_chunking(
      device,
      probe,
      &paths,
      first,
      seed,
      &format!("chunks: {label}"),
    );
  }
  golden.paths(&label, &paths);
}

fn family<const D: usize>(device: &Metal, base: &SystemProbe<D>, golden: &mut Golden) {
  for jumps in [false, true] {
    for gamma in [false, true] {
      let probe = ShapeProbe {
        base,
        jumps,
        gamma,
        step_first: false,
        terms: None,
      };
      for seed in SEEDS {
        case(device, &probe, FIRST, seed, golden, seed == SEEDS[0]);
      }
    }
  }
  let probe = ShapeProbe {
    base,
    jumps: true,
    gamma: true,
    step_first: true,
    terms: None,
  };
  case(device, &probe, FIRST, SEEDS[0], golden, true);
  if matches!(base.spec.encode().0, 0 | 91) {
    case(
      device,
      &probe,
      u32::MAX as usize - ROWS,
      SEEDS[2],
      golden,
      true,
    );
  }
  if base.spec.encode().0 == 115 {
    for terms in [1, 511] {
      let probe = ShapeProbe {
        base,
        jumps: false,
        gamma: false,
        step_first: false,
        terms: Some(terms),
      };
      for seed in [SEEDS[0], SEEDS[2]] {
        case(device, &probe, FIRST, seed, golden, true);
      }
    }
  }
}

fn all_families(golden: &mut Golden) {
  let device = Metal::default();
  for probe in every_family() {
    let probe = SystemProbe {
      spec: probe.spec,
      x0: [probe.x0],
      lift: probe.lift,
      program: probe.program,
    };
    family(&device, &probe, golden);
  }
  for probe in every_two_component_family() {
    family(&device, &probe, golden);
  }
  for probe in every_three_component_family() {
    family(&device, &probe, golden);
  }
  for probe in every_four_component_family() {
    family(&device, &probe, golden);
  }
}

#[test]
#[ignore = "Record the Euler fixture explicitly on Metal hardware"]
fn record_euler_paths() {
  let label = std::env::var("STOCHASTIC_RS_EULER_BASELINE_LABEL")
    .expect("set baseline revision, GPU, OS, Xcode/SDK and math mode in the baseline label");
  assert!(
    !label.contains(['\n', '\r']),
    "baseline label must be one line"
  );
  assert!(
    !fixture_path().exists() && !fixture_path().with_extension("meta").exists(),
    "Euler fixture or provenance exists; choose a fresh output path"
  );
  let mut golden = Golden::new(true);
  all_families(&mut golden);
  golden.finish();
}

#[test]
#[ignore = "Run the complete Euler shape matrix explicitly on Metal hardware"]
fn euler_paths_match_recording() {
  let mut golden = Golden::new(false);
  all_families(&mut golden);
  golden.finish();
}
