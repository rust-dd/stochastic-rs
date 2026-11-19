use super::*;

// BSM vanilla call: S=100,K=100,r=0.05,q=0,sigma=0.2,T=1 → 10.4506
const BSM_VANILLA_CALL: f64 = 10.4506;
// BSM vanilla put: S=100,K=100,r=0.05,q=0,sigma=0.2,T=1 → 5.5735
const BSM_VANILLA_PUT: f64 = 5.5735;

fn base(barrier_type: BarrierType, option_type: OptionType, h: f64) -> f64 {
  let option = BarrierOption {
    strike: 100.0,
    barrier: h,
    rebate: 0.0,
    barrier_type,
    option_type,
  };
  BarrierPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0)
}

fn corridor(lower: f64, upper: f64) -> DoubleBarrierOption {
  DoubleBarrierOption {
    strike: 100.0,
    lower,
    upper,
    option_type: OptionType::Call,
  }
}

#[test]
fn in_out_parity_down_call() {
  let di = base(BarrierType::DownAndIn, OptionType::Call, 90.0);
  let do_ = base(BarrierType::DownAndOut, OptionType::Call, 90.0);
  assert!(
    (di + do_ - BSM_VANILLA_CALL).abs() < 0.1,
    "DI+DO={}",
    di + do_
  );
}

#[test]
fn in_out_parity_up_call() {
  let ui = base(BarrierType::UpAndIn, OptionType::Call, 110.0);
  let uo = base(BarrierType::UpAndOut, OptionType::Call, 110.0);
  assert!(
    (ui + uo - BSM_VANILLA_CALL).abs() < 0.1,
    "UI+UO={}",
    ui + uo
  );
}

#[test]
fn in_out_parity_down_put() {
  let di = base(BarrierType::DownAndIn, OptionType::Put, 90.0);
  let do_ = base(BarrierType::DownAndOut, OptionType::Put, 90.0);
  assert!(
    (di + do_ - BSM_VANILLA_PUT).abs() < 0.1,
    "DI+DO={}",
    di + do_
  );
}

#[test]
fn in_out_parity_up_put() {
  let ui = base(BarrierType::UpAndIn, OptionType::Put, 110.0);
  let uo = base(BarrierType::UpAndOut, OptionType::Put, 110.0);
  assert!((ui + uo - BSM_VANILLA_PUT).abs() < 0.1, "UI+UO={}", ui + uo);
}

#[test]
fn down_and_out_call_haug() {
  // Haug (2007): S=100, K=100, H=90, r=0.05, σ=0.2, T=1 → 8.6655
  let price = base(BarrierType::DownAndOut, OptionType::Call, 90.0);
  assert!((price - 8.6655).abs() < 0.05, "DO call={price}");
}

#[test]
fn knockout_less_than_vanilla() {
  let do_ = base(BarrierType::DownAndOut, OptionType::Call, 90.0);
  let uo = base(BarrierType::UpAndOut, OptionType::Call, 110.0);
  assert!(do_ < BSM_VANILLA_CALL + 0.5, "DO call={do_}");
  assert!(uo < BSM_VANILLA_CALL + 0.5, "UO call={uo}");
  assert!(do_ > 0.0);
  assert!(uo > 0.0);
}

#[test]
fn double_barrier_knockout() {
  let price = DoubleBarrierPricer::new(0.2).price(&corridor(80.0, 120.0), 100.0, 0.05, 0.0, 1.0);
  // Ikeda-Kunitomo series with narrow barriers can overshoot with few terms
  assert!(price > 0.0, "double barrier={price}");
}

#[test]
fn double_barrier_wide_converges_to_vanilla() {
  let price = DoubleBarrierPricer::new(0.2).price(&corridor(1.0, 500.0), 100.0, 0.05, 0.0, 1.0);
  assert!(
    (price - BSM_VANILLA_CALL).abs() < 0.5,
    "wide double barrier={price}"
  );
}

#[test]
#[should_panic(expected = "terms must satisfy `terms >= 0`, got terms = -1")]
fn double_barrier_refuses_a_negative_series_length() {
  let _ = DoubleBarrierPricer::new(0.2).with_terms(-1);
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0")]
fn a_closed_form_barrier_divides_by_sigma() {
  let _ = BarrierPricer::new(0.0);
}
