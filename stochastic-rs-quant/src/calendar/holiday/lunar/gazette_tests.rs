use chrono::Datelike;
use chrono::NaiveDate;
use chrono::Weekday;

use super::is_hkex_holiday;
use super::is_sgx_holiday;

/// Hong Kong's weekday general holidays, `year: MM-DD ...`: the gov.hk list of
/// each year, which the HKEX holiday schedule repeats day for day.
const HKEX: &str = "
  2020: 01-01 01-27 01-28 04-10 04-13 04-30 05-01 06-25 07-01 10-01 10-02 10-26 12-25
  2021: 01-01 02-12 02-15 04-02 04-05 04-06 05-19 06-14 07-01 09-22 10-01 10-14 12-27
  2022: 02-01 02-02 02-03 04-05 04-15 04-18 05-02 05-09 06-03 07-01 09-12 10-04 12-26 12-27
  2023: 01-02 01-23 01-24 01-25 04-05 04-07 04-10 05-01 05-26 06-22 10-02 10-23 12-25 12-26
  2024: 01-01 02-12 02-13 03-29 04-01 04-04 05-01 05-15 06-10 07-01 09-18 10-01 10-11 12-25 12-26
  2025: 01-01 01-29 01-30 01-31 04-04 04-18 04-21 05-01 05-05 07-01 10-01 10-07 10-29 12-25 12-26
  2026: 01-01 02-17 02-18 02-19 04-03 04-06 04-07 05-01 05-25 06-19 07-01 10-01 10-19 12-25
  2027: 01-01 02-08 02-09 03-26 03-29 04-05 05-13 06-09 07-01 09-16 10-01 10-08 12-27
";

/// Singapore's weekday public holidays with their announced Monday
/// substitutes, `year: MM-DD ...`: the MOM list of each year, which the SGX CDP
/// settlement-holiday lists repeat for 2020–2026. Polling Day (2020-07-10,
/// 2023-09-01) is declared under the election Acts, not the Holidays Act, so
/// the calendar leaves it to `Calendar::add_holiday`.
const SGX: &str = "
  2020: 01-01 01-27 04-10 05-01 05-07 05-25 07-31 08-10 12-25
  2021: 01-01 02-12 04-02 05-13 05-26 07-20 08-09 11-04
  2022: 02-01 02-02 04-15 05-02 05-03 05-16 07-11 08-09 10-24 12-26
  2023: 01-02 01-23 01-24 04-07 05-01 06-02 06-29 08-09 11-13 12-25
  2024: 01-01 02-12 03-29 04-10 05-01 05-22 06-17 08-09 10-31 12-25
  2025: 01-01 01-29 01-30 03-31 04-18 05-01 05-12 10-20 12-25
  2026: 01-01 02-17 02-18 04-03 05-01 05-27 06-01 08-10 11-09 12-25
  2027: 01-01 02-08 03-10 03-26 05-17 05-20 08-09 10-28
";

fn weekday_closures(year: i32, is_holiday: fn(NaiveDate) -> bool) -> Vec<String> {
  NaiveDate::from_ymd_opt(year, 1, 1)
    .unwrap()
    .iter_days()
    .take_while(|date| date.year() == year)
    .filter(|date| !matches!(date.weekday(), Weekday::Sat | Weekday::Sun) && is_holiday(*date))
    .map(|date| date.format("%m-%d").to_string())
    .collect()
}

fn mismatches(gazette: &str, is_holiday: fn(NaiveDate) -> bool) -> Vec<String> {
  gazette
    .lines()
    .filter_map(|line| line.split_once(':'))
    .filter_map(|(year, days)| {
      let year = year.trim().parse::<i32>().unwrap();
      let gazetted = days
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
      let reported = weekday_closures(year, is_holiday);
      let extra = reported
        .iter()
        .filter(|day| !gazetted.contains(day))
        .collect::<Vec<_>>();
      let missing = gazetted
        .iter()
        .filter(|day| !reported.contains(day))
        .collect::<Vec<_>>();
      (!extra.is_empty() || !missing.is_empty()).then(|| {
        format!("{year}: closed but not gazetted {extra:?}, gazetted but open {missing:?}")
      })
    })
    .collect()
}

#[test]
fn hkex_closes_on_exactly_the_gazetted_weekdays_2020_to_2027() {
  let found = mismatches(HKEX, is_hkex_holiday);
  assert!(found.is_empty(), "HKEX {found:#?}");
}

#[test]
fn sgx_closes_on_exactly_the_gazetted_weekdays_2020_to_2027() {
  let found = mismatches(SGX, is_sgx_holiday);
  assert!(found.is_empty(), "SGX {found:#?}");
}
