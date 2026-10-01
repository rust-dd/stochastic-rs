//! HKEX and SGX holiday calendars. Their lunar-calendar holidays follow the
//! Chinese lunisolar, Islamic and Hindu calendars, so the exchange closures
//! they cause are tabulated per year.
//!
//! Hong Kong sources: General Holidays Ordinance (Cap. 149), Schedule and
//! s 6(2); the general-holiday lists at gov.hk; the HKEX trading calendar and
//! holiday schedule; the Hong Kong Observatory Gregorian–lunar calendar
//! conversion tables. Singapore sources: Holidays Act 1998, ss 4(2) and 5(2);
//! the Ministry of Manpower (MOM) public-holiday lists; the SGX CDP
//! settlement-holiday lists.
//!
//! Singapore dates MOM has not gazetted are estimates. Vesak Day falls on the
//! fifteenth day of the fourth Chinese lunar month. Hari Raya Puasa and Hari
//! Raya Haji follow the MABIMS crescent criterion MUIS applies: altitude at
//! least 3° and elongation at least 6.4° at sunset in Singapore (Office of the
//! Mufti, Singapore, "Determining the Beginning of Ramadan 1445H/2024", 2024).
//! Deepavali falls on Naraka Chaturdashi, the day whose sunrise lies in the
//! fourteenth tithi before the new moon at which the Sun is in sidereal Libra
//! (Lahiri ayanamsa).

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Weekday;

use super::easter_sunday;

/// A lunar-calendar holiday under its local name. Each variant belongs to
/// exactly one exchange, so a row closes only that exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LunarHoliday {
  /// HKEX: Lunar New Year's Day and the two days after it, a Sunday among
  /// them replaced by the fourth day.
  LunarNewYear,
  /// HKEX: Ching Ming Festival.
  ChingMing,
  /// HKEX: the Birthday of the Buddha, the eighth day of the fourth lunar
  /// month.
  BuddhasBirthday,
  /// HKEX: Tuen Ng (Dragon Boat) Festival.
  DragonBoat,
  /// HKEX: the day following the Chinese Mid-Autumn Festival.
  MidAutumnNext,
  /// HKEX: Chung Yeung Festival.
  ChungYeung,
  /// SGX: the two days of Chinese New Year.
  ChineseNewYear,
  /// SGX: Vesak Day.
  Vesak,
  /// SGX: Hari Raya Puasa (Eid al-Fitr).
  EidAlFitr,
  /// SGX: Hari Raya Haji (Eid al-Adha).
  EidAlAdha,
  /// SGX: Deepavali.
  Deepavali,
}

/// Years for which the HKEX and SGX lunar-calendar closures are tabulated.
/// Outside this range those two calendars report only their rule-based
/// (fixed-date and Easter) holidays.
pub const LUNAR_TABLE_YEARS: std::ops::RangeInclusive<i32> = 2020..=2030;

/// Weekday exchange closures caused by the lunar-calendar holidays,
/// `(year, month, day, holiday)`, for [`LUNAR_TABLE_YEARS`]. A holiday that
/// falls on a Saturday has no row: Saturday is not a trading day and neither
/// exchange substitutes it. A Sunday holiday is recorded on its substitute
/// day. In Hong Kong a holiday that lands on another general holiday adds the
/// next free day (Cap. 149 s 6(2)); in Singapore a Sunday holiday moves to the
/// next day that is not already a public holiday (Holidays Act s 4(2)), and
/// two holidays on one weekday share it. Outside the window the HKEX and SGX
/// calendars report only their rule-based holidays.
static LUNAR_TABLE: &[(i32, u32, u32, LunarHoliday)] = &[
  // 2020 HKEX — gov.hk general holidays 2020, HKEX holiday schedule 2020
  (2020, 1, 27, LunarHoliday::LunarNewYear),
  (2020, 1, 28, LunarHoliday::LunarNewYear),
  (2020, 4, 30, LunarHoliday::BuddhasBirthday),
  (2020, 6, 25, LunarHoliday::DragonBoat),
  (2020, 10, 2, LunarHoliday::MidAutumnNext),
  (2020, 10, 26, LunarHoliday::ChungYeung),
  // 2020 SGX — MOM public holidays 2020, SGX CDP settlement holidays 2020
  (2020, 1, 27, LunarHoliday::ChineseNewYear),
  (2020, 5, 7, LunarHoliday::Vesak),
  (2020, 5, 25, LunarHoliday::EidAlFitr),
  (2020, 7, 31, LunarHoliday::EidAlAdha),
  // 2021 HKEX — gov.hk general holidays 2021, HKEX holiday schedule 2021
  (2021, 2, 12, LunarHoliday::LunarNewYear),
  (2021, 2, 15, LunarHoliday::LunarNewYear),
  (2021, 4, 6, LunarHoliday::ChingMing), // Easter Monday took 5 April
  (2021, 5, 19, LunarHoliday::BuddhasBirthday),
  (2021, 6, 14, LunarHoliday::DragonBoat),
  (2021, 9, 22, LunarHoliday::MidAutumnNext),
  (2021, 10, 14, LunarHoliday::ChungYeung),
  // 2021 SGX — MOM public holidays 2021, SGX CDP settlement holidays 2021
  (2021, 2, 12, LunarHoliday::ChineseNewYear),
  (2021, 5, 13, LunarHoliday::EidAlFitr),
  (2021, 5, 26, LunarHoliday::Vesak),
  (2021, 7, 20, LunarHoliday::EidAlAdha),
  (2021, 11, 4, LunarHoliday::Deepavali),
  // 2022 HKEX — gov.hk general holidays 2022, HKEX holiday schedule 2022
  (2022, 2, 1, LunarHoliday::LunarNewYear),
  (2022, 2, 2, LunarHoliday::LunarNewYear),
  (2022, 2, 3, LunarHoliday::LunarNewYear),
  (2022, 4, 5, LunarHoliday::ChingMing),
  (2022, 5, 9, LunarHoliday::BuddhasBirthday),
  (2022, 6, 3, LunarHoliday::DragonBoat),
  (2022, 9, 12, LunarHoliday::MidAutumnNext),
  (2022, 10, 4, LunarHoliday::ChungYeung),
  // 2022 SGX — MOM public holidays 2022, SGX CDP settlement holidays 2022
  (2022, 2, 1, LunarHoliday::ChineseNewYear),
  (2022, 2, 2, LunarHoliday::ChineseNewYear),
  (2022, 5, 3, LunarHoliday::EidAlFitr),
  (2022, 5, 16, LunarHoliday::Vesak),
  (2022, 7, 11, LunarHoliday::EidAlAdha),
  (2022, 10, 24, LunarHoliday::Deepavali),
  // 2023 HKEX — gov.hk general holidays 2023, HKEX holiday schedule 2023
  (2023, 1, 23, LunarHoliday::LunarNewYear),
  (2023, 1, 24, LunarHoliday::LunarNewYear),
  (2023, 1, 25, LunarHoliday::LunarNewYear),
  (2023, 4, 5, LunarHoliday::ChingMing),
  (2023, 5, 26, LunarHoliday::BuddhasBirthday),
  (2023, 6, 22, LunarHoliday::DragonBoat),
  (2023, 10, 23, LunarHoliday::ChungYeung),
  // 2023 SGX — MOM public holidays 2023, SGX CDP settlement holidays 2023
  (2023, 1, 23, LunarHoliday::ChineseNewYear),
  (2023, 1, 24, LunarHoliday::ChineseNewYear),
  (2023, 6, 2, LunarHoliday::Vesak),
  (2023, 6, 29, LunarHoliday::EidAlAdha),
  (2023, 11, 13, LunarHoliday::Deepavali),
  // 2024 HKEX — gov.hk general holidays 2024, HKEX holiday schedule 2024
  (2024, 2, 12, LunarHoliday::LunarNewYear),
  (2024, 2, 13, LunarHoliday::LunarNewYear),
  (2024, 4, 4, LunarHoliday::ChingMing),
  (2024, 5, 15, LunarHoliday::BuddhasBirthday),
  (2024, 6, 10, LunarHoliday::DragonBoat),
  (2024, 9, 18, LunarHoliday::MidAutumnNext),
  (2024, 10, 11, LunarHoliday::ChungYeung),
  // 2024 SGX — MOM public holidays 2024, SGX CDP settlement holidays 2024
  (2024, 2, 12, LunarHoliday::ChineseNewYear),
  (2024, 4, 10, LunarHoliday::EidAlFitr),
  (2024, 5, 22, LunarHoliday::Vesak),
  (2024, 6, 17, LunarHoliday::EidAlAdha),
  (2024, 10, 31, LunarHoliday::Deepavali),
  // 2025 HKEX — gov.hk general holidays 2025, HKEX holiday schedule 2025
  (2025, 1, 29, LunarHoliday::LunarNewYear),
  (2025, 1, 30, LunarHoliday::LunarNewYear),
  (2025, 1, 31, LunarHoliday::LunarNewYear),
  (2025, 4, 4, LunarHoliday::ChingMing),
  (2025, 5, 5, LunarHoliday::BuddhasBirthday),
  (2025, 10, 7, LunarHoliday::MidAutumnNext),
  (2025, 10, 29, LunarHoliday::ChungYeung),
  // 2025 SGX — MOM public holidays 2025, SGX CDP settlement holidays 2025
  (2025, 1, 29, LunarHoliday::ChineseNewYear),
  (2025, 1, 30, LunarHoliday::ChineseNewYear),
  (2025, 3, 31, LunarHoliday::EidAlFitr),
  (2025, 5, 12, LunarHoliday::Vesak),
  (2025, 10, 20, LunarHoliday::Deepavali),
  // 2026 HKEX — gov.hk general holidays 2026, HKEX holiday schedule 2026
  (2026, 2, 17, LunarHoliday::LunarNewYear),
  (2026, 2, 18, LunarHoliday::LunarNewYear),
  (2026, 2, 19, LunarHoliday::LunarNewYear),
  (2026, 4, 7, LunarHoliday::ChingMing), // Easter Monday took 6 April
  (2026, 5, 25, LunarHoliday::BuddhasBirthday),
  (2026, 6, 19, LunarHoliday::DragonBoat),
  (2026, 10, 19, LunarHoliday::ChungYeung),
  // 2026 SGX — MOM public holidays 2026, SGX CDP settlement holidays 2026
  (2026, 2, 17, LunarHoliday::ChineseNewYear),
  (2026, 2, 18, LunarHoliday::ChineseNewYear),
  (2026, 5, 27, LunarHoliday::EidAlAdha),
  (2026, 6, 1, LunarHoliday::Vesak),
  (2026, 11, 9, LunarHoliday::Deepavali),
  // 2027 HKEX — gov.hk general holidays 2027, HKEX holiday schedule 2027
  (2027, 2, 8, LunarHoliday::LunarNewYear),
  (2027, 2, 9, LunarHoliday::LunarNewYear),
  (2027, 4, 5, LunarHoliday::ChingMing),
  (2027, 5, 13, LunarHoliday::BuddhasBirthday),
  (2027, 6, 9, LunarHoliday::DragonBoat),
  (2027, 9, 16, LunarHoliday::MidAutumnNext),
  (2027, 10, 8, LunarHoliday::ChungYeung),
  // 2027 SGX — MOM public holidays 2027, data.gov.sg public holidays 2027
  (2027, 2, 8, LunarHoliday::ChineseNewYear),
  (2027, 3, 10, LunarHoliday::EidAlFitr),
  (2027, 5, 17, LunarHoliday::EidAlAdha),
  (2027, 5, 20, LunarHoliday::Vesak),
  (2027, 10, 28, LunarHoliday::Deepavali),
  // 2028 HKEX — HKO lunar calendar 2028 under Cap. 149, not yet gazetted
  (2028, 1, 26, LunarHoliday::LunarNewYear),
  (2028, 1, 27, LunarHoliday::LunarNewYear),
  (2028, 1, 28, LunarHoliday::LunarNewYear),
  (2028, 4, 4, LunarHoliday::ChingMing),
  (2028, 5, 2, LunarHoliday::BuddhasBirthday),
  (2028, 5, 29, LunarHoliday::DragonBoat),
  (2028, 10, 4, LunarHoliday::MidAutumnNext),
  (2028, 10, 26, LunarHoliday::ChungYeung),
  // 2028 SGX — HKO lunar calendar 2028 for Chinese New Year; Vesak Day, Hari Raya
  // Puasa, Hari Raya Haji and Deepavali are estimates until MOM gazettes them
  (2028, 1, 26, LunarHoliday::ChineseNewYear),
  (2028, 1, 27, LunarHoliday::ChineseNewYear),
  (2028, 2, 28, LunarHoliday::EidAlFitr),
  (2028, 5, 5, LunarHoliday::EidAlAdha),
  (2028, 5, 9, LunarHoliday::Vesak),
  (2028, 10, 17, LunarHoliday::Deepavali),
  // 2029 HKEX — HKO lunar calendar 2029 under Cap. 149, not yet gazetted
  (2029, 2, 13, LunarHoliday::LunarNewYear),
  (2029, 2, 14, LunarHoliday::LunarNewYear),
  (2029, 2, 15, LunarHoliday::LunarNewYear),
  (2029, 4, 4, LunarHoliday::ChingMing),
  (2029, 5, 21, LunarHoliday::BuddhasBirthday),
  (2029, 9, 24, LunarHoliday::MidAutumnNext),
  (2029, 10, 16, LunarHoliday::ChungYeung),
  // 2029 SGX — HKO lunar calendar 2029 for Chinese New Year; Vesak Day, Hari Raya
  // Puasa, Hari Raya Haji and Deepavali are estimates until MOM gazettes them
  (2029, 2, 13, LunarHoliday::ChineseNewYear),
  (2029, 2, 14, LunarHoliday::ChineseNewYear),
  (2029, 2, 15, LunarHoliday::EidAlFitr),
  (2029, 4, 25, LunarHoliday::EidAlAdha),
  (2029, 5, 28, LunarHoliday::Vesak),
  (2029, 11, 5, LunarHoliday::Deepavali),
  // 2030 HKEX — HKO lunar calendar 2030 under Cap. 149, not yet gazetted
  (2030, 2, 4, LunarHoliday::LunarNewYear),
  (2030, 2, 5, LunarHoliday::LunarNewYear),
  (2030, 2, 6, LunarHoliday::LunarNewYear),
  (2030, 4, 5, LunarHoliday::ChingMing),
  (2030, 5, 9, LunarHoliday::BuddhasBirthday),
  (2030, 6, 5, LunarHoliday::DragonBoat),
  (2030, 9, 13, LunarHoliday::MidAutumnNext),
  // 2030 SGX — HKO lunar calendar 2030 for Chinese New Year; Vesak Day, Hari Raya
  // Puasa and Hari Raya Haji are estimates until MOM gazettes them. The Deepavali
  // estimate, 26 October, is a Saturday; Hari Raya Puasa shares 4 February with
  // Chinese New Year, which adds no day unless the President appoints one (s 5(2)).
  (2030, 2, 4, LunarHoliday::ChineseNewYear),
  (2030, 2, 5, LunarHoliday::ChineseNewYear),
  (2030, 2, 4, LunarHoliday::EidAlFitr),
  (2030, 4, 15, LunarHoliday::EidAlAdha),
  (2030, 5, 16, LunarHoliday::Vesak),
];

fn lunar_holiday_on(date: NaiveDate, tag: LunarHoliday) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());
  LUNAR_TABLE
    .iter()
    .any(|&(yr, mo, da, t)| yr == y && mo == m && da == d && t == tag)
}

/// HKEX (Hong Kong Stock Exchange) calendar: the rule-based general holidays
/// plus the lunar-calendar closures in [`LUNAR_TABLE`]. Outside
/// [`LUNAR_TABLE_YEARS`] only the rule-based holidays are reported.
pub(super) fn is_hkex_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());
  let w = date.weekday();
  use Weekday::*;

  // New Year's Day (Jan 1, observed Mon if weekend).
  if (m == 1 && d == 1 && w != Sat && w != Sun)
    || (m == 1 && d == 2 && w == Mon)
    || (m == 1 && d == 3 && w == Mon)
  {
    return true;
  }

  let easter = easter_sunday(y);

  // Good Friday, day after Good Friday (HKEX convention), Easter Monday.
  if date == easter - Duration::days(2)
    || date == easter - Duration::days(1)
    || date == easter + Duration::days(1)
  {
    return true;
  }

  // Labour Day (May 1, observed Mon if Sun).
  if (m == 5 && d == 1 && w != Sat && w != Sun) || (m == 5 && d == 2 && w == Mon) {
    return true;
  }

  // HKSAR Establishment Day (Jul 1, observed Mon if Sun).
  if (m == 7 && d == 1 && w != Sat && w != Sun) || (m == 7 && d == 2 && w == Mon) {
    return true;
  }

  // National Day (Oct 1, observed Mon if Sun).
  if (m == 10 && d == 1 && w != Sat && w != Sun) || (m == 10 && d == 2 && w == Mon) {
    return true;
  }

  // Christmas Day + Boxing Day, observed.
  if (m == 12 && d == 25 && w != Sat && w != Sun) || (m == 12 && d == 27 && matches!(w, Mon | Tue))
  {
    return true;
  }
  if (m == 12 && d == 26 && w != Sat && w != Sun) || (m == 12 && d == 28 && matches!(w, Mon | Tue))
  {
    return true;
  }

  // Lunar / solar-term holidays via lookup.
  for tag in [
    LunarHoliday::LunarNewYear,
    LunarHoliday::ChingMing,
    LunarHoliday::BuddhasBirthday,
    LunarHoliday::DragonBoat,
    LunarHoliday::MidAutumnNext,
    LunarHoliday::ChungYeung,
  ] {
    if lunar_holiday_on(date, tag) {
      return true;
    }
  }

  false
}

/// SGX (Singapore Exchange) calendar: the rule-based public holidays, a
/// Sunday one observed on the Monday, plus the lunar-calendar closures in
/// [`LUNAR_TABLE`]. Outside [`LUNAR_TABLE_YEARS`] only the rule-based
/// holidays are reported.
pub(super) fn is_sgx_holiday(date: NaiveDate) -> bool {
  let (_y, m, d) = (date.year(), date.month(), date.day());
  let w = date.weekday();
  use Weekday::*;

  // New Year's Day (observed Mon if Sun; Sat stays).
  if (m == 1 && d == 1 && w != Sat && w != Sun) || (m == 1 && d == 2 && w == Mon) {
    return true;
  }

  let easter = easter_sunday(date.year());

  // Good Friday.
  if date == easter - Duration::days(2) {
    return true;
  }

  // Labour Day (observed Mon if Sun).
  if (m == 5 && d == 1 && w != Sat && w != Sun) || (m == 5 && d == 2 && w == Mon) {
    return true;
  }

  // National Day (Aug 9, observed Mon if Sun).
  if (m == 8 && d == 9 && w != Sat && w != Sun) || (m == 8 && d == 10 && w == Mon) {
    return true;
  }

  // Christmas Day (observed Mon if Sun).
  if (m == 12 && d == 25 && w != Sat && w != Sun) || (m == 12 && d == 26 && w == Mon) {
    return true;
  }

  for tag in [
    LunarHoliday::ChineseNewYear,
    LunarHoliday::Vesak,
    LunarHoliday::EidAlFitr,
    LunarHoliday::EidAlAdha,
    LunarHoliday::Deepavali,
  ] {
    if lunar_holiday_on(date, tag) {
      return true;
    }
  }

  false
}

#[cfg(test)]
mod tests {
  use chrono::NaiveDate;

  use super::super::Calendar;
  use super::super::HolidayCalendar;

  #[test]
  fn hkex_2024_official_holidays() {
    let cal = Calendar::new(HolidayCalendar::Hkex);
    let dates = [
      (2024, 1, 1),   // New Year
      (2024, 2, 12),  // third day of Lunar New Year (Day 1 = Sat Feb 10)
      (2024, 2, 13),  // fourth day, for the second day on Sun Feb 11
      (2024, 3, 29),  // Good Friday
      (2024, 4, 1),   // Easter Monday
      (2024, 4, 4),   // Ching Ming
      (2024, 5, 1),   // Labour Day
      (2024, 5, 15),  // Buddha's Birthday
      (2024, 6, 10),  // Tuen Ng
      (2024, 7, 1),   // HKSAR
      (2024, 9, 18),  // Day after Mid-Autumn
      (2024, 10, 1),  // National Day
      (2024, 10, 11), // Chung Yeung
      (2024, 12, 25), // Christmas
      (2024, 12, 26), // Boxing Day
    ];
    for (y, m, d) in dates {
      let date = NaiveDate::from_ymd_opt(y, m, d).unwrap();
      assert!(cal.is_holiday(date), "HKEX missed: {date}");
    }
  }

  #[test]
  fn sgx_2024_core_holidays() {
    let cal = Calendar::new(HolidayCalendar::Sgx);
    let dates = [
      (2024, 1, 1),   // New Year
      (2024, 2, 12),  // Chinese New Year, for the second day on Sun Feb 11
      (2024, 3, 29),  // Good Friday
      (2024, 4, 10),  // Eid al-Fitr
      (2024, 5, 1),   // Labour Day
      (2024, 5, 22),  // Vesak
      (2024, 6, 17),  // Eid al-Adha
      (2024, 8, 9),   // National Day
      (2024, 10, 31), // Deepavali
      (2024, 12, 25), // Christmas
    ];
    for (y, m, d) in dates {
      let date = NaiveDate::from_ymd_opt(y, m, d).unwrap();
      assert!(cal.is_holiday(date), "SGX missed: {date}");
    }
  }
}

#[cfg(test)]
mod lunar_coverage {
  use chrono::NaiveDate;

  use super::*;

  fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
  }

  #[test]
  fn hkex_closes_for_lunar_new_year_2026() {
    for day in [17, 18, 19] {
      assert!(is_hkex_holiday(d(2026, 2, day)), "2026-02-{day}");
    }
  }

  #[test]
  fn sgx_closes_for_lunar_new_year_2026() {
    for day in [17, 18] {
      assert!(is_sgx_holiday(d(2026, 2, day)), "2026-02-{day}");
    }
  }

  #[test]
  fn every_covered_year_has_both_exchanges_lunar_new_year() {
    for y in LUNAR_TABLE_YEARS {
      for tag in [LunarHoliday::LunarNewYear, LunarHoliday::ChineseNewYear] {
        assert!(
          LUNAR_TABLE.iter().any(|&(yr, _, _, t)| yr == y && t == tag),
          "no {tag:?} entry for {y}"
        );
      }
    }
  }

  #[test]
  fn a_date_past_the_table_does_not_panic() {
    let _ = is_hkex_holiday(d(2040, 2, 1));
    let _ = is_sgx_holiday(d(2040, 2, 1));
  }

  #[test]
  fn each_exchange_closes_only_for_its_own_holidays() {
    for (date, hkex, sgx) in [
      (d(2026, 2, 19), true, false), // third day of Lunar New Year
      (d(2026, 5, 25), true, false), // Birthday of the Buddha, observed
      (d(2026, 6, 1), false, true),  // Vesak Day, observed
    ] {
      assert_eq!(is_hkex_holiday(date), hkex, "HKEX on {date}");
      assert_eq!(is_sgx_holiday(date), sgx, "SGX on {date}");
    }
  }

  /// Ching Ming on Easter Sunday moves to Monday, which is Easter Monday, so
  /// General Holidays Ordinance s 6(2) closes the Tuesday as well.
  #[test]
  fn ching_ming_displaced_onto_easter_monday_closes_the_tuesday() {
    for date in [d(2021, 4, 6), d(2026, 4, 7)] {
      assert!(is_hkex_holiday(date), "{date}");
    }
  }

  #[test]
  fn the_2020_closures_match_the_exchange_calendars() {
    assert!(
      !is_hkex_holiday(d(2020, 1, 29)),
      "HKEX reopened after the fourth day of Lunar New Year"
    );
    assert!(
      is_sgx_holiday(d(2020, 5, 25)),
      "Hari Raya Puasa fell on Sunday 24 May"
    );
  }

  #[test]
  fn every_row_is_a_weekday_inside_the_window() {
    for &(y, m, day, tag) in LUNAR_TABLE {
      let date = d(y, m, day);
      assert!(
        LUNAR_TABLE_YEARS.contains(&y),
        "{date} {tag:?} lies outside the window"
      );
      assert!(
        !matches!(date.weekday(), Weekday::Sat | Weekday::Sun),
        "{date} {tag:?} falls on a weekend"
      );
    }
  }
}
