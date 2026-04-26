//! ISO 4217 currency definitions and metadata.
//!
//! Complete set of active ISO 4217 currencies including G10, emerging
//! markets, precious metals, and special drawing rights.
//!
//! Reference: ISO 4217:2015 — Codes for the representation of currencies.

/// An ISO 4217 currency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Currency {
  /// ISO 4217 alphabetic code (e.g. "USD").
  pub code: &'static str,
  /// ISO 4217 numeric code (e.g. 840).
  pub numeric: u16,
  /// Full English name.
  pub name: &'static str,
  /// Common symbol (e.g. "$").
  pub symbol: &'static str,
  /// Number of minor-unit decimal places (e.g. 2 for cents).
  pub minor_unit: u8,
}

impl std::fmt::Display for Currency {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.code)
  }
}

macro_rules! define_currencies {
  ($( $name:ident { $code:literal, $num:literal, $cname:literal, $sym:literal, $mu:literal } ),* $(,)?) => {
    $(
      pub const $name: Currency = Currency {
        code: $code, numeric: $num, name: $cname, symbol: $sym, minor_unit: $mu,
      };
    )*

    /// All defined currencies.
    pub const ALL_CURRENCIES: &[Currency] = &[$($name),*];

    /// Look up a currency by its ISO 4217 alphabetic code.
    pub fn from_code(code: &str) -> Option<Currency> {
      match code {
        $( $code => Some($name), )*
        _ => None,
      }
    }

    /// Look up a currency by its ISO 4217 numeric code.
    pub fn from_numeric(numeric: u16) -> Option<Currency> {
      match numeric {
        $( $num => Some($name), )*
        _ => None,
      }
    }
  };
}

define_currencies! {
  // G10 currencies
  USD { "USD", 840, "US dollar", "$", 2 },
  EUR { "EUR", 978, "Euro", "€", 2 },
  GBP { "GBP", 826, "Pound sterling", "£", 2 },
  JPY { "JPY", 392, "Japanese yen", "¥", 0 },
  CHF { "CHF", 756, "Swiss franc", "CHF", 2 },
  AUD { "AUD", 36, "Australian dollar", "A$", 2 },
  NZD { "NZD", 554, "New Zealand dollar", "NZ$", 2 },
  CAD { "CAD", 124, "Canadian dollar", "C$", 2 },
  SEK { "SEK", 752, "Swedish krona", "kr", 2 },
  NOK { "NOK", 578, "Norwegian krone", "kr", 2 },

  // Europe (non-G10)
  DKK { "DKK", 208, "Danish krone", "kr", 2 },
  PLN { "PLN", 985, "Polish zloty", "zł", 2 },
  CZK { "CZK", 203, "Czech koruna", "Kč", 2 },
  HUF { "HUF", 348, "Hungarian forint", "Ft", 2 },
  RON { "RON", 946, "Romanian leu", "lei", 2 },
  BGN { "BGN", 975, "Bulgarian lev", "лв", 2 },
  RSD { "RSD", 941, "Serbian dinar", "din", 2 },
  ISK { "ISK", 352, "Icelandic krona", "kr", 0 },
  TRY { "TRY", 949, "Turkish lira", "₺", 2 },
  UAH { "UAH", 980, "Ukrainian hryvnia", "₴", 2 },
  GEL { "GEL", 981, "Georgian lari", "₾", 2 },
  AMD { "AMD", 51, "Armenian dram", "֏", 2 },
  AZN { "AZN", 944, "Azerbaijani manat", "₼", 2 },
  BYN { "BYN", 933, "Belarusian ruble", "Br", 2 },
  MDL { "MDL", 498, "Moldovan leu", "L", 2 },
  MKD { "MKD", 807, "Macedonian denar", "ден", 2 },
  ALL { "ALL", 8, "Albanian lek", "L", 2 },
  BAM { "BAM", 977, "Bosnia mark", "KM", 2 },
  RUB { "RUB", 643, "Russian ruble", "₽", 2 },
  GIP { "GIP", 292, "Gibraltar pound", "£", 2 },

  // Americas
  BRL { "BRL", 986, "Brazilian real", "R$", 2 },
  MXN { "MXN", 484, "Mexican peso", "Mex$", 2 },
  ARS { "ARS", 32, "Argentine peso", "$", 2 },
  CLP { "CLP", 152, "Chilean peso", "$", 0 },
  CLF { "CLF", 990, "Chilean Unidad de Fomento", "UF", 4 },
  COP { "COP", 170, "Colombian peso", "$", 2 },
  PEN { "PEN", 604, "Peruvian sol", "S/", 2 },
  UYU { "UYU", 858, "Uruguayan peso", "$U", 2 },
  PYG { "PYG", 600, "Paraguayan guarani", "₲", 0 },
  BOB { "BOB", 68, "Bolivian boliviano", "Bs.", 2 },
  VES { "VES", 928, "Venezuelan bolivar soberano", "Bs.S", 2 },
  VED { "VED", 926, "Venezuelan bolivar digital", "Bs.D", 2 },
  DOP { "DOP", 214, "Dominican peso", "RD$", 2 },
  GTQ { "GTQ", 320, "Guatemalan quetzal", "Q", 2 },
  HNL { "HNL", 340, "Honduran lempira", "L", 2 },
  NIO { "NIO", 558, "Nicaraguan cordoba", "C$", 2 },
  CRC { "CRC", 188, "Costa Rican colon", "₡", 2 },
  PAB { "PAB", 590, "Panamanian balboa", "B/.", 2 },
  TTD { "TTD", 780, "Trinidad and Tobago dollar", "TT$", 2 },
  JMD { "JMD", 388, "Jamaican dollar", "J$", 2 },
  BBD { "BBD", 52, "Barbadian dollar", "Bds$", 2 },
  BSD { "BSD", 44, "Bahamian dollar", "B$", 2 },
  BZD { "BZD", 84, "Belize dollar", "BZ$", 2 },
  GYD { "GYD", 328, "Guyanese dollar", "G$", 2 },
  SRD { "SRD", 968, "Surinamese dollar", "SRD", 2 },
  HTG { "HTG", 332, "Haitian gourde", "G", 2 },
  SVC { "SVC", 222, "Salvadoran colon", "₡", 2 },
  CUP { "CUP", 192, "Cuban peso", "₱", 2 },
  AWG { "AWG", 533, "Aruban florin", "ƒ", 2 },
  BMD { "BMD", 60, "Bermudian dollar", "BD$", 2 },
  KYD { "KYD", 136, "Cayman Islands dollar", "CI$", 2 },
  XCD { "XCD", 951, "East Caribbean dollar", "EC$", 2 },
  ANG { "ANG", 532, "Netherlands Antillean guilder", "ƒ", 2 },
  FKP { "FKP", 238, "Falkland Islands pound", "FK£", 2 },

  // Asia & Pacific
  CNY { "CNY", 156, "Chinese yuan", "¥", 2 },
  HKD { "HKD", 344, "Hong Kong dollar", "HK$", 2 },
  MOP { "MOP", 446, "Macanese pataca", "MOP$", 2 },
  SGD { "SGD", 702, "Singapore dollar", "S$", 2 },
  INR { "INR", 356, "Indian rupee", "₹", 2 },
  KRW { "KRW", 410, "South Korean won", "₩", 0 },
  KPW { "KPW", 408, "North Korean won", "₩", 2 },
  TWD { "TWD", 901, "New Taiwan dollar", "NT$", 2 },
  THB { "THB", 764, "Thai baht", "฿", 2 },
  IDR { "IDR", 360, "Indonesian rupiah", "Rp", 2 },
  MYR { "MYR", 458, "Malaysian ringgit", "RM", 2 },
  PHP { "PHP", 608, "Philippine peso", "₱", 2 },
  VND { "VND", 704, "Vietnamese dong", "₫", 0 },
  PKR { "PKR", 586, "Pakistani rupee", "₨", 2 },
  BDT { "BDT", 50, "Bangladeshi taka", "৳", 2 },
  LKR { "LKR", 144, "Sri Lankan rupee", "Rs", 2 },
  NPR { "NPR", 524, "Nepalese rupee", "Rs", 2 },
  BTN { "BTN", 64, "Bhutanese ngultrum", "Nu", 2 },
  MMK { "MMK", 104, "Myanmar kyat", "K", 2 },
  KHR { "KHR", 116, "Cambodian riel", "៛", 2 },
  LAK { "LAK", 418, "Lao kip", "₭", 2 },
  MNT { "MNT", 496, "Mongolian tugrik", "₮", 2 },
  KZT { "KZT", 398, "Kazakhstani tenge", "₸", 2 },
  UZS { "UZS", 860, "Uzbekistani som", "сўм", 2 },
  KGS { "KGS", 417, "Kyrgyzstani som", "сом", 2 },
  TJS { "TJS", 972, "Tajikistani somoni", "SM", 2 },
  TMT { "TMT", 934, "Turkmenistani manat", "T", 2 },
  AFN { "AFN", 971, "Afghan afghani", "؋", 2 },
  BND { "BND", 96, "Brunei dollar", "B$", 2 },
  FJD { "FJD", 242, "Fijian dollar", "FJ$", 2 },
  PGK { "PGK", 598, "Papua New Guinean kina", "K", 2 },
  WST { "WST", 882, "Samoan tala", "WS$", 2 },
  TOP { "TOP", 776, "Tongan paʻanga", "T$", 2 },
  VUV { "VUV", 548, "Vanuatu vatu", "VT", 0 },
  SBD { "SBD", 90, "Solomon Islands dollar", "SI$", 2 },
  XPF { "XPF", 953, "CFP franc", "F", 0 },
  MVR { "MVR", 462, "Maldivian rufiyaa", "Rf", 2 },

  // Middle East
  ILS { "ILS", 376, "Israeli new shekel", "₪", 2 },
  SAR { "SAR", 682, "Saudi riyal", "﷼", 2 },
  AED { "AED", 784, "UAE dirham", "د.إ", 2 },
  QAR { "QAR", 634, "Qatari riyal", "ر.ق", 2 },
  KWD { "KWD", 414, "Kuwaiti dinar", "د.ك", 3 },
  BHD { "BHD", 48, "Bahraini dinar", "BD", 3 },
  OMR { "OMR", 512, "Omani rial", "ر.ع.", 3 },
  JOD { "JOD", 400, "Jordanian dinar", "JD", 3 },
  LBP { "LBP", 422, "Lebanese pound", "ل.ل", 2 },
  IQD { "IQD", 368, "Iraqi dinar", "ع.د", 3 },
  IRR { "IRR", 364, "Iranian rial", "﷼", 2 },
  SYP { "SYP", 760, "Syrian pound", "£S", 2 },
  YER { "YER", 886, "Yemeni rial", "﷼", 2 },

  // Africa
  ZAR { "ZAR", 710, "South African rand", "R", 2 },
  NGN { "NGN", 566, "Nigerian naira", "₦", 2 },
  EGP { "EGP", 818, "Egyptian pound", "E£", 2 },
  KES { "KES", 404, "Kenyan shilling", "KSh", 2 },
  GHS { "GHS", 936, "Ghanaian cedi", "GH₵", 2 },
  TZS { "TZS", 834, "Tanzanian shilling", "TSh", 2 },
  UGX { "UGX", 800, "Ugandan shilling", "USh", 0 },
  MAD { "MAD", 504, "Moroccan dirham", "د.م.", 2 },
  TND { "TND", 788, "Tunisian dinar", "د.ت", 3 },
  DZD { "DZD", 12, "Algerian dinar", "د.ج", 2 },
  MRU { "MRU", 929, "Mauritanian ouguiya", "UM", 2 },
  LYD { "LYD", 434, "Libyan dinar", "ل.د", 3 },
  ETB { "ETB", 230, "Ethiopian birr", "Br", 2 },
  XOF { "XOF", 952, "West African CFA franc", "CFA", 0 },
  XAF { "XAF", 950, "Central African CFA franc", "FCFA", 0 },
  MUR { "MUR", 480, "Mauritian rupee", "₨", 2 },
  SCR { "SCR", 690, "Seychellois rupee", "₨", 2 },
  BWP { "BWP", 72, "Botswana pula", "P", 2 },
  MZN { "MZN", 943, "Mozambican metical", "MT", 2 },
  ZMW { "ZMW", 967, "Zambian kwacha", "ZK", 2 },
  MWK { "MWK", 454, "Malawian kwacha", "MK", 2 },
  AOA { "AOA", 973, "Angolan kwanza", "Kz", 2 },
  NAD { "NAD", 516, "Namibian dollar", "N$", 2 },
  RWF { "RWF", 646, "Rwandan franc", "RF", 0 },
  CDF { "CDF", 976, "Congolese franc", "FC", 2 },
  SOS { "SOS", 706, "Somali shilling", "Sh", 2 },
  DJF { "DJF", 262, "Djiboutian franc", "Fdj", 0 },
  SDG { "SDG", 938, "Sudanese pound", "ج.س.", 2 },
  SSP { "SSP", 728, "South Sudanese pound", "SS£", 2 },
  ERN { "ERN", 232, "Eritrean nakfa", "Nfk", 2 },
  GMD { "GMD", 270, "Gambian dalasi", "D", 2 },
  GNF { "GNF", 324, "Guinean franc", "FG", 0 },
  SLL { "SLL", 694, "Sierra Leonean leone (old)", "Le", 2 },
  SLE { "SLE", 925, "Sierra Leonean leone", "Le", 2 },
  LRD { "LRD", 430, "Liberian dollar", "L$", 2 },
  CVE { "CVE", 132, "Cape Verdean escudo", "Esc", 2 },
  STN { "STN", 930, "Sao Tome and Principe dobra", "Db", 2 },
  MGA { "MGA", 969, "Malagasy ariary", "Ar", 2 },
  KMF { "KMF", 174, "Comorian franc", "CF", 0 },
  BIF { "BIF", 108, "Burundian franc", "FBu", 0 },
  SZL { "SZL", 748, "Eswatini lilangeni", "E", 2 },
  LSL { "LSL", 426, "Lesotho loti", "L", 2 },
  ZWG { "ZWG", 924, "Zimbabwe Gold", "ZiG", 2 },

  // Precious metals & special
  XAU { "XAU", 959, "Gold (troy ounce)", "XAU", 0 },
  XAG { "XAG", 961, "Silver (troy ounce)", "XAG", 0 },
  XPT { "XPT", 962, "Platinum (troy ounce)", "XPT", 0 },
  XPD { "XPD", 964, "Palladium (troy ounce)", "XPD", 0 },
  XDR { "XDR", 960, "IMF Special Drawing Rights", "SDR", 0 },
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn from_code_known_currencies() {
    assert_eq!(from_code("USD").unwrap().numeric, 840);
    assert_eq!(from_code("EUR").unwrap().numeric, 978);
    assert_eq!(from_code("JPY").unwrap().minor_unit, 0);
    assert!(from_code("XXX").is_none());
  }

  #[test]
  fn from_numeric_round_trip() {
    let usd = from_numeric(840).unwrap();
    assert_eq!(usd.code, "USD");
    let eur = from_numeric(978).unwrap();
    assert_eq!(eur.code, "EUR");
  }

  #[test]
  fn jpy_zero_minor_units() {
    assert_eq!(JPY.minor_unit, 0);
  }

  #[test]
  fn currency_display_uses_code() {
    assert_eq!(format!("{USD}"), "USD");
  }
}
