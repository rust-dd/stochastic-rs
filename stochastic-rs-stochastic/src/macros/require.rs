/// A parameter precondition in the crate's one message, ``x must satisfy `pred`, got x = value``.
macro_rules! require {
  ($pred:expr, $name:ident) => {
    assert!(
      $pred,
      concat!(
        stringify!($name),
        " must satisfy `",
        stringify!($pred),
        "`, got ",
        stringify!($name),
        " = {:?}"
      ),
      $name
    )
  };
}
