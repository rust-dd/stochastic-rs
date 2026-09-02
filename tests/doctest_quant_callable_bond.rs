// docs: quant#callable-and-puttable-bonds-on-the-hullwhite-tree
//! Backs the callable bond example on the quant catalog page.

use stochastic_rs::quant::lattice::CallableBondSpec;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;
use stochastic_rs::quant::lattice::price_callable_bond;

#[test]
fn callable_and_puttable_bonds_bracket_the_straight_bond() {
  // Hull–White tree: r₀ = 4 %, a = 0.3, θ = 4 %, σ = 1 %; monthly steps to the 3-year maturity.
  let tree = HullWhiteTree::new(HullWhiteTreeModel::new(0.04, 0.3, 0.04, 0.01), 3.0, 36);
  // 6 % annual coupon, face 100, callable and puttable at par after years 1 and 2.
  let bond = CallableBondSpec::new(100.0, 0.06, vec![1.0, 2.0, 3.0]);
  let callable = bond.clone().with_calls(vec![(1.0, 100.0), (2.0, 100.0)]);
  let puttable = bond.clone().with_puts(vec![(1.0, 100.0), (2.0, 100.0)]);

  let straight = price_callable_bond(&tree.tree, &tree.model, &bond);
  let called = price_callable_bond(&tree.tree, &tree.model, &callable);
  let put = price_callable_bond(&tree.tree, &tree.model, &puttable);

  // The issuer's call lowers the price, the holder's put raises it.
  assert!(called.price < straight.price && put.price > straight.price);
  assert!(called.call_value > 0.0 && put.put_value > 0.0);
  assert_eq!(straight.price, straight.straight_price);
}
