#[cfg(all(feature = "metal", target_os = "macos"))]
mod gallery;

fn main() {
  #[cfg(all(feature = "metal", target_os = "macos"))]
  gallery::main();
}
