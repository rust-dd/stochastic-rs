use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=src/stochastic/cuda/fgn_exports.cu");
  println!("cargo:rerun-if-changed=src/stochastic/cuda/fgn_common.cuh");
  println!("cargo:rerun-if-changed=src/stochastic/cuda/fgn_f32.cuh");
  println!("cargo:rerun-if-changed=src/stochastic/cuda/fgn_f64.cuh");
  println!("cargo:rerun-if-env-changed=CUDA_ARCH");
  println!("cargo:rerun-if-env-changed=CUDA_NVCC");
  println!("cargo:rerun-if-env-changed=STOCHASTIC_RS_SKIP_CUDA_BUILD");

  if env::var_os("CARGO_FEATURE_CUDA").is_none() {
    return;
  }

  if env::var_os("STOCHASTIC_RS_SKIP_CUDA_BUILD").is_some() {
    println!("cargo:warning=Skipping CUDA build because STOCHASTIC_RS_SKIP_CUDA_BUILD is set");
    return;
  }

  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  if target_os != "linux" && target_os != "windows" {
    println!("cargo:warning=Automatic CUDA build is only supported on Linux/Windows");
    return;
  }

  let nvcc = env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
  let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());
  let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
  let source = PathBuf::from("src/stochastic/cuda/fgn_exports.cu");
  let output = if target_os == "windows" {
    out_dir.join("fgn_cuda.dll")
  } else {
    out_dir.join("libfgn_cuda.so")
  };

  let mut cmd = Command::new(&nvcc);
  cmd
    .arg("-O3")
    .arg("-use_fast_math")
    .arg(format!("-arch={arch}"))
    .arg("-shared")
    .arg(source)
    .arg("-o")
    .arg(&output)
    .arg("-lcufft")
    .arg("-lcurand");
  if target_os == "linux" {
    cmd.arg("-Xcompiler").arg("-fPIC");
  }

  let result = cmd.output();
  match result {
    Ok(out) if out.status.success() => {
      println!(
        "cargo:rustc-env=STOCHASTIC_RS_CUDA_FGN_LIB={}",
        output.display()
      );
      println!(
        "cargo:warning=Built CUDA FGN library at {}",
        output.display()
      );
    }
    Ok(out) => {
      let stderr = String::from_utf8_lossy(&out.stderr);
      println!(
        "cargo:warning=CUDA build failed with status: {}",
        out.status
      );
      println!("cargo:warning=nvcc stderr: {}", stderr.trim());
      println!("cargo:warning=Falling back to runtime/manual CUDA library loading");
    }
    Err(err) => {
      println!(
        "cargo:warning=Failed to run nvcc ({nvcc}): {err}. Falling back to runtime/manual CUDA library loading"
      );
    }
  }
}
