//! pegaflow-transfer build script.
//!
//! Generates FFI bindings for libibverbs, cuda driver, cuda runtime, and
//! gdrapi when the `v2` cargo feature is enabled. With `v2` off, this is a
//! complete no-op so a barebones dev box (no CUDA SDK, no rdma-core) can
//! still `cargo check` the v1 build.

use std::{
    env,
    path::{Path, PathBuf},
};

use bindgen::callbacks::{ItemInfo, ParseCallbacks};

fn find_package(
    provider: &str,
    env_var: &str,
    default_paths: &[&str],
    check_file: &str,
) -> PathBuf {
    println!("cargo:rerun-if-env-changed={}", env_var);
    env::var_os(env_var)
        .map(PathBuf::from)
        .into_iter()
        .chain(default_paths.iter().map(PathBuf::from))
        .find(|dir| dir.join(check_file).is_file())
        .unwrap_or_else(|| {
            panic!(
                "{provider}: required header `{check_file}` not found. \
                 Looked at `${env_var}` ({env_status}) and default paths {default_paths:?}. \
                 Hint: install the provider headers or set `{env_var}` to their install root.",
                env_status = env::var_os(env_var)
                    .map(|v| format!("set to {:?}", v))
                    .unwrap_or_else(|| "unset".to_string()),
            )
        })
}

fn build_libibverbs(out_dir: &Path, manifest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let home = find_package(
        "libibverbs",
        "LIBIBVERBS_HOME",
        &["/usr"],
        "include/infiniband/verbs.h",
    );

    let wrapper = manifest.join("build_wrappers/libibverbs.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}/include", home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(ibv_|IBV_|ib_|IB_).*")
        .derive_debug(false)
        .derive_default(true)
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_dir.join("wrap_static_fns.c"))
        .allowlist_item(r"pthread_.*")
        .opaque_type(r"pthread_.*")
        .no_default(r"pthread_.*")
        .generate()
        .map_err(|e| format!("libibverbs bindgen failed: {}", e))?;
    bindings.write_to_file(out_dir.join("libibverbs-bindings.rs"))?;

    cc::Build::new()
        .file(out_dir.join("wrap_static_fns.c"))
        .include(home.join("include"))
        .include(manifest.join("build_wrappers"))
        .compile("wrap_static_fns");

    println!("cargo:rustc-link-search=native={}/lib", home.display());
    println!("cargo:rustc-link-lib=ibverbs");
    Ok(())
}

fn build_cuda(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let home = find_package("cuda", "CUDA_HOME", &["/usr/local/cuda"], "include/cuda.h");
    let bindings = bindgen::Builder::default()
        .header(home.join("include/cuda.h").to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(cu|CU).*")
        .derive_default(true)
        .generate()
        .map_err(|e| format!("cuda driver bindgen failed: {}", e))?;
    bindings.write_to_file(out_dir.join("cuda-bindings.rs"))?;
    println!(
        "cargo:rustc-link-search=native={}/lib64/stubs",
        home.display()
    );
    println!("cargo:rustc-link-lib=cuda");
    Ok(())
}

#[derive(Debug)]
struct CudartRenameCallback;
impl ParseCallbacks for CudartRenameCallback {
    fn item_name(&self, item_info: ItemInfo<'_>) -> Option<String> {
        match item_info.name {
            // CUDA 12 defines cudaGetDeviceProperties as cudaGetDeviceProperties_v2.
            "cudaGetDeviceProperties_v2" => Some("cudaGetDeviceProperties".into()),
            _ => None,
        }
    }
}

fn build_cudart(out_dir: &Path, manifest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let home = find_package(
        "cudart",
        "CUDA_HOME",
        &["/usr/local/cuda"],
        "include/cuda.h",
    );
    let wrapper = manifest.join("build_wrappers/cudart.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}/include", home.display()))
        .parse_callbacks(Box::new(CudartRenameCallback))
        .prepend_enum_name(false)
        .allowlist_item(r"cuda.*")
        .derive_default(true)
        .generate()
        .map_err(|e| format!("cuda runtime bindgen failed: {}", e))?;
    bindings.write_to_file(out_dir.join("cudart-bindings.rs"))?;
    println!("cargo:rustc-link-search=native={}/lib64", home.display());
    println!("cargo:rustc-link-lib=cudart");
    Ok(())
}

fn build_gdrapi(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let home = find_package("gdrapi", "GDRAPI_HOME", &["/usr"], "include/gdrapi.h");
    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <gdrapi.h>")
        .clang_arg(format!("-I{}/include", home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"gdr.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .map_err(|e| format!("gdrapi bindgen failed: {}", e))?;
    bindings.write_to_file(out_dir.join("gdrapi-bindings.rs"))?;
    println!("cargo:rustc-link-lib=gdrapi");
    println!("cargo:rustc-link-search=native={}/lib", home.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // v2 feature off: do nothing, no probing, no link directives.
    if env::var_os("CARGO_FEATURE_V2").is_none() {
        return Ok(());
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=build_wrappers/libibverbs.h");
    println!("cargo:rerun-if-changed=build_wrappers/cudart.h");

    build_libibverbs(&out_dir, &manifest)?;
    build_cuda(&out_dir)?;
    build_cudart(&out_dir, &manifest)?;
    build_gdrapi(&out_dir)?;
    Ok(())
}
