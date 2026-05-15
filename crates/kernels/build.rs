use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUDA_FILES: &[&str] = &[
    "activation_ffi.cu",
    "attention_ffi.cu",
    "cast_ffi.cu",
    "norm_ffi.cu",
    "ops_ffi.cu",
    "pos_embed_ffi.cu",
];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.parent().unwrap().parent().unwrap();
    let kernels_dir = workspace.join("kernels");
    let ffi_dir = kernels_dir.join("ffi");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let obj_dir = out_dir.join("cuda-objects");
    fs::create_dir_all(&obj_dir).unwrap();

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    let nvcc = find_nvcc();
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_owned());
    let mut objects = Vec::with_capacity(CUDA_FILES.len());

    for file in CUDA_FILES {
        let src = ffi_dir.join(file);
        let obj = obj_dir.join(file.replace(".cu", ".o"));
        compile_cuda(&nvcc, &src, &obj, &kernels_dir, &arch);
        objects.push(obj);
    }

    let lib = obj_dir.join("libkernels_cuda_ffi.a");
    archive_objects(&lib, &objects);

    println!("cargo:rustc-link-search=native={}", obj_dir.display());
    println!("cargo:rustc-link-lib=static=kernels_cuda_ffi");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if let Some(cuda_home) = cuda_home() {
        println!("cargo:rustc-link-search=native={}", cuda_home.join("lib64").display());
    }
}

fn compile_cuda(nvcc: &Path, src: &Path, obj: &Path, kernels_dir: &Path, arch: &str) {
    let status = Command::new(nvcc)
        .arg("-std=c++17")
        .arg(format!("-arch={arch}"))
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-I")
        .arg(kernels_dir)
        .arg("-I")
        .arg(kernels_dir.join("include"))
        .arg("-c")
        .arg(src)
        .arg("-o")
        .arg(obj)
        .status()
        .unwrap_or_else(|err| panic!("failed to execute nvcc at {}: {err}", nvcc.display()));
    assert!(status.success(), "nvcc failed compiling {}", src.display());
}

fn archive_objects(lib: &Path, objects: &[PathBuf]) {
    if lib.exists() {
        fs::remove_file(lib).unwrap();
    }
    let build = cc::Build::new();
    let mut ar = build.get_archiver();
    let status = ar
        .arg("crs")
        .arg(lib)
        .args(objects)
        .status()
        .expect("failed to execute archiver");
    assert!(status.success(), "archiver failed creating {}", lib.display());
}

fn find_nvcc() -> PathBuf {
    if let Some(home) = cuda_home() {
        let candidate = home.join("bin").join("nvcc");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("nvcc")
}

fn cuda_home() -> Option<PathBuf> {
    env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .or_else(|| {
            let default = PathBuf::from("/usr/local/cuda");
            default.exists().then_some(default)
        })
}
