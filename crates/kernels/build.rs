use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
struct CudaSource {
    source: PathBuf,
    ptx: String,
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=../../kernels");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    fs::create_dir_all(&out_dir).expect("failed to create build output directory");

    let kernels_dir = manifest_dir.join("../../kernels");
    let include_dir = kernels_dir.join("include");
    let cuda_include_dir = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .map(|path| PathBuf::from(path).join("include"))
        .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda/include"));

    let sources = cuda_sources(&kernels_dir);
    for source in &sources {
        println!("cargo::rerun-if-changed={}", source.source.display());
        let ptx = out_dir.join(format!("{}.ptx", source.ptx));
        compile_kernels(
            &source.source,
            &ptx,
            &[
                include_dir.clone(),
                cuda_include_dir.clone(),
            ],
        );
    }
}

fn cuda_sources(kernels_dir: &Path) -> Vec<CudaSource> {
    let mut files = Vec::new();
    collect_cuda_sources(kernels_dir, &mut files);
    files.sort();

    files
        .into_iter()
        .map(|source| {
            let relative = source
                .strip_prefix(kernels_dir)
                .expect("CUDA source is under kernels directory");
            let mut components = relative.components();
            let category = components
                .next()
                .expect("CUDA source has category directory")
                .as_os_str()
                .to_string_lossy()
                .into_owned();
            let stem = source
                .file_stem()
                .expect("CUDA source has file stem")
                .to_string_lossy();
            let ptx = format!("{}_{}", rust_name(&category), rust_name(&stem));

            CudaSource {
                source,
                ptx,
            }
        })
        .collect()
}

fn collect_cuda_sources(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("failed to read kernels directory") {
        let path = entry.expect("failed to read kernels directory entry").path();
        if path.is_dir() {
            collect_cuda_sources(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "cu") {
            files.push(path);
        }
    }
}

fn rust_name(name: &str) -> String {
    let mut out = String::new();
    let mut previous_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_underscore = false;
        } else if !previous_underscore {
            out.push('_');
            previous_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn compile_kernels(source: &Path, output: &Path, include_dirs: &[PathBuf]) {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut command = Command::new(&nvcc);
    command
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--gpu-architecture=compute_89")
        .arg("--output-file")
        .arg(output);

    for include_dir in include_dirs {
        command.arg(format!("--include-path={}", include_dir.display()));
    }

    let output_result = command
        .arg(source)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {nvcc}: {error}"));

    if !output_result.status.success() {
        panic!(
            "failed to compile CUDA source with {nvcc}:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output_result.stdout),
            String::from_utf8_lossy(&output_result.stderr),
        );
    }
}
