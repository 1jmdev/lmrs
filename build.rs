use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn which_cmd(cmd: &str) -> Option<String> {
    Command::new("which")
        .arg(cmd)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|stdout| stdout.trim().to_string())
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_root = manifest_dir.join("vendor").join("llama.cpp");
    let llama_include = llama_root.join("include");
    let ggml_include = llama_root.join("ggml").join("include");
    let llama_header = llama_include.join("llama.h");

    println!("cargo:rerun-if-changed={}", llama_header.display());
    println!(
        "cargo:rerun-if-changed={}",
        llama_root.join("CMakeLists.txt").display()
    );

    let mut cmake_config = cmake::Config::new(&llama_root);
    cmake_config
        .generator("Ninja")
        .define("CMAKE_CUDA_ARCHITECTURES", "native")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("GGML_BACKEND_DL", "ON")
        .define("GGML_CPU_ALL_VARIANTS", "ON")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_LTO", "ON")
        .define("GGML_OPENMP", "ON")
        .define("GGML_CPU_REPACK", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_WEBUI", "OFF")
        .profile("Release");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let nvcc = env::var("CUDACXX").ok().or_else(|| which_cmd("nvcc"));

    let cuda_enabled = if let Some(ref nvcc_path) = nvcc {
        if Path::new(nvcc_path).exists() {
            cmake_config.define("GGML_CUDA", "ON");
            cmake_config.define("CMAKE_CUDA_COMPILER", nvcc_path);
            true
        } else {
            false
        }
    } else {
        false
    };

    if !cuda_enabled {
        match target_os.as_str() {
            "macos" => {
                cmake_config.define("GGML_METAL", "ON");
            }
            "linux" => {
                let hipcc = env::var("HIPCC").ok().or_else(|| which_cmd("hipcc"));
                let rocm_enabled = if let Some(ref hipcc_path) = hipcc {
                    if Path::new(hipcc_path).exists() {
                        let hipcc_path = Path::new(hipcc_path);
                        let derived_clang = hipcc_path.parent().and_then(|bin| {
                            ["../llvm/bin/clang", "../lib/llvm/bin/clang"]
                                .iter()
                                .find_map(|relative| {
                                    let candidate = bin.join(relative);
                                    candidate
                                        .canonicalize()
                                        .ok()
                                        .filter(|path| path.exists())
                                        .map(|path| path.to_string_lossy().into_owned())
                                })
                        });
                        let rocm_clang = derived_clang
                            .or_else(|| {
                                let path = Path::new("/opt/rocm/lib/llvm/bin/clang");
                                path.exists().then(|| path.to_string_lossy().into_owned())
                            })
                            .or_else(|| {
                                let path = Path::new("/opt/rocm/llvm/bin/clang");
                                path.exists().then(|| path.to_string_lossy().into_owned())
                            })
                            .or_else(|| which_cmd("amdclang"));

                        if let Some(clang_path) = rocm_clang {
                            cmake_config.define("GGML_HIP", "ON");
                            cmake_config.define("CMAKE_HIP_COMPILER", clang_path);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !rocm_enabled {
                    let has_vulkan = env::var("VULKAN_SDK").is_ok()
                        || which_cmd("glslc").is_some()
                        || Path::new("/usr/include/vulkan/vulkan.h").exists();
                    if has_vulkan {
                        cmake_config.define("GGML_VULKAN", "ON");
                    }
                }
            }
            "windows" => {
                cmake_config.generator("Ninja");
                if env::var("VULKAN_SDK").is_ok() {
                    cmake_config.define("GGML_VULKAN", "ON");
                }
            }
            _ => {}
        }
    }

    let dst = cmake_config.build();
    let build_dir = dst.join("build");
    let llama_lib = build_dir.join("src");
    let ggml_src = build_dir.join("ggml").join("src");
    let bin_out = build_dir.join("bin");

    println!("cargo:rustc-link-search=native={}", bin_out.display());
    println!("cargo:rustc-link-search=native={}", llama_lib.display());
    println!("cargo:rustc-link-search=native={}", ggml_src.display());

    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bin_dest = out_dir
        .parent()
        .and_then(|path| path.parent())
        .and_then(|path| path.parent())
        .map(|path| path.to_path_buf());

    if let Some(destination) = bin_dest {
        let shared_ext = match target_os.as_str() {
            "macos" => "dylib",
            "windows" => "dll",
            _ => "so",
        };
        for search_dir in [&llama_lib, &ggml_src, &bin_out] {
            if let Ok(entries) = std::fs::read_dir(search_dir) {
                for entry in entries.filter_map(Result::ok) {
                    let path = entry.path();
                    let file_name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("");
                    let is_runtime_lib = file_name.contains(&format!(".{shared_ext}"))
                        && (file_name.starts_with("libggml") || file_name.starts_with("libllama"));
                    if is_runtime_lib {
                        let _ = std::fs::copy(&path, destination.join(path.file_name().unwrap()));
                    }
                }
            }
        }
    }

    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=dylib=m");
        }
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-lib=dylib=c++");
            if nvcc.is_none() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    ggml_src.join("ggml-metal").display()
                );
            }
        }
        _ => {}
    }

    let ggml_build_include = build_dir.join("ggml").join("include");
    let mut include_paths = vec![llama_include, ggml_include];
    if ggml_build_include.exists() {
        include_paths.push(ggml_build_include);
    }

    let clang_args: Vec<String> = include_paths
        .iter()
        .flat_map(|path| [format!("-I{}", path.display())])
        .collect();

    let bindings = bindgen::Builder::default()
        .header(llama_header.display().to_string())
        .clang_args(clang_args)
        .allowlist_function("llama_.*")
        .allowlist_type("(llama|ggml)_.*")
        .allowlist_var("(LLAMA|GGML)_.*")
        .generate_comments(false)
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap();

    bindings
        .write_to_file(out_dir.join("llama_bindings.rs"))
        .unwrap();
}
