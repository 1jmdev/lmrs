use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_root = manifest_dir.join("vendor").join("llama.cpp");
    let llama_header = llama_root.join("include").join("llama.h");
    let ggml_include = llama_root.join("ggml").join("include");

    println!("cargo:rerun-if-changed={}", llama_header.display());
    println!(
        "cargo:rerun-if-changed={}",
        llama_root.join("CMakeLists.txt").display()
    );

    let dst = cmake::Config::new(&llama_root)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BACKEND_DL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_WEBUI", "OFF")
        .profile("Release")
        .build();

    let build_dir = dst.join("build");
    let llama_lib = build_dir.join("src");
    let ggml_lib = build_dir.join("ggml").join("src");

    println!("cargo:rustc-link-search=native={}", llama_lib.display());
    println!("cargo:rustc-link-search=native={}", ggml_lib.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
        _ => {}
    }

    let bindings = bindgen::Builder::default()
        .header(llama_header.display().to_string())
        .clang_arg(format!("-I{}", llama_root.join("include").display()))
        .clang_arg(format!("-I{}", ggml_include.display()))
        .allowlist_type("(llama|ggml)_.*")
        .allowlist_function("(llama|ggml)_.*")
        .allowlist_var("(LLAMA|GGML)_.*")
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap();

    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("llama_bindings.rs"))
        .unwrap();
}
