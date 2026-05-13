fn main() {
    println!("cargo::rerun-if-changed=kernels/");
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        use std::{env, path::PathBuf};

        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
        let bindings = bindgen_cuda::Builder::default()
            .kernel_paths_glob("kernels/**/*.cu")
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3")
            .build_ptx()
            .expect("failed to compile CUDA kernels");

        bindings
            .write(out_dir.join("lmrs_kernels_ptx.rs"))
            .expect("failed to write PTX bindings");
    }
}
