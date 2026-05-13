mod generic;
mod qwen;

pub use generic::{causal_mask, gpu_argmax};
pub use qwen::fused_silu_mul;

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/lmrs_kernels_ptx.rs"));
}

const GENERIC_MODULE_NAME: &str = "lmrs_generic_kernels";
const QWEN_MODULE_NAME: &str = "lmrs_qwen_kernels";
