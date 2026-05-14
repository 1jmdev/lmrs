use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
use candle_core::{DType, Device, Result, Tensor};

use crate::ptx;

const MODULE_NAME: &str = "lmrs_utils_fill";

/// Computes the argmax token id for a BF16 CUDA logits vector.
pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    if logits.dtype() != DType::BF16 {
        candle_core::bail!("gpu_argmax is BF16-only");
    }
    let dev = match logits.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax requires CUDA"),
    };
    let logits = logits.contiguous()?.flatten_all()?;
    let vocab_size = logits.elem_count();
    let (storage, layout) = logits.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("gpu_argmax requires CUDA storage"),
    };
    let (o1, _o2) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("gpu_argmax requires contiguous logits".into()))?;

    let out = unsafe { dev.alloc::<i32>(1)? };
    let func = dev.get_or_load_custom_func("generic_argmax_bf16", MODULE_NAME, ptx::UTILS_FILL)?;
    let vocab_size_i32 = vocab_size as i32;
    match &storage.slice {
        CudaStorageSlice::BF16(s) => {
            let src = s.slice(o1..);
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&out);
            builder.arg(&vocab_size_i32);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }
        _ => candle_core::bail!("gpu_argmax is BF16-only"),
    }
    let token = dev.clone_dtoh(&out)?;
    Ok(token[0] as u32)
}
