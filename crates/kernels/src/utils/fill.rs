use cudarc::driver::CudaView;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{DType, Result, Tensor, TensorError};

use crate::ptx;

const MODULE_FUNCTION: &str = "generic_argmax_bf16";

/// Computes the argmax token id for a contiguous BF16 CUDA logits vector.
pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    if logits.dtype() != DType::BF16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::BF16.name(),
            actual: logits.dtype().name(),
        });
    }
    if !logits.is_contiguous() {
        return Err(TensorError::NonContiguous);
    }

    let vocab_size = logits.numel();
    if vocab_size == 0 {
        return Err(TensorError::InvalidArgument(
            "gpu_argmax requires a non-empty logits tensor".to_string(),
        ));
    }

    let stream = logits.storage().buffer().as_slice().stream();
    let out = unsafe { stream.alloc::<i32>(1)? };
    let src = bf16_view(logits)?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::UTILS_FILL))?;
    let func = module.load_function(MODULE_FUNCTION)?;
    let vocab_size_i32 = to_i32(vocab_size, "vocab_size")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&src);
    builder.arg(&out);
    builder.arg(&vocab_size_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let token = stream.clone_dtoh(&out)?;
    Ok(token[0] as u32)
}

fn bf16_view(tensor: &Tensor) -> Result<CudaView<'_, u16>> {
    unsafe {
        tensor
            .storage()
            .buffer()
            .as_slice()
            .transmute::<u16>(tensor.numel())
    }
    .ok_or_else(|| TensorError::InvalidArgument("failed to create BF16 tensor view".to_string()))
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds i32::MAX")))
}
