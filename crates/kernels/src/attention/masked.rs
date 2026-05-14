use cudarc::driver::CudaView;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

const APPLY_CAUSAL_MASK_BF16: &str = "apply_causal_mask_bf16";

/// Applies a causal mask directly to a contiguous BF16 scores tensor.
pub fn apply_causal_mask(
    scores: &Tensor,
    query_len: usize,
    key_len: usize,
    start_pos: usize,
) -> Result<Tensor> {
    validate_bf16(scores)?;
    if !scores.is_contiguous() {
        return Err(TensorError::NonContiguous);
    }
    if query_len == 0 || key_len == 0 {
        return Err(TensorError::InvalidArgument(format!(
            "causal mask dimensions must be non-zero, got query_len={query_len}, key_len={key_len}"
        )));
    }
    if scores.numel() % (query_len * key_len) != 0 {
        return Err(TensorError::ShapeMismatch(format!(
            "scores numel {} is not divisible by query_len * key_len {}",
            scores.numel(),
            query_len * key_len
        )));
    }

    let stream = scores.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(scores.len_bytes())? };
    let src = bf16_view(scores)?;
    let mut dst = unsafe { out.transmute_mut::<u16>(scores.numel()) }.ok_or_else(|| {
        TensorError::InvalidArgument("failed to create BF16 causal mask output view".to_string())
    })?;

    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ATTENTION_MASKED))?;
    let func = module.load_function(APPLY_CAUSAL_MASK_BF16)?;
    let total_elements = to_i32(scores.numel(), "total_elements")?;
    let query_len = to_i32(query_len, "query_len")?;
    let key_len = to_i32(key_len, "key_len")?;
    let start_pos = to_i32(start_pos, "start_pos")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&src);
    builder.arg(&mut dst);
    builder.arg(&total_elements);
    builder.arg(&query_len);
    builder.arg(&key_len);
    builder.arg(&start_pos);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(scores.numel().div_ceil(256), "grid_x")?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        scores.shape().clone(),
        Stride::contiguous(scores.shape()),
        DType::BF16,
    ))
}

fn validate_bf16(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::BF16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::BF16.name(),
            actual: tensor.dtype().name(),
        });
    }
    Ok(())
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

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds u32::MAX")))
}
