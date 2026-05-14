use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use runtime::CudaContext;
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

const MODULE_FUNCTION: &str = "generic_causal_mask_bf16";

/// Builds a BF16 causal attention mask on the selected CUDA device.
///
/// The result shape is `[1, 1, seq_len, total_len]` so it broadcasts over
/// batch and head dimensions before softmax.
pub fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    context: &CudaContext,
) -> Result<Tensor> {
    if seq_len == 0 || total_len == 0 {
        return Err(TensorError::InvalidArgument(format!(
            "causal_mask dimensions must be non-zero, got seq_len={seq_len}, total_len={total_len}"
        )));
    }

    let shape = Shape::new([1, 1, seq_len, total_len])
        .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let elem_count = shape.numel();
    let mut out = unsafe {
        context
            .cudarc()
            .default_stream()
            .alloc::<u8>(elem_count * DType::BF16.size_in_bytes())?
    };
    let mut out_view = unsafe { out.transmute_mut::<u16>(elem_count) }.ok_or_else(|| {
        TensorError::InvalidArgument("failed to create BF16 causal mask view".to_string())
    })?;

    let stream = context.cudarc().default_stream();
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ATTENTION_PAGED_ATTN_FWD))?;
    let func = module.load_function(MODULE_FUNCTION)?;
    let seq_len_i32 = to_i32(seq_len, "seq_len")?;
    let total_len_i32 = to_i32(total_len, "total_len")?;
    let start_pos_i32 = to_i32(start_pos, "start_pos")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut out_view);
    builder.arg(&seq_len_i32);
    builder.arg(&total_len_i32);
    builder.arg(&start_pos_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                to_u32(total_len.div_ceil(256), "grid_x")?,
                to_u32(seq_len, "grid_y")?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let stride = Stride::contiguous(&shape);
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(storage, shape, stride, DType::BF16))
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds i32::MAX")))
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds u32::MAX")))
}
