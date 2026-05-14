use cudarc::driver::CudaView;
use cudarc::driver::CudaViewMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

/// CUDA module name for out-of-place rotary embedding.
pub const ROTARY_FWD_MODULE: &str = "lmrs_pos_embed_rotary_fwd";

/// CUDA module name for in-place rotary embedding.
pub const ROTARY_INPLACE_MODULE: &str = "lmrs_pos_embed_rotary_inplace";

/// Exported BF16 out-of-place rotary symbol.
pub const ROTARY_FWD_BF16: &str = "rotary_fwd_bf16";

/// Exported BF16 in-place rotary symbol.
pub const ROTARY_INPLACE_BF16: &str = "rotary_inplace_bf16";

/// Rotary launch metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotaryLaunch {
    /// Number of token rows.
    pub tokens: usize,
    /// Rotary dimension.
    pub dim: usize,
}

impl RotaryLaunch {
    /// Creates rotary launch metadata.
    pub fn new(tokens: usize, dim: usize) -> Self {
        Self { tokens, dim }
    }
}

pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    validate_bf16_contiguous(x)?;
    validate_bf16_contiguous(cos)?;
    validate_bf16_contiguous(sin)?;
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(TensorError::ShapeMismatch(format!(
            "rotary input must be rank 4, got rank {}",
            dims.len()
        )));
    }
    let tokens = dims[0] * dims[1] * dims[2];
    let dim = dims[3];
    let half = dim / 2;
    if dim % 2 != 0 {
        return Err(TensorError::ShapeMismatch(format!(
            "rotary dim must be even, got {dim}"
        )));
    }
    if cos.shape().dims() != [dims[2], half] || sin.shape().dims() != [dims[2], half] {
        return Err(TensorError::ShapeMismatch(format!(
            "rotary cos/sin must have shape [{}, {half}], got {:?} and {:?}",
            dims[2],
            cos.shape().dims(),
            sin.shape().dims()
        )));
    }
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let x_view = bf16_view(x)?;
    let cos_view = bf16_view(cos)?;
    let sin_view = bf16_view(sin)?;
    let mut out_view = bf16_mut_view(&mut out, x.numel(), "rotary output")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::POS_EMBED_ROTARY_FWD))?;
    let func = module.load_function(ROTARY_FWD_BF16)?;
    let tokens_i32 = to_i32(tokens, "tokens")?;
    let seq_len_i32 = to_i32(dims[2], "seq_len")?;
    let dim_i32 = to_i32(dim, "dim")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_view);
    builder.arg(&cos_view);
    builder.arg(&sin_view);
    builder.arg(&mut out_view);
    builder.arg(&tokens_i32);
    builder.arg(&seq_len_i32);
    builder.arg(&dim_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(x.numel().div_ceil(256), "grid_x")?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        Shape::new(dims.to_vec()).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?,
        Stride::contiguous(x.shape()),
        DType::BF16,
    ))
}

fn validate_bf16_contiguous(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::BF16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::BF16.name(),
            actual: tensor.dtype().name(),
        });
    }
    if !tensor.is_contiguous() {
        return Err(TensorError::NonContiguous);
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

fn bf16_mut_view<'a>(
    out: &'a mut cudarc::driver::CudaSlice<u8>,
    numel: usize,
    name: &str,
) -> Result<CudaViewMut<'a, u16>> {
    unsafe { out.transmute_mut::<u16>(numel) }.ok_or_else(|| {
        TensorError::InvalidArgument(format!("failed to create BF16 {name} view"))
    })
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds i32::MAX")))
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds u32::MAX")))
}
