use cudarc::driver::CudaView;
use cudarc::driver::CudaViewMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

/// CUDA module name for LayerNorm forward.
pub const LAYER_NORM_MODULE: &str = "lmrs_norm_layer";

/// Exported BF16 LayerNorm forward symbol.
pub const LAYER_NORM_FWD_BF16: &str = "layer_norm_fwd_bf16";

/// LayerNorm launch metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerNormLaunch {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns per row.
    pub cols: usize,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl LayerNormLaunch {
    /// Creates LayerNorm launch metadata.
    pub fn new(rows: usize, cols: usize, eps: f32) -> Self {
        Self { rows, cols, eps }
    }
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    validate_bf16_contiguous(x)?;
    validate_bf16_contiguous(weight)?;
    validate_bf16_contiguous(bias)?;
    let cols = x.shape().dims().last().copied().ok_or_else(|| {
        TensorError::ShapeMismatch("layer_norm input must have at least one dimension".to_string())
    })?;
    if weight.shape().dims() != [cols] || bias.shape().dims() != [cols] {
        return Err(TensorError::ShapeMismatch(format!(
            "layer_norm weight and bias must have shape [{cols}], got {:?} and {:?}",
            weight.shape().dims(),
            bias.shape().dims()
        )));
    }
    let rows = x.numel() / cols;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let x_view = bf16_view(x)?;
    let weight_view = bf16_view(weight)?;
    let bias_view = bf16_view(bias)?;
    let mut out_view = bf16_mut_view(&mut out, x.numel(), "layer_norm output")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::NORM_LAYER_NORM))?;
    let func = module.load_function(LAYER_NORM_FWD_BF16)?;
    let rows_i32 = to_i32(rows, "rows")?;
    let cols_i32 = to_i32(cols, "cols")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_view);
    builder.arg(&weight_view);
    builder.arg(&bias_view);
    builder.arg(&mut out_view);
    builder.arg(&rows_i32);
    builder.arg(&cols_i32);
    builder.arg(&eps);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(rows, "grid_x")?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        x.shape().clone(),
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
