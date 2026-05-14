use cudarc::driver::CudaView;
use cudarc::driver::CudaViewMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

/// CUDA module name for RMSNorm forward.
pub const RMS_NORM_FWD_MODULE: &str = "lmrs_norm_rms_fwd";

/// CUDA module name for RMSNorm backward.
pub const RMS_NORM_BWD_MODULE: &str = "lmrs_norm_rms_bwd";

/// Exported BF16 RMSNorm forward symbol.
pub const RMS_NORM_FWD_BF16: &str = "rms_norm_fwd_bf16";

/// Exported BF16 RMSNorm backward symbol.
pub const RMS_NORM_BWD_BF16: &str = "rms_norm_bwd_bf16";

/// Row-wise normalization shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormShape {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns per row.
    pub cols: usize,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl NormShape {
    /// Creates a normalization shape descriptor.
    pub fn new(rows: usize, cols: usize, eps: f32) -> Self {
        Self { rows, cols, eps }
    }
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    validate_bf16_contiguous(x)?;
    validate_bf16_contiguous(weight)?;
    let cols = x.shape().dims().last().copied().ok_or_else(|| {
        TensorError::ShapeMismatch("rms_norm input must have at least one dimension".to_string())
    })?;
    if weight.shape().dims() != [cols] {
        return Err(TensorError::ShapeMismatch(format!(
            "rms_norm weight must have shape [{cols}], got {:?}",
            weight.shape().dims()
        )));
    }
    let rows = x.numel() / cols;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let x_view = bf16_view(x)?;
    let weight_view = bf16_view(weight)?;
    let mut out_view = bf16_mut_view(&mut out, x.numel(), "rms_norm output")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::NORM_RMS_NORM_FWD))?;
    let func = module.load_function(RMS_NORM_FWD_BF16)?;
    let rows_i32 = to_i32(rows, "rows")?;
    let cols_i32 = to_i32(cols, "cols")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_view);
    builder.arg(&weight_view);
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
