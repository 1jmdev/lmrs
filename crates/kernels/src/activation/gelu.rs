use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

const GELU_BF16: &str = "gelu_bf16";
const GELU_F32: &str = "gelu_f32";

/// Describes the row-wise fused activation launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusedGeluMulLaunch {
    /// Number of packed rows in the input.
    pub rows: usize,
    /// Size of the post-activation row.
    pub intermediate_size: usize,
}

/// Applies GELU on CUDA using the fast tanh approximation.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    validate_contiguous(x)?;

    match x.dtype() {
        DType::BF16 => launch_bf16(x),
        DType::F32 => launch_f32(x),
        dtype => Err(TensorError::DTypeMismatch {
            expected: "bf16 or f32",
            actual: dtype.name(),
        }),
    }
}

impl FusedGeluMulLaunch {
    /// Creates launch metadata for a fused GELU multiply pass.
    pub fn new(rows: usize, intermediate_size: usize) -> Self {
        Self {
            rows,
            intermediate_size,
        }
    }
}

fn launch_bf16(x: &Tensor) -> Result<Tensor> {
    let mut out = alloc_output(x)?;
    let src = bf16_view(x)?;
    let mut dst = bf16_view_mut(&mut out, x.numel())?;
    launch(x, GELU_BF16, &src, &mut dst)?;
    Ok(output_tensor(x, out))
}

fn launch_f32(x: &Tensor) -> Result<Tensor> {
    let mut out = alloc_output(x)?;
    let src = f32_view(x)?;
    let mut dst = f32_view_mut(&mut out, x.numel())?;
    launch(x, GELU_F32, &src, &mut dst)?;
    Ok(output_tensor(x, out))
}

fn alloc_output(x: &Tensor) -> Result<cudarc::driver::CudaSlice<u8>> {
    Ok(unsafe {
        x.storage()
            .buffer()
            .as_slice()
            .stream()
            .alloc::<u8>(x.len_bytes())?
    })
}

fn output_tensor(x: &Tensor, out: cudarc::driver::CudaSlice<u8>) -> Tensor {
    let shape = x.shape().clone();
    let stride = Stride::contiguous(&shape);
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Tensor::from_storage(storage, shape, stride, x.dtype())
}

fn launch<T>(x: &Tensor, symbol: &str, src: &CudaView<'_, T>, dst: &mut CudaViewMut<'_, T>) -> Result<()> {
    let stream = x.storage().buffer().as_slice().stream();
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ACTIVATION_GELU))?;
    let func = module.load_function(symbol)?;
    let n = to_i32(x.numel(), "numel")?;
    let block = 256u32;
    let grid = ((x.numel() as u32).saturating_add(block - 1) / block).clamp(1, 65_535);
    let mut builder = stream.launch_builder(&func);
    builder.arg(src);
    builder.arg(dst);
    builder.arg(&n);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    Ok(())
}

fn validate_contiguous(tensor: &Tensor) -> Result<()> {
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

fn bf16_view_mut(
    out: &mut cudarc::driver::CudaSlice<u8>,
    numel: usize,
) -> Result<CudaViewMut<'_, u16>> {
    unsafe { out.transmute_mut::<u16>(numel) }.ok_or_else(|| {
        TensorError::InvalidArgument("failed to create mutable BF16 output view".to_string())
    })
}

fn f32_view(tensor: &Tensor) -> Result<CudaView<'_, f32>> {
    unsafe {
        tensor
            .storage()
            .buffer()
            .as_slice()
            .transmute::<f32>(tensor.numel())
    }
    .ok_or_else(|| TensorError::InvalidArgument("failed to create F32 tensor view".to_string()))
}

fn f32_view_mut(
    out: &mut cudarc::driver::CudaSlice<u8>,
    numel: usize,
) -> Result<CudaViewMut<'_, f32>> {
    unsafe { out.transmute_mut::<f32>(numel) }.ok_or_else(|| {
        TensorError::InvalidArgument("failed to create mutable F32 output view".to_string())
    })
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds i32::MAX")))
}
