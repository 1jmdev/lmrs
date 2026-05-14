use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

const MODULE_FUNCTION: &str = "qwen_fused_silu_mul_bf16";
const TWO_INPUT_FUNCTION: &str = "qwen_silu_mul_bf16";

/// Applies the Qwen-style fused SiLU-and-multiply MLP activation.
///
/// The input last dimension must be `intermediate_size * 2` and contain the
/// concatenated gate and up projections in contiguous BF16 CUDA storage.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    validate_dtype(gate_up, DType::BF16)?;
    validate_contiguous(gate_up)?;

    let dims = gate_up.shape().dims();
    let last = *dims.last().unwrap_or(&0);
    let expected_last = intermediate_size.checked_mul(2).ok_or_else(|| {
        TensorError::InvalidArgument("intermediate_size * 2 overflowed usize".to_string())
    })?;
    if last != expected_last {
        return Err(TensorError::ShapeMismatch(format!(
            "qwen_fused_silu_mul last dim {last} != {expected_last}"
        )));
    }

    let rows = gate_up.numel() / last;
    let out_el = rows.checked_mul(intermediate_size).ok_or_else(|| {
        TensorError::InvalidArgument("fused_silu_mul output elements overflowed usize".to_string())
    })?;
    let mut out = unsafe {
        gate_up
            .storage()
            .buffer()
            .as_slice()
            .stream()
            .alloc::<u8>(out_el * DType::BF16.size_in_bytes())?
    };

    let src = bf16_view(gate_up)?;
    let mut dst = unsafe { out.transmute_mut::<u16>(out_el) }.ok_or_else(|| {
        TensorError::InvalidArgument("failed to create BF16 output view".to_string())
    })?;

    let stream = gate_up.storage().buffer().as_slice().stream();
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ACTIVATION_FUSED_SILU_MUL))?;
    let func = module.load_function(MODULE_FUNCTION)?;
    let intermediate_size_i32 = to_i32(intermediate_size, "intermediate_size")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&src);
    builder.arg(&mut dst);
    builder.arg(&intermediate_size_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(rows, "rows")?, 1, 1),
            block_dim: (intermediate_size.min(1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let mut out_dims = dims.to_vec();
    if let Some(last_dim) = out_dims.last_mut() {
        *last_dim = intermediate_size;
    }
    let shape = Shape::new(out_dims).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stride = Stride::contiguous(&shape);
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(storage, shape, stride, DType::BF16))
}

/// Applies SiLU to `gate` and multiplies by `up` elementwise on CUDA.
///
/// This variant is used when gate and up projections are produced separately.
/// Inputs must be contiguous CUDA BF16 tensors with identical shapes.
///
/// # Example
///
/// ```no_run
/// use kernels::silu_mul;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([1, 4])?;
/// let gate = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 4])?;
/// let up = copy_h2d(&context, shape, DType::BF16, &[0u16; 4])?;
/// let out = silu_mul(&gate, &up)?;
/// assert_eq!(out.shape().dims(), &[1, 4]);
/// # Ok(())
/// # }
/// ```
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    validate_dtype(gate, DType::BF16)?;
    validate_dtype(up, DType::BF16)?;
    validate_contiguous(gate)?;
    validate_contiguous(up)?;
    if gate.shape().dims() != up.shape().dims() {
        return Err(TensorError::ShapeMismatch(format!(
            "silu_mul expected matching shapes, got {:?} and {:?}",
            gate.shape().dims(),
            up.shape().dims()
        )));
    }
    let stream = gate.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(gate.len_bytes())? };
    let gate_view = bf16_view(gate)?;
    let up_view = bf16_view(up)?;
    let mut out_view = bf16_mut_view(&mut out, gate.numel(), "silu_mul output")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ACTIVATION_FUSED_SILU_MUL))?;
    let func = module.load_function(TWO_INPUT_FUNCTION)?;
    let total_elements = to_i32(gate.numel(), "total_elements")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&gate_view);
    builder.arg(&up_view);
    builder.arg(&mut out_view);
    builder.arg(&total_elements);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(gate.numel().div_ceil(256), "grid_x")?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    let shape = gate.shape().clone();
    let stride = Stride::contiguous(&shape);
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(storage, shape, stride, DType::BF16))
}

fn validate_dtype(tensor: &Tensor, expected: DType) -> Result<()> {
    if tensor.dtype() != expected {
        return Err(TensorError::DTypeMismatch {
            expected: expected.name(),
            actual: tensor.dtype().name(),
        });
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
