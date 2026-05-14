use cudarc::driver::CudaView;
use cudarc::driver::CudaViewMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

const REPEAT_KV_BF16: &str = "repeat_kv_bf16";
const SDPA_BF16: &str = "sdpa_bf16";

/// Repeats grouped-query key/value heads into full attention heads.
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    validate_bf16_contiguous(x)?;
    if n_rep == 0 {
        return Err(TensorError::InvalidArgument(
            "repeat_kv n_rep must be greater than zero".to_string(),
        ));
    }
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let dims = dims4(x, "repeat_kv input")?;
    let out_shape = Shape::new([dims[0], dims[1] * n_rep, dims[2], dims[3]])
        .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let out_numel = out_shape.numel();
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(out_numel * DType::BF16.size_in_bytes())? };
    let src = bf16_view(x)?;
    let mut dst = bf16_mut_view(&mut out, out_numel, "repeat_kv output")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ATTENTION_SDPA))?;
    let func = module.load_function(REPEAT_KV_BF16)?;
    let total_elements = to_i32(out_numel, "total_elements")?;
    let kv_heads = to_i32(dims[1], "kv_heads")?;
    let n_rep_i32 = to_i32(n_rep, "n_rep")?;
    let seq_len = to_i32(dims[2], "seq_len")?;
    let head_dim = to_i32(dims[3], "head_dim")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&src);
    builder.arg(&mut dst);
    builder.arg(&total_elements);
    builder.arg(&kv_heads);
    builder.arg(&n_rep_i32);
    builder.arg(&seq_len);
    builder.arg(&head_dim);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(out_numel.div_ceil(256), "grid_x")?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        out_shape.clone(),
        Stride::contiguous(&out_shape),
        DType::BF16,
    ))
}

/// Runs BF16 scaled dot-product attention for contiguous `[batch, heads, seq, dim]` tensors.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    causal: bool,
    start_pos: usize,
) -> Result<Tensor> {
    validate_bf16_contiguous(q)?;
    validate_bf16_contiguous(k)?;
    validate_bf16_contiguous(v)?;
    let q_dims = dims4(q, "q")?;
    let k_dims = dims4(k, "k")?;
    let v_dims = dims4(v, "v")?;
    if q_dims[0] != k_dims[0]
        || q_dims[0] != v_dims[0]
        || q_dims[1] != k_dims[1]
        || q_dims[1] != v_dims[1]
        || k_dims[2] != v_dims[2]
        || q_dims[3] != head_dim
        || k_dims[3] != head_dim
        || v_dims[3] != head_dim
    {
        return Err(TensorError::ShapeMismatch(format!(
            "sdpa expected q/k/v [batch, heads, seq, {head_dim}], got {:?}, {:?}, {:?}",
            q_dims, k_dims, v_dims
        )));
    }

    let out_shape = Shape::new(q_dims).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = q.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(q.len_bytes())? };
    let q_view = bf16_view(q)?;
    let k_view = bf16_view(k)?;
    let v_view = bf16_view(v)?;
    let mut out_view = bf16_mut_view(&mut out, q.numel(), "sdpa output")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(ptx::ATTENTION_SDPA))?;
    let func = module.load_function(SDPA_BF16)?;
    let batch = to_i32(q_dims[0], "batch")?;
    let heads = to_i32(q_dims[1], "heads")?;
    let query_len = to_i32(q_dims[2], "query_len")?;
    let key_len = to_i32(k_dims[2], "key_len")?;
    let head_dim_i32 = to_i32(head_dim, "head_dim")?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal = i32::from(causal);
    let start_pos = to_i32(start_pos, "start_pos")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_view);
    builder.arg(&k_view);
    builder.arg(&v_view);
    builder.arg(&mut out_view);
    builder.arg(&batch);
    builder.arg(&heads);
    builder.arg(&query_len);
    builder.arg(&key_len);
    builder.arg(&head_dim_i32);
    builder.arg(&scale);
    builder.arg(&causal);
    builder.arg(&start_pos);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                to_u32(q_dims[0], "grid_x")?,
                to_u32(q_dims[1], "grid_y")?,
                to_u32(q_dims[2], "grid_z")?,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }

    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        out_shape.clone(),
        Stride::contiguous(&out_shape),
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

fn dims4(tensor: &Tensor, name: &str) -> Result<[usize; 4]> {
    let dims = tensor.shape().dims();
    if dims.len() != 4 {
        return Err(TensorError::ShapeMismatch(format!(
            "{name} must be rank 4, got rank {}",
            dims.len()
        )));
    }
    Ok([dims[0], dims[1], dims[2], dims[3]])
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
