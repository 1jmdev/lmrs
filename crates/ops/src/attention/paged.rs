use std::ffi::c_void;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use kernels::{KernelDType, attention::{PagedAttentionParams, paged_attention_v1_raw}};
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

/// CUDA BF16 paged attention launch configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedAttentionConfig {
    /// Tokens stored in each KV-cache block.
    pub block_size: usize,
    /// Query-to-key scale, usually `1.0 / sqrt(head_dim)`.
    pub scale: f32,
    /// Whether to apply causal masking for multi-token queries.
    pub causal: bool,
}

/// Runs BF16 paged attention over a block-structured KV cache.
///
/// Shapes:
/// `q`: `[batch, heads, query_len, head_dim]`
/// `k_cache`, `v_cache`: `[blocks, block_size, kv_heads, head_dim]`
/// `block_tables`: `[batch, max_blocks_per_seq]` with I32 physical block ids
/// `context_lens`: `[batch]` with I32 valid key lengths
pub fn paged_attention(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    config: PagedAttentionConfig,
) -> Result<Tensor> {
    validate_dtype(q, DType::BF16)?;
    validate_dtype(k_cache, DType::BF16)?;
    validate_dtype(v_cache, DType::BF16)?;
    validate_dtype(block_tables, DType::I32)?;
    validate_dtype(context_lens, DType::I32)?;
    validate_contiguous(q)?;
    validate_contiguous(k_cache)?;
    validate_contiguous(v_cache)?;
    validate_contiguous(block_tables)?;
    validate_contiguous(context_lens)?;

    let q_dims = dims(q, 4, "q")?;
    let k_dims = dims(k_cache, 4, "k_cache")?;
    let v_dims = dims(v_cache, 4, "v_cache")?;
    let table_dims = dims(block_tables, 2, "block_tables")?;
    let lens_dims = dims(context_lens, 1, "context_lens")?;

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let query_len = q_dims[2];
    let head_dim = q_dims[3];
    let num_blocks = k_dims[0];
    let kv_block_size = k_dims[1];
    let num_kv_heads = k_dims[2];
    let max_blocks_per_seq = table_dims[1];

    if k_dims != v_dims {
        return Err(TensorError::ShapeMismatch(format!(
            "k_cache shape {:?} must match v_cache shape {:?}",
            k_dims, v_dims
        )));
    }
    if config.block_size != kv_block_size {
        return Err(TensorError::InvalidArgument(format!(
            "config block_size {} must match cache block size {kv_block_size}",
            config.block_size
        )));
    }
    if table_dims[0] != batch || lens_dims[0] != batch {
        return Err(TensorError::ShapeMismatch(format!(
            "metadata batch must match q batch {batch}, got block_tables {:?}, context_lens {:?}",
            table_dims, lens_dims
        )));
    }
    if num_blocks == 0 || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(TensorError::InvalidArgument(format!(
            "invalid heads/blocks: blocks={num_blocks}, heads={num_heads}, kv_heads={num_kv_heads}"
        )));
    }
    if config.block_size == 0 || query_len == 0 || head_dim == 0 {
        return Err(TensorError::InvalidArgument(format!(
            "invalid paged attention dimensions: block_size={}, query_len={query_len}, head_dim={head_dim}",
            config.block_size
        )));
    }

    let stream = q.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(q.len_bytes())? };
    let batch_i32 = to_i32(batch, "batch")?;
    let num_heads_i32 = to_i32(num_heads, "num_heads")?;
    let num_kv_heads_i32 = to_i32(num_kv_heads, "num_kv_heads")?;
    let head_dim_i32 = to_i32(head_dim, "head_dim")?;
    let block_size_i32 = to_i32(config.block_size, "block_size")?;
    let max_blocks_per_seq_i32 = to_i32(max_blocks_per_seq, "max_blocks_per_seq")?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (q_ptr, _q_sync) = q.storage().buffer().as_slice().device_ptr(stream);
    let (k_ptr, _k_sync) = k_cache.storage().buffer().as_slice().device_ptr(stream);
    let (v_ptr, _v_sync) = v_cache.storage().buffer().as_slice().device_ptr(stream);
    let (tables_ptr, _tables_sync) = block_tables.storage().buffer().as_slice().device_ptr(stream);
    let (lens_ptr, _lens_sync) = context_lens.storage().buffer().as_slice().device_ptr(stream);
    unsafe {
        paged_attention_v1_raw(
            out_ptr as *mut c_void,
            q_ptr as *const c_void,
            k_ptr as *const c_void,
            v_ptr as *const c_void,
            tables_ptr as *const i32,
            lens_ptr as *const i32,
            PagedAttentionParams {
                num_seqs: batch_i32,
                num_heads: num_heads_i32,
                num_kv_heads: num_kv_heads_i32,
                head_size: head_dim_i32,
                block_size: block_size_i32,
                max_num_blocks_per_seq: max_blocks_per_seq_i32,
                scale: config.scale,
                dtype: KernelDType::Bf16,
            },
            stream.cu_stream(),
        );
    }
    drop(out_sync);

    let shape = Shape::new(q_dims).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
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

fn dims(tensor: &Tensor, rank: usize, name: &str) -> Result<Vec<usize>> {
    let dims = tensor.shape().dims();
    if dims.len() != rank {
        return Err(TensorError::ShapeMismatch(format!(
            "{name} must have rank {rank}, got shape {:?}",
            dims
        )));
    }
    Ok(dims.to_vec())
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds i32::MAX")))
}
