use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

use crate::{BinaryOp, KernelDType, UnaryOp, activation, norm, ops, pos_embed};

unsafe extern "C" {
    fn fused_silu_split_fwd(out:*mut c_void, gate_up:*const c_void, rows:i64, inner:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn affine_fwd(out:*mut c_void, x:*const c_void, n:i64, scale:f32, offset:f32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn ge_fwd(out:*mut c_void, a:*const c_void, b:*const c_void, n:i64, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn where_fwd(out:*mut c_void, c:*const c_void, t:*const c_void, f:*const c_void, n:i64, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn add_bias_fwd(out:*mut c_void, x:*const c_void, bias:*const c_void, n:i64, last:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn narrow_dim1_fwd(out:*mut c_void, inp:*const c_void, d0:i32, d1:i32, d2:i32, start:i32, len:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn transpose12_fwd(out:*mut c_void, inp:*const c_void, b:i32, d1:i32, d2:i32, d3:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn concat_dim2_fwd(out:*mut c_void, a:*const c_void, b:*const c_void, d0:i32, d1:i32, a2:i32, b2:i32, d3:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn repeat_kv_fwd(out:*mut c_void, x:*const c_void, b:i32, kv:i32, seq:i32, d:i32, rep:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn linear_fwd(out:*mut c_void, x:*const c_void, w:*const c_void, bias:*const c_void, rows:i32, in_f:i32, out_f:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn embedding_fwd(out:*mut c_void, ids:*const i32, emb:*const c_void, tokens:i32, hidden:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn causal_mask_fwd(out:*mut c_void, scores:*const c_void, n:i64, q_len:i32, k_len:i32, start_pos:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
    fn sdpa_fwd(out:*mut c_void, q:*const c_void, k:*const c_void, v:*const c_void, b:i32, h:i32, q_len:i32, k_len:i32, d:i32, scale:f32, causal:i32, start_pos:i32, dtype:i32, stream:cudarc::driver::sys::CUstream);
}

pub fn add(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    binary(left, right, BinaryOp::Add)
}

pub fn mul_bf16(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    binary(left, right, BinaryOp::Mul)
}

pub fn sub_bf16(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    binary(left, right, BinaryOp::Sub)
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    unary(x, UnaryOp::Gelu)
}

pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    same_shape(gate, up, "silu_mul")?;
    let stream = gate.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(gate.len_bytes())? };
    let dtype = kernel_dtype(gate.dtype())?;
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (gate_ptr, _gate_sync) = gate.storage().buffer().as_slice().device_ptr(stream);
    let (up_ptr, _up_sync) = up.storage().buffer().as_slice().device_ptr(stream);
    unsafe {
        activation::silu_mul_raw(
            out_ptr as *mut c_void,
            gate_ptr as *const c_void,
            up_ptr as *const c_void,
            gate.numel() as i64,
            dtype,
            stream.cu_stream(),
        )
    };
    drop(_out_sync);
    Ok(tensor_from_slice(out, gate))
}

pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    let dims = gate_up.shape().dims();
    let Some(&last) = dims.last() else {
        return Err(TensorError::ShapeMismatch("fused_silu_mul requires rank >= 1".to_string()));
    };
    if last != intermediate_size * 2 {
        return Err(TensorError::ShapeMismatch(format!(
            "fused_silu_mul last dim must be {}, got {last}",
            intermediate_size * 2
        )));
    }
    let rows = gate_up.numel() / (intermediate_size * 2);
    let out_shape = Shape::new([rows, intermediate_size]).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = gate_up.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(rows * intermediate_size * gate_up.dtype().size_in_bytes())? };
    let dtype = kernel_dtype(gate_up.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(&stream);
    let (in_ptr, _in_sync) = gate_up.storage().buffer().as_slice().device_ptr(stream);
    unsafe { fused_silu_split_fwd(out_ptr as *mut c_void, in_ptr as *const c_void, rows as i64, intermediate_size as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(Tensor::from_storage(SharedStorage::new(CudaBuf::from_slice(out)), out_shape.clone(), Stride::contiguous(&out_shape), gate_up.dtype()))
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    norm_common(x, weight, None, eps)
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    norm_common(x, weight, Some(bias), eps)
}

pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    validate_cuda_input(x, "apply_rotary")?;
    validate_cuda_input(cos, "apply_rotary")?;
    validate_cuda_input(sin, "apply_rotary")?;
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(TensorError::ShapeMismatch(format!("apply_rotary expects rank 4, got {:?}", dims)));
    }
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let dtype = kernel_dtype(x.dtype())?;
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    let (cos_ptr, _cos_sync) = cos.storage().buffer().as_slice().device_ptr(stream);
    let (sin_ptr, _sin_sync) = sin.storage().buffer().as_slice().device_ptr(stream);
    unsafe {
        pos_embed::rope_raw(
            out_ptr as *mut c_void,
            x_ptr as *const c_void,
            cos_ptr as *const c_void,
            sin_ptr as *const c_void,
            (dims[0] * dims[1] * dims[2]) as u32,
            dims[3] as u32,
            dims[3] as u32,
            dims[3] as u32,
            dtype,
            stream.cu_stream(),
        )
    };
    drop(_out_sync);
    Ok(tensor_from_slice(out, x))
}

pub fn narrow_dim1(input: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    validate_cuda_input(input, "narrow_dim1")?;
    let dims = dims(input, 3, "narrow_dim1")?;
    if start + len > dims[1] { return Err(TensorError::ShapeMismatch("narrow_dim1 range exceeds dim 1".to_string())); }
    let shape = Shape::new([dims[0], len, dims[2]]).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
    launch_shape(input, &shape, |out, inp, dtype, stream| unsafe { narrow_dim1_fwd(out, inp, dims[0] as i32, dims[1] as i32, dims[2] as i32, start as i32, len as i32, dtype as i32, stream) })
}

pub fn transpose_1_2(input: &Tensor) -> Result<Tensor> {
    validate_cuda_input(input, "transpose_1_2")?;
    let d = dims(input, 4, "transpose_1_2")?;
    let shape = Shape::new([d[0], d[2], d[1], d[3]]).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
    launch_shape(input, &shape, |out, inp, dtype, stream| unsafe { transpose12_fwd(out, inp, d[0] as i32, d[1] as i32, d[2] as i32, d[3] as i32, dtype as i32, stream) })
}

pub fn affine(x: &Tensor, scale: f32, offset: f32) -> Result<Tensor> {
    validate_cuda_input(x, "affine")?;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let dtype = kernel_dtype(x.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(&stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    unsafe { affine_fwd(out_ptr as *mut c_void, x_ptr as *const c_void, x.numel() as i64, scale, offset, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(tensor_from_slice(out, x))
}

pub fn scale_bf16(x: &Tensor, scale: f32) -> Result<Tensor> { affine(x, scale, 0.0) }

pub fn greater_equal_bf16(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    same_shape(left, right, "greater_equal")?;
    launch_binary_custom(left, right, |out, a, b, dtype, stream| unsafe { ge_fwd(out, a, b, left.numel() as i64, dtype as i32, stream) })
}

pub fn where_cond_bf16(cond: &Tensor, true_val: &Tensor, false_val: &Tensor) -> Result<Tensor> {
    same_shape(cond, true_val, "where_cond")?;
    same_shape(cond, false_val, "where_cond")?;
    let stream = cond.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(cond.len_bytes())? };
    let dtype = kernel_dtype(cond.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (c_ptr, _c_sync) = cond.storage().buffer().as_slice().device_ptr(stream);
    let (t_ptr, _t_sync) = true_val.storage().buffer().as_slice().device_ptr(stream);
    let (f_ptr, _f_sync) = false_val.storage().buffer().as_slice().device_ptr(stream);
    unsafe { where_fwd(out_ptr as *mut c_void, c_ptr as *const c_void, t_ptr as *const c_void, f_ptr as *const c_void, cond.numel() as i64, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(tensor_from_slice(out, cond))
}

pub fn cast_bf16_to_f32(x: &Tensor) -> Result<Vec<f32>> {
    if x.dtype() != DType::BF16 { return Err(TensorError::DTypeMismatch { expected: DType::BF16.name(), actual: x.dtype().name() }); }
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<f32>(x.numel())? };
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    unsafe { crate::cast::cast_raw(out_ptr as *mut c_void, x_ptr as *const c_void, x.numel() as i64, KernelDType::Bf16, KernelDType::F32, stream.cu_stream()) };
    drop(out_sync);
    stream.synchronize()?;
    Ok(stream.clone_dtoh(&out)?)
}

pub fn apply_causal_mask(scores: &Tensor, query_len: usize, key_len: usize, start_pos: usize) -> Result<Tensor> {
    validate_cuda_input(scores, "causal_mask")?;
    let stream = scores.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(scores.len_bytes())? };
    let dtype = kernel_dtype(scores.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (s_ptr, _s_sync) = scores.storage().buffer().as_slice().device_ptr(stream);
    unsafe { causal_mask_fwd(out_ptr as *mut c_void, s_ptr as *const c_void, scores.numel() as i64, query_len as i32, key_len as i32, start_pos as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(tensor_from_slice(out, scores))
}

pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize, causal: bool, start_pos: usize) -> Result<Tensor> {
    same_shape(k, v, "sdpa kv")?;
    validate_cuda_input(q, "sdpa")?;
    let qd = dims(q, 4, "sdpa q")?;
    let kd = dims(k, 4, "sdpa k")?;
    if qd[0] != kd[0] || qd[1] != kd[1] || qd[3] != head_dim || kd[3] != head_dim { return Err(TensorError::ShapeMismatch("sdpa q/k/v dimensions are incompatible".to_string())); }
    let stream = q.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(q.len_bytes())? };
    let dtype = kernel_dtype(q.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (q_ptr, _q_sync) = q.storage().buffer().as_slice().device_ptr(stream);
    let (k_ptr, _k_sync) = k.storage().buffer().as_slice().device_ptr(stream);
    let (v_ptr, _v_sync) = v.storage().buffer().as_slice().device_ptr(stream);
    unsafe { sdpa_fwd(out_ptr as *mut c_void, q_ptr as *const c_void, k_ptr as *const c_void, v_ptr as *const c_void, qd[0] as i32, qd[1] as i32, qd[2] as i32, kd[2] as i32, head_dim as i32, 1.0 / (head_dim as f32).sqrt(), i32::from(causal), start_pos as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(tensor_from_slice(out, q))
}

pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    validate_cuda_input(x, "repeat_kv")?;
    let d = dims(x, 4, "repeat_kv")?;
    let shape = Shape::new([d[0], d[1] * n_rep, d[2], d[3]]).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
    launch_shape(x, &shape, |out, inp, dtype, stream| unsafe { repeat_kv_fwd(out, inp, d[0] as i32, d[1] as i32, d[2] as i32, d[3] as i32, n_rep as i32, dtype as i32, stream) })
}

pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    validate_cuda_input(x, "linear")?;
    validate_cuda_input(weight, "linear")?;
    let xd = x.shape().dims();
    let wd = dims(weight, 2, "linear weight")?;
    let in_f = *xd.last().ok_or_else(|| TensorError::ShapeMismatch("linear input needs rank >= 1".to_string()))?;
    if in_f != wd[1] { return Err(TensorError::ShapeMismatch("linear input and weight dimensions mismatch".to_string())); }
    let rows = x.numel() / in_f;
    let mut out_dims = xd.to_vec();
    *out_dims.last_mut().unwrap() = wd[0];
    let shape = Shape::new(out_dims).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(shape.numel() * x.dtype().size_in_bytes())? };
    let dtype = kernel_dtype(x.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    let (w_ptr, _w_sync) = weight.storage().buffer().as_slice().device_ptr(stream);
    let (bias_ptr, _bias_sync) = match bias { Some(b) => { let (p, s) = b.storage().buffer().as_slice().device_ptr(stream); (p as *const c_void, Some(s)) }, None => (std::ptr::null(), None) };
    unsafe { linear_fwd(out_ptr as *mut c_void, x_ptr as *const c_void, w_ptr as *const c_void, bias_ptr, rows as i32, in_f as i32, wd[0] as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(Tensor::from_storage(SharedStorage::new(CudaBuf::from_slice(out)), shape.clone(), Stride::contiguous(&shape), x.dtype()))
}

pub fn embedding_lookup(input_ids: &Tensor, embeddings: &Tensor) -> Result<Tensor> {
    if input_ids.dtype() != DType::I32 { return Err(TensorError::DTypeMismatch { expected: DType::I32.name(), actual: input_ids.dtype().name() }); }
    validate_cuda_input(embeddings, "embedding_lookup")?;
    let ed = dims(embeddings, 2, "embedding")?;
    let mut out_dims = input_ids.shape().dims().to_vec();
    out_dims.push(ed[1]);
    let shape = Shape::new(out_dims).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
    let stream = input_ids.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(shape.numel() * embeddings.dtype().size_in_bytes())? };
    let dtype = kernel_dtype(embeddings.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (ids_ptr, _ids_sync) = input_ids.storage().buffer().as_slice().device_ptr(stream);
    let (emb_ptr, _emb_sync) = embeddings.storage().buffer().as_slice().device_ptr(stream);
    unsafe { embedding_fwd(out_ptr as *mut c_void, ids_ptr as *const i32, emb_ptr as *const c_void, input_ids.numel() as i32, ed[1] as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(Tensor::from_storage(SharedStorage::new(CudaBuf::from_slice(out)), shape.clone(), Stride::contiguous(&shape), embeddings.dtype()))
}

pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    validate_cuda_input(x, "add_bias")?;
    validate_cuda_input(bias, "add_bias")?;
    let last = *x.shape().dims().last().ok_or_else(|| TensorError::ShapeMismatch("add_bias input needs rank >= 1".to_string()))?;
    if bias.shape().dims() != [last] { return Err(TensorError::ShapeMismatch(format!("bias must have shape [{last}]"))); }
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let dtype = kernel_dtype(x.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    let (b_ptr, _b_sync) = bias.storage().buffer().as_slice().device_ptr(stream);
    unsafe { add_bias_fwd(out_ptr as *mut c_void, x_ptr as *const c_void, b_ptr as *const c_void, x.numel() as i64, last as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(tensor_from_slice(out, x))
}

pub fn zeros_f32_like_shape(context: &runtime::CudaContext, shape: Shape) -> Result<Tensor> {
    let stream = context.cudarc().default_stream();
    let mut out = unsafe { stream.alloc::<u8>(shape.numel() * DType::F32.size_in_bytes())? };
    let zero = 0.0f32;
    let (out_ptr, out_sync) = out.device_ptr_mut(&stream);
    unsafe {
        ops::fill_raw(
            out_ptr as *mut c_void,
            (&zero as *const f32).cast(),
            shape.numel() as i64,
            KernelDType::F32,
            stream.cu_stream(),
        )
    };
    drop(out_sync);
    Ok(Tensor::from_storage(
        SharedStorage::new(CudaBuf::from_slice(out)),
        shape.clone(),
        Stride::contiguous(&shape),
        DType::F32,
    ))
}

pub fn concat_dim2(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    validate_cuda_input(left, "concat_dim2")?;
    validate_cuda_input(right, "concat_dim2")?;
    let a = dims(left, 4, "concat_dim2 left")?;
    let b = dims(right, 4, "concat_dim2 right")?;
    if a[0] != b[0] || a[1] != b[1] || a[3] != b[3] { return Err(TensorError::ShapeMismatch("concat_dim2 non-concat dimensions mismatch".to_string())); }
    let shape = Shape::new([a[0], a[1], a[2] + b[2], a[3]]).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
    let stream = left.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(shape.numel() * left.dtype().size_in_bytes())? };
    let dtype = kernel_dtype(left.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (a_ptr, _a_sync) = left.storage().buffer().as_slice().device_ptr(stream);
    let (b_ptr, _b_sync) = right.storage().buffer().as_slice().device_ptr(stream);
    unsafe { concat_dim2_fwd(out_ptr as *mut c_void, a_ptr as *const c_void, b_ptr as *const c_void, a[0] as i32, a[1] as i32, a[2] as i32, b[2] as i32, a[3] as i32, dtype as i32, stream.cu_stream()) };
    drop(out_sync);
    Ok(Tensor::from_storage(SharedStorage::new(CudaBuf::from_slice(out)), shape.clone(), Stride::contiguous(&shape), left.dtype()))
}

fn unary(x: &Tensor, op: UnaryOp) -> Result<Tensor> {
    validate_cuda_input(x, "unary")?;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let dtype = kernel_dtype(x.dtype())?;
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    unsafe { ops::unary_raw(out_ptr as *mut c_void, x_ptr as *const c_void, x.numel() as i64, op, dtype, stream.cu_stream()) };
    drop(_out_sync);
    Ok(tensor_from_slice(out, x))
}

fn binary(left: &Tensor, right: &Tensor, op: BinaryOp) -> Result<Tensor> {
    same_shape(left, right, "binary")?;
    let stream = left.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(left.len_bytes())? };
    let dtype = kernel_dtype(left.dtype())?;
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (left_ptr, _left_sync) = left.storage().buffer().as_slice().device_ptr(stream);
    let (right_ptr, _right_sync) = right.storage().buffer().as_slice().device_ptr(stream);
    unsafe {
        ops::binary_raw(
            out_ptr as *mut c_void,
            left_ptr as *const c_void,
            right_ptr as *const c_void,
            left.numel() as i64,
            op,
            dtype,
            stream.cu_stream(),
        )
    };
    drop(_out_sync);
    Ok(tensor_from_slice(out, left))
}

fn norm_common(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    validate_cuda_input(x, "norm")?;
    validate_cuda_input(weight, "norm")?;
    let dims = x.shape().dims();
    let Some(&cols) = dims.last() else {
        return Err(TensorError::ShapeMismatch("norm requires rank >= 1".to_string()));
    };
    if weight.shape().dims() != [cols] {
        return Err(TensorError::ShapeMismatch(format!("norm weight must have shape [{cols}]")));
    }
    let rows = x.numel() / cols;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let dtype = kernel_dtype(x.dtype())?;
    let block = 256;
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (x_ptr, _x_sync) = x.storage().buffer().as_slice().device_ptr(stream);
    let (w_ptr, _w_sync) = weight.storage().buffer().as_slice().device_ptr(stream);
    match bias {
        Some(bias) => {
            validate_cuda_input(bias, "layer_norm")?;
            let (b_ptr, _b_sync) = bias.storage().buffer().as_slice().device_ptr(stream);
            unsafe { norm::layernorm_raw(out_ptr as *mut c_void, x_ptr as *const c_void, w_ptr as *const c_void, b_ptr as *const c_void, rows as i32, cols as i32, block, eps, dtype, stream.cu_stream()) };
        }
        None => unsafe { norm::rmsnorm_raw(out_ptr as *mut c_void, x_ptr as *const c_void, w_ptr as *const c_void, rows as i32, cols as i32, block, eps, dtype, stream.cu_stream()) },
    }
    drop(_out_sync);
    Ok(tensor_from_slice(out, x))
}

fn tensor_from_slice(out: CudaSlice<u8>, like: &Tensor) -> Tensor {
    Tensor::from_storage(
        SharedStorage::new(CudaBuf::from_slice(out)),
        like.shape().clone(),
        Stride::contiguous(like.shape()),
        like.dtype(),
    )
}

fn same_shape(left: &Tensor, right: &Tensor, op: &'static str) -> Result<()> {
    validate_cuda_input(left, op)?;
    validate_cuda_input(right, op)?;
    if left.shape().dims() != right.shape().dims() {
        return Err(TensorError::ShapeMismatch(format!("{op} shape mismatch: {:?} vs {:?}", left.shape().dims(), right.shape().dims())));
    }
    if left.dtype() != right.dtype() {
        return Err(TensorError::DTypeMismatch { expected: left.dtype().name(), actual: right.dtype().name() });
    }
    Ok(())
}

fn dims(tensor: &Tensor, rank: usize, name: &str) -> Result<Vec<usize>> {
    let dims = tensor.shape().dims();
    if dims.len() != rank {
        return Err(TensorError::ShapeMismatch(format!("{name} expects rank {rank}, got {:?}", dims)));
    }
    Ok(dims.to_vec())
}

fn launch_shape(
    input: &Tensor,
    shape: &Shape,
    launch: impl FnOnce(*mut c_void, *const c_void, KernelDType, cudarc::driver::sys::CUstream),
) -> Result<Tensor> {
    let stream = input.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(shape.numel() * input.dtype().size_in_bytes())? };
    let dtype = kernel_dtype(input.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (in_ptr, _in_sync) = input.storage().buffer().as_slice().device_ptr(stream);
    launch(out_ptr as *mut c_void, in_ptr as *const c_void, dtype, stream.cu_stream());
    drop(out_sync);
    Ok(Tensor::from_storage(
        SharedStorage::new(CudaBuf::from_slice(out)),
        shape.clone(),
        Stride::contiguous(shape),
        input.dtype(),
    ))
}

fn launch_binary_custom(
    left: &Tensor,
    right: &Tensor,
    launch: impl FnOnce(*mut c_void, *const c_void, *const c_void, KernelDType, cudarc::driver::sys::CUstream),
) -> Result<Tensor> {
    let stream = left.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(left.len_bytes())? };
    let dtype = kernel_dtype(left.dtype())?;
    let (out_ptr, out_sync) = out.device_ptr_mut(stream);
    let (left_ptr, _left_sync) = left.storage().buffer().as_slice().device_ptr(stream);
    let (right_ptr, _right_sync) = right.storage().buffer().as_slice().device_ptr(stream);
    launch(out_ptr as *mut c_void, left_ptr as *const c_void, right_ptr as *const c_void, dtype, stream.cu_stream());
    drop(out_sync);
    Ok(tensor_from_slice(out, left))
}

fn validate_cuda_input(tensor: &Tensor, _: &'static str) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(TensorError::NonContiguous);
    }
    let _ = kernel_dtype(tensor.dtype())?;
    Ok(())
}

fn kernel_dtype(dtype: DType) -> Result<KernelDType> {
    match dtype {
        DType::BF16 => Ok(KernelDType::Bf16),
        DType::F16 => Ok(KernelDType::F16),
        DType::F32 => Ok(KernelDType::F32),
        other => Err(TensorError::DTypeMismatch { expected: "bf16/f16/f32", actual: other.name() }),
    }
}
