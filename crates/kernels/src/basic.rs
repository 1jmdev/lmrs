use cudarc::driver::CudaView;
use cudarc::driver::CudaViewMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use tensor::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor, TensorError};

use crate::ptx;

const ADD_BIAS_BF16: &str = "add_bias_bf16";
const ADD_BF16: &str = "add_bf16";
const LINEAR_BF16: &str = "linear_bf16";
const EMBEDDING_LOOKUP_I32_BF16: &str = "embedding_lookup_i32_bf16";
const ZERO_F32: &str = "zero_f32";
const CONCAT_DIM2_BF16: &str = "concat_dim2_bf16";
const TRANSPOSE_1_2_BF16: &str = "transpose_1_2_bf16";
const NARROW_DIM1_BF16: &str = "narrow_dim1_bf16";

pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    validate_bf16_contiguous(x)?;
    validate_bf16_contiguous(bias)?;
    let cols = last_dim(x, "x")?;
    if bias.shape().dims() != [cols] {
        return Err(TensorError::ShapeMismatch(format!(
            "bias must have shape [{cols}], got {:?}",
            bias.shape().dims()
        )));
    }
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(x.len_bytes())? };
    let x_view = bf16_view(x)?;
    let bias_view = bf16_view(bias)?;
    let mut out_view = bf16_mut_view(&mut out, x.numel(), "add_bias output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(ADD_BIAS_BF16)?;
    let total_elements = to_i32(x.numel(), "total_elements")?;
    let cols = to_i32(cols, "cols")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_view);
    builder.arg(&bias_view);
    builder.arg(&mut out_view);
    builder.arg(&total_elements);
    builder.arg(&cols);
    launch_1d(&mut builder, x.numel())?;
    output_like(x, out)
}

/// Adds two contiguous CUDA BF16 tensors with the same shape.
///
/// # Example
///
/// ```no_run
/// use kernels::add;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([2])?;
/// let left = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
/// let right = copy_h2d(&context, shape, DType::BF16, &[0u16; 2])?;
/// let out = add(&left, &right)?;
/// assert_eq!(out.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn add(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    validate_bf16_contiguous(left)?;
    validate_bf16_contiguous(right)?;
    if left.shape().dims() != right.shape().dims() {
        return Err(TensorError::ShapeMismatch(format!(
            "add expected matching shapes, got {:?} and {:?}",
            left.shape().dims(),
            right.shape().dims()
        )));
    }
    let stream = left.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(left.len_bytes())? };
    let left_view = bf16_view(left)?;
    let right_view = bf16_view(right)?;
    let mut out_view = bf16_mut_view(&mut out, left.numel(), "add output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(ADD_BF16)?;
    let total_elements = to_i32(left.numel(), "total_elements")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&left_view);
    builder.arg(&right_view);
    builder.arg(&mut out_view);
    builder.arg(&total_elements);
    launch_1d(&mut builder, left.numel())?;
    output_like(left, out)
}

pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    validate_bf16_contiguous(x)?;
    validate_bf16_contiguous(weight)?;
    let in_features = last_dim(x, "x")?;
    let weight_dims = weight.shape().dims();
    if weight_dims.len() != 2 || weight_dims[1] != in_features {
        return Err(TensorError::ShapeMismatch(format!(
            "weight must have shape [out_features, {in_features}], got {weight_dims:?}"
        )));
    }
    if let Some(bias) = bias {
        validate_bf16_contiguous(bias)?;
        if bias.shape().dims() != [weight_dims[0]] {
            return Err(TensorError::ShapeMismatch(format!(
                "bias must have shape [{}], got {:?}",
                weight_dims[0],
                bias.shape().dims()
            )));
        }
    }
    let rows = x.numel() / in_features;
    let out_features = weight_dims[0];
    let mut out_dims = x.shape().dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = out_features;
    let out_shape = Shape::new(out_dims).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = x.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(out_shape.numel() * DType::BF16.size_in_bytes())? };
    let x_view = bf16_view(x)?;
    let weight_view = bf16_view(weight)?;
    let empty_bias = unsafe { stream.alloc::<u8>(DType::BF16.size_in_bytes())? };
    let bias_view = match bias {
        Some(bias) => bf16_view(bias)?,
        None => unsafe { empty_bias.transmute::<u16>(1) }.ok_or_else(|| {
            TensorError::InvalidArgument("failed to create empty bias view".to_string())
        })?,
    };
    let mut out_view = bf16_mut_view(&mut out, out_shape.numel(), "linear output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(LINEAR_BF16)?;
    let rows_i32 = to_i32(rows, "rows")?;
    let in_features_i32 = to_i32(in_features, "in_features")?;
    let out_features_i32 = to_i32(out_features, "out_features")?;
    let has_bias = i32::from(bias.is_some());
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_view);
    builder.arg(&weight_view);
    builder.arg(&bias_view);
    builder.arg(&mut out_view);
    builder.arg(&rows_i32);
    builder.arg(&in_features_i32);
    builder.arg(&out_features_i32);
    builder.arg(&has_bias);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                to_u32(out_features.div_ceil(16), "grid_x")?,
                to_u32(rows.div_ceil(16), "grid_y")?,
                1,
            ),
            block_dim: (16, 16, 1),
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

pub fn embedding_lookup(input_ids: &Tensor, embeddings: &Tensor) -> Result<Tensor> {
    validate_contiguous(input_ids)?;
    if input_ids.dtype() != DType::I32 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::I32.name(),
            actual: input_ids.dtype().name(),
        });
    }
    validate_bf16_contiguous(embeddings)?;
    let embed_dims = embeddings.shape().dims();
    if embed_dims.len() != 2 {
        return Err(TensorError::ShapeMismatch(format!(
            "embeddings must be rank 2, got rank {}",
            embed_dims.len()
        )));
    }
    let hidden_size = embed_dims[1];
    let mut out_dims = input_ids.shape().dims().to_vec();
    out_dims.push(hidden_size);
    let out_shape = Shape::new(out_dims).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = embeddings.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(out_shape.numel() * DType::BF16.size_in_bytes())? };
    let ids = i32_view(input_ids)?;
    let embeddings_view = bf16_view(embeddings)?;
    let mut out_view = bf16_mut_view(&mut out, out_shape.numel(), "embedding output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(EMBEDDING_LOOKUP_I32_BF16)?;
    let tokens = to_i32(input_ids.numel(), "tokens")?;
    let hidden_size_i32 = to_i32(hidden_size, "hidden_size")?;
    let vocab_size = to_i32(embed_dims[0], "vocab_size")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&ids);
    builder.arg(&embeddings_view);
    builder.arg(&mut out_view);
    builder.arg(&tokens);
    builder.arg(&hidden_size_i32);
    builder.arg(&vocab_size);
    launch_1d(&mut builder, out_shape.numel())?;
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        out_shape.clone(),
        Stride::contiguous(&out_shape),
        DType::BF16,
    ))
}

pub fn zeros_f32_like_shape(context: &runtime::CudaContext, shape: Shape) -> Result<Tensor> {
    let stream = context.cudarc().default_stream();
    let mut out = unsafe { stream.alloc::<u8>(shape.numel() * DType::F32.size_in_bytes())? };
    let mut out_view = unsafe { out.transmute_mut::<f32>(shape.numel()) }.ok_or_else(|| {
        TensorError::InvalidArgument("failed to create F32 zero output view".to_string())
    })?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(ZERO_F32)?;
    let total_elements = to_i32(shape.numel(), "total_elements")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut out_view);
    builder.arg(&total_elements);
    launch_1d(&mut builder, shape.numel())?;
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        shape.clone(),
        Stride::contiguous(&shape),
        DType::F32,
    ))
}

pub fn concat_dim2(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    validate_bf16_contiguous(left)?;
    validate_bf16_contiguous(right)?;
    let left_dims = dims4(left, "left")?;
    let right_dims = dims4(right, "right")?;
    if left_dims[0] != right_dims[0]
        || left_dims[1] != right_dims[1]
        || left_dims[3] != right_dims[3]
    {
        return Err(TensorError::ShapeMismatch(format!(
            "concat_dim2 expected matching batch/heads/head_dim, got {:?} and {:?}",
            left_dims, right_dims
        )));
    }
    let out_shape = Shape::new([
        left_dims[0],
        left_dims[1],
        left_dims[2] + right_dims[2],
        left_dims[3],
    ])
    .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = left.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(out_shape.numel() * DType::BF16.size_in_bytes())? };
    let left_view = bf16_view(left)?;
    let right_view = bf16_view(right)?;
    let mut out_view = bf16_mut_view(&mut out, out_shape.numel(), "concat output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(CONCAT_DIM2_BF16)?;
    let total_elements = to_i32(out_shape.numel(), "total_elements")?;
    let batch = to_i32(left_dims[0], "batch")?;
    let heads = to_i32(left_dims[1], "heads")?;
    let left_seq = to_i32(left_dims[2], "left_seq")?;
    let right_seq = to_i32(right_dims[2], "right_seq")?;
    let head_dim = to_i32(left_dims[3], "head_dim")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&left_view);
    builder.arg(&right_view);
    builder.arg(&mut out_view);
    builder.arg(&total_elements);
    builder.arg(&batch);
    builder.arg(&heads);
    builder.arg(&left_seq);
    builder.arg(&right_seq);
    builder.arg(&head_dim);
    launch_1d(&mut builder, out_shape.numel())?;
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        out_shape.clone(),
        Stride::contiguous(&out_shape),
        DType::BF16,
    ))
}

/// Transposes dimensions 1 and 2 of a rank-4 contiguous CUDA BF16 tensor.
///
/// # Example
///
/// ```no_run
/// use kernels::transpose_1_2;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([1, 2, 3, 4])?, DType::BF16, &[0u16; 24])?;
/// let y = transpose_1_2(&x)?;
/// assert_eq!(y.shape().dims(), &[1, 3, 2, 4]);
/// # Ok(())
/// # }
/// ```
pub fn transpose_1_2(input: &Tensor) -> Result<Tensor> {
    validate_bf16_contiguous(input)?;
    let dims = dims4(input, "input")?;
    let out_shape = Shape::new([dims[0], dims[2], dims[1], dims[3]])
        .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = input.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(input.len_bytes())? };
    let input_view = bf16_view(input)?;
    let mut out_view = bf16_mut_view(&mut out, input.numel(), "transpose output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(TRANSPOSE_1_2_BF16)?;
    let batch = to_i32(dims[0], "batch")?;
    let dim1 = to_i32(dims[1], "dim1")?;
    let dim2 = to_i32(dims[2], "dim2")?;
    let dim3 = to_i32(dims[3], "dim3")?;
    let total_elements = to_i32(input.numel(), "total_elements")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_view);
    builder.arg(&mut out_view);
    builder.arg(&batch);
    builder.arg(&dim1);
    builder.arg(&dim2);
    builder.arg(&dim3);
    builder.arg(&total_elements);
    launch_1d(&mut builder, input.numel())?;
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        out_shape.clone(),
        Stride::contiguous(&out_shape),
        DType::BF16,
    ))
}

/// Copies a contiguous dimension-1 slice from a rank-3 CUDA BF16 tensor.
///
/// # Example
///
/// ```no_run
/// use kernels::narrow_dim1;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([1, 2, 4])?, DType::BF16, &[0u16; 8])?;
/// let y = narrow_dim1(&x, 1, 1)?;
/// assert_eq!(y.shape().dims(), &[1, 1, 4]);
/// # Ok(())
/// # }
/// ```
pub fn narrow_dim1(input: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    validate_bf16_contiguous(input)?;
    let dims = input.shape().dims();
    if dims.len() != 3 {
        return Err(TensorError::ShapeMismatch(format!(
            "narrow_dim1 input must be rank 3, got rank {}",
            dims.len()
        )));
    }
    if start.checked_add(len).is_none_or(|end| end > dims[1]) {
        return Err(TensorError::InvalidArgument(format!(
            "narrow_dim1 range [{start}, {}) exceeds dim {}",
            start + len,
            dims[1]
        )));
    }
    let out_shape = Shape::new([dims[0], len, dims[2]])
        .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    let stream = input.storage().buffer().as_slice().stream();
    let mut out = unsafe { stream.alloc::<u8>(out_shape.numel() * DType::BF16.size_in_bytes())? };
    let input_view = bf16_view(input)?;
    let mut out_view = bf16_mut_view(&mut out, out_shape.numel(), "narrow output")?;
    let module = stream.context().load_module(Ptx::from_src(ptx::BASIC))?;
    let func = module.load_function(NARROW_DIM1_BF16)?;
    let batch = to_i32(dims[0], "batch")?;
    let dim1 = to_i32(dims[1], "dim1")?;
    let width = to_i32(dims[2], "width")?;
    let start = to_i32(start, "start")?;
    let len_i32 = to_i32(len, "len")?;
    let total_elements = to_i32(out_shape.numel(), "total_elements")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_view);
    builder.arg(&mut out_view);
    builder.arg(&batch);
    builder.arg(&dim1);
    builder.arg(&width);
    builder.arg(&start);
    builder.arg(&len_i32);
    builder.arg(&total_elements);
    launch_1d(&mut builder, out_shape.numel())?;
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        out_shape.clone(),
        Stride::contiguous(&out_shape),
        DType::BF16,
    ))
}

fn output_like(input: &Tensor, out: cudarc::driver::CudaSlice<u8>) -> Result<Tensor> {
    let storage = SharedStorage::new(CudaBuf::from_slice(out));
    Ok(Tensor::from_storage(
        storage,
        input.shape().clone(),
        Stride::contiguous(input.shape()),
        input.dtype(),
    ))
}

fn validate_bf16_contiguous(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::BF16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::BF16.name(),
            actual: tensor.dtype().name(),
        });
    }
    validate_contiguous(tensor)
}

fn validate_contiguous(tensor: &Tensor) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(TensorError::NonContiguous);
    }
    Ok(())
}

fn last_dim(tensor: &Tensor, name: &str) -> Result<usize> {
    tensor.shape().dims().last().copied().ok_or_else(|| {
        TensorError::ShapeMismatch(format!("{name} must have at least one dimension"))
    })
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

fn i32_view(tensor: &Tensor) -> Result<CudaView<'_, i32>> {
    unsafe {
        tensor
            .storage()
            .buffer()
            .as_slice()
            .transmute::<i32>(tensor.numel())
    }
    .ok_or_else(|| TensorError::InvalidArgument("failed to create I32 tensor view".to_string()))
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

fn launch_1d(builder: &mut cudarc::driver::LaunchArgs<'_>, elements: usize) -> Result<()> {
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (to_u32(elements.div_ceil(256), "grid_x")?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    Ok(())
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds i32::MAX")))
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| TensorError::InvalidArgument(format!("{name} {value} exceeds u32::MAX")))
}
