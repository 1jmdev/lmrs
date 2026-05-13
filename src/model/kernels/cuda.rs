use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Tensor};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/lmrs_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "lmrs_qwen_kernels";

pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    if !gate_up.device().is_cuda() {
        candle_core::bail!("qwen_fused_silu_mul requires a CUDA tensor");
    }
    gate_up.apply_op1_no_bwd(&FusedSiluMul { intermediate_size })
}

pub fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if !device.is_cuda() {
        candle_core::bail!("qwen_causal_mask requires CUDA");
    }
    let zeros = Tensor::zeros((seq_len, total_len), DType::BF16, device)?;
    zeros
        .apply_op1_no_bwd(&CausalMask {
            seq_len,
            total_len,
            start_pos,
        })?
        .unsqueeze(0)?
        .unsqueeze(0)
}

struct FusedSiluMul {
    intermediate_size: usize,
}

struct CausalMask {
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
}

impl candle_core::CustomOp1 for CausalMask {
    fn name(&self) -> &'static str {
        "qwen-causal-mask"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen_causal_mask is CUDA-only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("qwen_causal_mask requires contiguous input");
        }
        let dev = storage.device();
        let elem_count = layout.shape().elem_count();
        if storage.dtype() != DType::BF16 {
            candle_core::bail!("qwen_causal_mask is BF16-only");
        }
        let fn_name = "qwen_causal_mask_bf16";
        let func = dev.get_or_load_custom_func(fn_name, MODULE_NAME, ptx::QWEN_KERNELS)?;
        let seq_len = self.seq_len as i32;
        let total_len = self.total_len as i32;
        let start_pos = self.start_pos as i32;
        let cfg = LaunchConfig {
            grid_dim: (
                (self.total_len as u32).div_ceil(256),
                self.seq_len as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let slice = match &storage.slice {
            CudaStorageSlice::BF16(_) => {
                let dst = unsafe { dev.alloc::<half::bf16>(elem_count)? };
                let mut builder = func.builder();
                builder.arg(&dst);
                builder.arg(&seq_len);
                builder.arg(&total_len);
                builder.arg(&start_pos);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            _ => candle_core::bail!("qwen_causal_mask is BF16-only"),
        };
        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            layout.shape().clone(),
        ))
    }
}

impl candle_core::CustomOp1 for FusedSiluMul {
    fn name(&self) -> &'static str {
        "qwen-fused-silu-mul"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen_fused_silu_mul is CUDA-only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = storage.device();
        let dims = layout.shape().dims();
        let last = *dims.last().unwrap_or(&0);
        if last != self.intermediate_size * 2 {
            candle_core::bail!(
                "qwen_fused_silu_mul last dim {last} != {}",
                self.intermediate_size * 2
            );
        }
        let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("qwen_fused_silu_mul requires contiguous input".into())
        })?;
        let rows = (o2 - o1) / last;
        let out_el = rows * self.intermediate_size;
        if storage.dtype() != DType::BF16 {
            candle_core::bail!("qwen_fused_silu_mul is BF16-only");
        }
        let fn_name = "qwen_fused_silu_mul_bf16";
        let func = dev.get_or_load_custom_func(fn_name, MODULE_NAME, ptx::QWEN_KERNELS)?;
        let intermediate_size = self.intermediate_size as i32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: ((self.intermediate_size as u32).min(1024), 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => {
                let src = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::bf16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&intermediate_size);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            _ => candle_core::bail!("qwen_fused_silu_mul is BF16-only"),
        };
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = self.intermediate_size;
        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            Shape::from_dims(&out_dims),
        ))
    }
}

pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    if logits.dtype() != DType::BF16 {
        candle_core::bail!("qwen_argmax is BF16-only");
    }
    let dev = match logits.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("qwen_argmax requires CUDA"),
    };
    let logits = logits.contiguous()?.flatten_all()?;
    let vocab_size = logits.elem_count();
    let (storage, layout) = logits.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qwen_argmax requires CUDA storage"),
    };
    let (o1, _o2) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("qwen_argmax requires contiguous logits".into()))?;

    let out = unsafe { dev.alloc::<i32>(1)? };
    let func = dev.get_or_load_custom_func("qwen_argmax_bf16", MODULE_NAME, ptx::QWEN_KERNELS)?;
    let vocab_size_i32 = vocab_size as i32;
    match &storage.slice {
        CudaStorageSlice::BF16(s) => {
            let src = s.slice(o1..);
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&out);
            builder.arg(&vocab_size_i32);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }
        _ => candle_core::bail!("qwen_argmax is BF16-only"),
    }
    let token = dev.clone_dtoh(&out)?;
    Ok(token[0] as u32)
}
