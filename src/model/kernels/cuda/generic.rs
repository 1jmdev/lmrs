use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Tensor};

use super::{GENERIC_MODULE_NAME, ptx};

pub fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if !device.is_cuda() {
        candle_core::bail!("causal_mask requires CUDA");
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

struct CausalMask {
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
}

impl candle_core::CustomOp1 for CausalMask {
    fn name(&self) -> &'static str {
        "generic-causal-mask"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("causal_mask is CUDA-only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("causal_mask requires contiguous input");
        }
        if storage.dtype() != DType::BF16 {
            candle_core::bail!("causal_mask is BF16-only");
        }
        let dev = storage.device();
        let elem_count = layout.shape().elem_count();
        let func = dev.get_or_load_custom_func(
            "generic_causal_mask_bf16",
            GENERIC_MODULE_NAME,
            ptx::GENERIC_KERNELS,
        )?;
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
            _ => candle_core::bail!("causal_mask is BF16-only"),
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

pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    if logits.dtype() != DType::BF16 {
        candle_core::bail!("gpu_argmax is BF16-only");
    }
    let dev = match logits.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax requires CUDA"),
    };
    let logits = logits.contiguous()?.flatten_all()?;
    let vocab_size = logits.elem_count();
    let (storage, layout) = logits.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("gpu_argmax requires CUDA storage"),
    };
    let (o1, _o2) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("gpu_argmax requires contiguous logits".into()))?;

    let out = unsafe { dev.alloc::<i32>(1)? };
    let func = dev.get_or_load_custom_func(
        "generic_argmax_bf16",
        GENERIC_MODULE_NAME,
        ptx::GENERIC_KERNELS,
    )?;
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
        _ => candle_core::bail!("gpu_argmax is BF16-only"),
    }
    let token = dev.clone_dtoh(&out)?;
    Ok(token[0] as u32)
}
