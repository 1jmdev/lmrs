use crate::ptx;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Tensor};

const MODULE_NAME: &str = "lmrs_attention_paged";

/// Builds a BF16 causal attention mask on the current CUDA device.
///
/// The result shape is `[1, 1, seq_len, total_len]` so it broadcasts over
/// batch and head dimensions before softmax.
pub fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    device: &Device,
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
        "attention-causal-mask"
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
            MODULE_NAME,
            ptx::ATTENTION_PAGED_ATTN_FWD,
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
