use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};

use super::{QWEN_MODULE_NAME, ptx};

pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    if !gate_up.device().is_cuda() {
        candle_core::bail!("qwen_fused_silu_mul requires CUDA");
    }
    gate_up.apply_op1_no_bwd(&FusedSiluMul { intermediate_size })
}

struct FusedSiluMul {
    intermediate_size: usize,
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
        let func = dev.get_or_load_custom_func(
            "qwen_fused_silu_mul_bf16",
            QWEN_MODULE_NAME,
            ptx::QWEN_KERNELS,
        )?;
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
