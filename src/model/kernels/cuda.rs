use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor, WithDType};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/lmrs_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "lmrs_qwen_kernels";

pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    if !gate_up.device().is_cuda() {
        return super::fallback_silu_mul(gate_up, intermediate_size);
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

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(
            src: &[T],
            layout: &Layout,
            intermediate_size: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
                candle_core::Error::Msg("qwen_fused_silu_mul requires contiguous input".into())
            })?;
            let src = &src[o1..o2];
            let dims = layout.shape().dims();
            let last = *dims.last().unwrap_or(&0);
            if last != intermediate_size * 2 {
                candle_core::bail!(
                    "qwen_fused_silu_mul last dim {last} != {}",
                    intermediate_size * 2
                );
            }
            let rows = src.len() / last;
            let mut dst = vec![T::zero(); rows * intermediate_size];
            for row in 0..rows {
                for i in 0..intermediate_size {
                    let g = src[row * last + i].to_f64();
                    let u = src[row * last + intermediate_size + i].to_f64();
                    let silu = g / (1.0 + (-g).exp());
                    dst[row * intermediate_size + i] = T::from_f64(silu * u);
                }
            }
            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = intermediate_size;
            Ok((T::to_cpu_storage_owned(dst), Shape::from_dims(&out_dims)))
        }

        match storage {
            CpuStorage::BF16(s) => inner(s, layout, self.intermediate_size),
            CpuStorage::F16(s) => inner(s, layout, self.intermediate_size),
            CpuStorage::F32(s) => inner(s, layout, self.intermediate_size),
            CpuStorage::F64(s) => inner(s, layout, self.intermediate_size),
            _ => candle_core::bail!("qwen_fused_silu_mul unsupported dtype"),
        }
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
        let fn_name = match storage.dtype() {
            DType::BF16 => "qwen_fused_silu_mul_bf16",
            DType::F16 => "qwen_fused_silu_mul_f16",
            DType::F32 => "qwen_fused_silu_mul_f32",
            dt => candle_core::bail!("qwen_fused_silu_mul unsupported dtype {dt:?}"),
        };
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
            CudaStorageSlice::F16(s) => {
                let src = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::f16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&intermediate_size);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            CudaStorageSlice::F32(s) => {
                let src = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<f32>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&intermediate_size);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            _ => candle_core::bail!("qwen_fused_silu_mul unsupported CUDA storage"),
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
