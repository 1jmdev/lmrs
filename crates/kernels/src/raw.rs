use std::ffi::c_void;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, sys};

pub type Result<T> = std::result::Result<T, KernelError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum KernelDType {
    Bf16 = 0,
    F16 = 1,
    F32 = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum UnaryOp {
    Silu = 0,
    Gelu = 1,
    Relu = 2,
    Exp = 3,
    Log = 4,
    Sigmoid = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum BinaryOp {
    Add = 0,
    Mul = 1,
    Sub = 2,
    Div = 3,
    Maximum = 4,
    Minimum = 5,
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("{op} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{op} requires {required} bytes, buffer has {actual}")]
    BufferTooSmall {
        op: &'static str,
        required: usize,
        actual: usize,
    },
}

unsafe extern "C" {
    fn silu_mul_fwd(
        out: *mut c_void,
        gate: *const c_void,
        up: *const c_void,
        numel: i64,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn paged_attention_v1_fwd(
        out: *mut c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const i32,
        context_lens: *const i32,
        num_seqs: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_size: i32,
        block_size: i32,
        max_num_blocks_per_seq: i32,
        scale: f32,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn cast_fwd(
        out: *mut c_void,
        inp: *const c_void,
        numel: i64,
        src_dtype: i32,
        dst_dtype: i32,
        stream: sys::CUstream,
    );
    fn rmsnorm_fwd(
        dst: *mut c_void,
        src: *const c_void,
        alpha: *const c_void,
        n_rows: i32,
        n_cols: i32,
        block_size: i32,
        eps: f32,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn layernorm_fwd(
        dst: *mut c_void,
        src: *const c_void,
        alpha: *const c_void,
        beta: *const c_void,
        n_rows: i32,
        n_cols: i32,
        block_size: i32,
        eps: f32,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn unary_fwd(
        out: *mut c_void,
        inp: *const c_void,
        numel: i64,
        op: i32,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn binary_fwd(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        numel: i64,
        op: i32,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn fill_fwd(
        out: *mut c_void,
        value: *const c_void,
        numel: i64,
        dtype: i32,
        stream: sys::CUstream,
    );
    fn rope_fwd(
        dst: *mut c_void,
        src: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        bh: u32,
        td: u32,
        d: u32,
        stride_b: u32,
        dtype: i32,
        stream: sys::CUstream,
    );
}

pub mod activation {
    use super::*;

    pub unsafe fn silu_mul_raw(
        out: *mut c_void,
        gate: *const c_void,
        up: *const c_void,
        numel: i64,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe { silu_mul_fwd(out, gate, up, numel, dtype as i32, stream) };
    }

    pub fn silu_mul<T>(
        stream: &CudaStream,
        out: &mut CudaSlice<T>,
        gate: &CudaSlice<T>,
        up: &CudaSlice<T>,
        numel: usize,
        dtype: KernelDType,
    ) -> Result<()> {
        ensure_len("silu_mul", out.len(), numel)?;
        ensure_len("silu_mul", gate.len(), numel)?;
        ensure_len("silu_mul", up.len(), numel)?;
        let (out, _out_sync) = out.device_ptr_mut(stream);
        let (gate, _gate_sync) = gate.device_ptr(stream);
        let (up, _up_sync) = up.device_ptr(stream);
        unsafe {
            silu_mul_raw(
                out as *mut c_void,
                gate as *const c_void,
                up as *const c_void,
                numel as i64,
                dtype,
                stream.cu_stream(),
            )
        };
        Ok(())
    }
}

pub mod attention {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct PagedAttentionParams {
        pub num_seqs: i32,
        pub num_heads: i32,
        pub num_kv_heads: i32,
        pub head_size: i32,
        pub block_size: i32,
        pub max_num_blocks_per_seq: i32,
        pub scale: f32,
        pub dtype: KernelDType,
    }

    pub unsafe fn paged_attention_v1_raw(
        out: *mut c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const i32,
        context_lens: *const i32,
        params: PagedAttentionParams,
        stream: sys::CUstream,
    ) {
        unsafe {
            paged_attention_v1_fwd(
                out,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                params.num_seqs,
                params.num_heads,
                params.num_kv_heads,
                params.head_size,
                params.block_size,
                params.max_num_blocks_per_seq,
                params.scale,
                params.dtype as i32,
                stream,
            )
        };
    }
}

pub mod cast {
    use super::*;

    pub unsafe fn cast_raw(
        out: *mut c_void,
        inp: *const c_void,
        numel: i64,
        src_dtype: KernelDType,
        dst_dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe { cast_fwd(out, inp, numel, src_dtype as i32, dst_dtype as i32, stream) };
    }
}

pub mod norm {
    use super::*;

    pub unsafe fn rmsnorm_raw(
        dst: *mut c_void,
        src: *const c_void,
        alpha: *const c_void,
        n_rows: i32,
        n_cols: i32,
        block_size: i32,
        eps: f32,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe {
            rmsnorm_fwd(
                dst,
                src,
                alpha,
                n_rows,
                n_cols,
                block_size,
                eps,
                dtype as i32,
                stream,
            )
        };
    }

    pub unsafe fn layernorm_raw(
        dst: *mut c_void,
        src: *const c_void,
        alpha: *const c_void,
        beta: *const c_void,
        n_rows: i32,
        n_cols: i32,
        block_size: i32,
        eps: f32,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe {
            layernorm_fwd(
                dst,
                src,
                alpha,
                beta,
                n_rows,
                n_cols,
                block_size,
                eps,
                dtype as i32,
                stream,
            )
        };
    }
}

pub mod ops {
    use super::*;

    pub unsafe fn unary_raw(
        out: *mut c_void,
        inp: *const c_void,
        numel: i64,
        op: UnaryOp,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe { unary_fwd(out, inp, numel, op as i32, dtype as i32, stream) };
    }

    pub unsafe fn binary_raw(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        numel: i64,
        op: BinaryOp,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe { binary_fwd(out, lhs, rhs, numel, op as i32, dtype as i32, stream) };
    }

    pub unsafe fn fill_raw(
        out: *mut c_void,
        value: *const c_void,
        numel: i64,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe { fill_fwd(out, value, numel, dtype as i32, stream) };
    }
}

pub mod pos_embed {
    use super::*;

    pub unsafe fn rope_raw(
        dst: *mut c_void,
        src: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        bh: u32,
        td: u32,
        d: u32,
        stride_b: u32,
        dtype: KernelDType,
        stream: sys::CUstream,
    ) {
        unsafe { rope_fwd(dst, src, cos, sin, bh, td, d, stride_b, dtype as i32, stream) };
    }
}

fn ensure_len(op: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        Err(KernelError::LengthMismatch {
            op,
            expected,
            actual,
        })
    } else {
        Ok(())
    }
}
