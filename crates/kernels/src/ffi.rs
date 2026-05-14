use core::ffi::c_void;

pub type F16 = u16;
pub type BF16 = u16;
pub type F8E4M3 = u8;
pub type CudaStream = *mut c_void;

macro_rules! affine_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                inp: *const $ty,
                out: *mut $ty,
                mul: $ty,
                add: $ty,
            );)+
        }
    };
}

macro_rules! unary_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                inp: *const $ty,
                out: *mut $ty,
            );)+
        }
    };
}

macro_rules! unary_param_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                param: $ty,
                inp: *const $ty,
                out: *mut $ty,
            );)+
        }
    };
}

macro_rules! binary_fns {
    ($($name:ident($ty:ty, $out:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                lhs: *const $ty,
                rhs: *const $ty,
                out: *mut $out,
            );)+
        }
    };
}

macro_rules! cast_fns {
    ($($name:ident($src:ty, $dst:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                inp: *const $src,
                out: *mut $dst,
            );)+
        }
    };
}

macro_rules! fill_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(buf: *mut $ty, value: $ty, numel: usize);)+
        }
    };
}

macro_rules! copy2d_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                src: *const $ty,
                dst: *mut $ty,
                d1: u32,
                d2: u32,
                src_s: u32,
                dst_s: u32,
            );)+
        }
    };
}

macro_rules! const_set_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                inp: $ty,
                out: *mut $ty,
            );)+
        }
    };
}

macro_rules! sort_fns {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(x: *const $ty, dst: *mut u32, ncols: i32, ncols_pad: i32);)+
        }
    };
}

macro_rules! where_fns {
    ($($name:ident($id:ty, $ty:ty)),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                numel: usize,
                num_dims: usize,
                info: *const usize,
                ids: *const $id,
                t: *const $ty,
                f: *const $ty,
                out: *mut $ty,
            );)+
        }
    };
}

macro_rules! dequant_vec_fns {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(vx: *const c_void, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32);)+
        }
    };
}

macro_rules! mat_vec_q8_1_fns {
    ($($name:ident),+ $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(
                vx: *const c_void,
                y: *const c_void,
                dst: *mut f32,
                ncols: i32,
                nrows: i32,
                nchannels_x: i32,
                nchannels_y: i32,
            );)+
        }
    };
}

pub mod moe {
    use super::{CudaStream, c_void};

    unsafe extern "C" {
        pub fn moe_gemm_wmma(
            input: *const c_void,
            weights: *const c_void,
            sorted_token_ids: *const i32,
            expert_ids: *const i32,
            topk_weights: *const f32,
            output: *mut c_void,
            expert_counts: *mut i32,
            expert_offsets: *mut i32,
            num_experts: i32,
            topk: i32,
            size_m: i32,
            size_n: i32,
            size_k: i32,
            dtype: i32,
            is_prefill: bool,
            stream: CudaStream,
        );
    }
}

pub mod nn {
    use super::{BF16, F8E4M3, F16};

    affine_fns! {
        affine_bf16(BF16), affine_f8_e4m3(F8E4M3), affine_f16(F16), affine_f32(f32), affine_f64(f64),
        affine_u8(u8), affine_u32(u32), affine_i16(i16), affine_i32(i32), affine_i64(i64),
    }
}

pub mod ops {
    use super::{BF16, F8E4M3, F16};

    fill_fns! {
        fill_u8(u8), fill_u32(u32), fill_i64(i64), fill_f32(f32), fill_f64(f64),
        fill_f16(F16), fill_bf16(BF16), fill_f8_e4m3(F8E4M3),
    }

    copy2d_fns! {
        copy2d_f32(f32), copy2d_f64(f64), copy2d_u8(u8), copy2d_u32(u32), copy2d_i64(i64),
        copy2d_f16(F16), copy2d_bf16(BF16), copy2d_f8_e4m3(F8E4M3),
    }

    const_set_fns! {
        const_set_f32(f32), const_set_f64(f64), const_set_u8(u8), const_set_u32(u32), const_set_i64(i64),
        const_set_f16(F16), const_set_bf16(BF16), const_set_f8_e4m3(F8E4M3),
    }

    sort_fns! {
        asort_asc_bf16(BF16), asort_desc_bf16(BF16), asort_asc_f16(F16), asort_desc_f16(F16),
        asort_asc_f32(f32), asort_desc_f32(f32), asort_asc_f64(f64), asort_desc_f64(f64),
        asort_asc_u8(u8), asort_desc_u8(u8), asort_asc_u32(u32), asort_desc_u32(u32),
        asort_asc_i64(i64), asort_desc_i64(i64),
    }

    cast_fns! {
    cast_bf16_bf16(BF16, BF16), cast_f8_e4m3_f8_e4m3(F8E4M3, F8E4M3),
    cast_bf16_u32(BF16, u32), cast_bf16_f32(BF16, f32), cast_bf16_f64(BF16, f64),
    cast_u8_bf16(u8, BF16), cast_u32_bf16(u32, BF16), cast_f32_bf16(f32, BF16), cast_f64_bf16(f64, BF16),
    cast_bf16_u8(BF16, u8), cast_bf16_f16(BF16, F16), cast_f16_bf16(F16, BF16),
    cast_f8_e4m3_f32(F8E4M3, f32), cast_f32_f8_e4m3(f32, F8E4M3), cast_f8_e4m3_u8(F8E4M3, u8),
    cast_f8_e4m3_f16(F8E4M3, F16), cast_f8_e4m3_f64(F8E4M3, f64), cast_f16_f8_e4m3(F16, F8E4M3),
    cast_f64_f8_e4m3(f64, F8E4M3), cast_u8_f8_e4m3(u8, F8E4M3), cast_i32_f8_e4m3(i32, F8E4M3),
    cast_f8_e4m3_i32(F8E4M3, i32), cast_f8_e4m3_bf16(F8E4M3, BF16), cast_bf16_f8_e4m3(BF16, F8E4M3),
    cast_f16_f16(F16, F16), cast_f16_u8(F16, u8), cast_f16_u32(F16, u32), cast_f16_f32(F16, f32),
    cast_f16_f64(F16, f64), cast_u8_f16(u8, F16), cast_u32_f16(u32, F16), cast_f32_f16(f32, F16),
    cast_f64_f16(f64, F16), cast_u32_u32(u32, u32), cast_u32_u8(u32, u8), cast_u32_i64(u32, i64),
    cast_u32_f32(u32, f32), cast_u32_f64(u32, f64), cast_u8_u32(u8, u32), cast_u8_u8(u8, u8),
    cast_u8_i64(u8, i64), cast_u8_f32(u8, f32), cast_u8_f64(u8, f64), cast_i64_u32(i64, u32),
    cast_i64_u8(i64, u8), cast_i64_i64(i64, i64), cast_i64_f32(i64, f32), cast_i64_f64(i64, f64),
    cast_f32_u8(f32, u8), cast_f32_u32(f32, u32), cast_f32_i64(f32, i64), cast_f32_f32(f32, f32),
    cast_f32_f64(f32, f64), cast_f64_u8(f64, u8), cast_f64_u32(f64, u32), cast_f64_i64(f64, i64),
    cast_f64_f32(f64, f32), cast_f64_f64(f64, f64),
    }

    binary_fns! {
    badd_bf16(BF16, BF16), bdiv_bf16(BF16, BF16), bmul_bf16(BF16, BF16), bsub_bf16(BF16, BF16),
    bmaximum_bf16(BF16, BF16), bminimum_bf16(BF16, BF16), eq_bf16(BF16, u8), ne_bf16(BF16, u8), lt_bf16(BF16, u8), le_bf16(BF16, u8), gt_bf16(BF16, u8), ge_bf16(BF16, u8),
    badd_f8_e4m3(F8E4M3, F8E4M3), bdiv_f8_e4m3(F8E4M3, F8E4M3), bmul_f8_e4m3(F8E4M3, F8E4M3), bsub_f8_e4m3(F8E4M3, F8E4M3),
    bmaximum_f8_e4m3(F8E4M3, F8E4M3), bminimum_f8_e4m3(F8E4M3, F8E4M3), eq_f8_e4m3(F8E4M3, u8), ne_f8_e4m3(F8E4M3, u8), lt_f8_e4m3(F8E4M3, u8), le_f8_e4m3(F8E4M3, u8), gt_f8_e4m3(F8E4M3, u8), ge_f8_e4m3(F8E4M3, u8),
    badd_f16(F16, F16), bdiv_f16(F16, F16), bmul_f16(F16, F16), bsub_f16(F16, F16), bmaximum_f16(F16, F16), bminimum_f16(F16, F16), eq_f16(F16, u8), ne_f16(F16, u8), lt_f16(F16, u8), le_f16(F16, u8), gt_f16(F16, u8), ge_f16(F16, u8),
    badd_f32(f32, f32), badd_f64(f64, f64), badd_u8(u8, u8), badd_u32(u32, u32), badd_i64(i64, i64),
    bdiv_f32(f32, f32), bdiv_f64(f64, f64), bdiv_u8(u8, u8), bdiv_u32(u32, u32), bdiv_i64(i64, i64),
    bmul_f32(f32, f32), bmul_f64(f64, f64), bmul_u8(u8, u8), bmul_u32(u32, u32), bmul_i64(i64, i64),
    bsub_f32(f32, f32), bsub_f64(f64, f64), bsub_u8(u8, u8), bsub_u32(u32, u32), bsub_i64(i64, i64),
    bminimum_f32(f32, f32), bminimum_f64(f64, f64), bminimum_u8(u8, u8), bminimum_u32(u32, u32), bminimum_i64(i64, i64),
    bmaximum_f32(f32, f32), bmaximum_f64(f64, f64), bmaximum_u8(u8, u8), bmaximum_u32(u32, u32), bmaximum_i64(i64, i64),
    eq_f32(f32, u8), eq_f64(f64, u8), eq_u8(u8, u8), eq_u32(u32, u8), eq_i64(i64, u8),
    ne_f32(f32, u8), ne_f64(f64, u8), ne_u8(u8, u8), ne_u32(u32, u8), ne_i64(i64, u8),
    lt_f32(f32, u8), lt_f64(f64, u8), lt_u8(u8, u8), lt_u32(u32, u8), lt_i64(i64, u8),
    le_f32(f32, u8), le_f64(f64, u8), le_u8(u8, u8), le_u32(u32, u8), le_i64(i64, u8),
    gt_f32(f32, u8), gt_f64(f64, u8), gt_u8(u8, u8), gt_u32(u32, u8), gt_i64(i64, u8),
    ge_f32(f32, u8), ge_f64(f64, u8), ge_u8(u8, u8), ge_u32(u32, u8), ge_i64(i64, u8),
    }

    unary_fns! {
    ucopy_bf16(BF16), uneg_bf16(BF16), urecip_bf16(BF16), uexp_bf16(BF16), ulog_bf16(BF16), usin_bf16(BF16), ucos_bf16(BF16), utanh_bf16(BF16), uerf_bf16(BF16), uceil_bf16(BF16), ufloor_bf16(BF16), uround_bf16(BF16), unormcdf_bf16(BF16), uabs_bf16(BF16), usqr_bf16(BF16), usqrt_bf16(BF16), ugelu_bf16(BF16), ugelu_erf_bf16(BF16), urelu_bf16(BF16), usilu_bf16(BF16), usign_bf16(BF16), usigmoid_bf16(BF16),
    ucopy_f8_e4m3(F8E4M3), uneg_fp8_e4m3(F8E4M3), urecip_fp8_e4m3(F8E4M3), uexp_fp8_e4m3(F8E4M3), ulog_fp8_e4m3(F8E4M3), usin_fp8_e4m3(F8E4M3), ucos_fp8_e4m3(F8E4M3), utanh_fp8_e4m3(F8E4M3), uerf_fp8_e4m3(F8E4M3), uceil_fp8_e4m3(F8E4M3), ufloor_fp8_e4m3(F8E4M3), uround_fp8_e4m3(F8E4M3), unormcdf_fp8_e4m3(F8E4M3), uabs_fp8_e4m3(F8E4M3), usqr_fp8_e4m3(F8E4M3), usqrt_fp8_e4m3(F8E4M3), ugelu_fp8_e4m3(F8E4M3), ugelu_erf_fp8_e4m3(F8E4M3), urelu_fp8_e4m3(F8E4M3), usilu_fp8_e4m3(F8E4M3), usign_fp8_e4m3(F8E4M3), usigmoid_fp8_e4m3(F8E4M3),
    ucopy_f16(F16), uneg_f16(F16), urecip_f16(F16), uexp_f16(F16), ulog_f16(F16), usin_f16(F16), ucos_f16(F16), utanh_f16(F16), uerf_f16(F16), uceil_f16(F16), ufloor_f16(F16), uround_f16(F16), unormcdf_f16(F16), uabs_f16(F16), usqr_f16(F16), usqrt_f16(F16), ugelu_f16(F16), ugelu_erf_f16(F16), urelu_f16(F16), usilu_f16(F16), usign_f16(F16), usigmoid_f16(F16),
    ucopy_u8(u8), ucopy_u32(u32), ucopy_i64(i64), ucopy_f32(f32), ucopy_f64(f64), uneg_f32(f32), uneg_f64(f64), urecip_f32(f32), urecip_f64(f64), uexp_f32(f32), uexp_f64(f64), ulog_f32(f32), ulog_f64(f64), usin_f32(f32), usin_f64(f64), ucos_f32(f32), ucos_f64(f64), utanh_f32(f32), utanh_f64(f64), uerf_f32(f32), uerf_f64(f64), uceil_f32(f32), uceil_f64(f64), ufloor_f32(f32), ufloor_f64(f64), uround_f32(f32), uround_f64(f64), unormcdf_f32(f32), unormcdf_f64(f64), uabs_f32(f32), uabs_f64(f64), usqr_f32(f32), usqr_f64(f64), usqrt_f32(f32), usqrt_f64(f64), ugelu_f32(f32), ugelu_f64(f64), ugelu_erf_f32(f32), ugelu_erf_f64(f64), urelu_f32(f32), urelu_f64(f64), usilu_f32(f32), usilu_f64(f64), usign_f32(f32), usign_f64(f64), usigmoid_f32(f32), usigmoid_f64(f64),
    }

    unary_param_fns! {
    uelu_bf16(BF16), upowf_bf16(BF16), uelu_fp8_e4m3(F8E4M3), upowf_fp8_e4m3(F8E4M3),
    uelu_f16(F16), upowf_f16(F16), uelu_f32(f32), uelu_f64(f64), upowf_f32(f32), upowf_f64(f64),
    }

    where_fns! {
    where_i64_bf16(i64, BF16), where_u32_bf16(u32, BF16), where_u8_bf16(u8, BF16),
    where_i16_fp8_e4m3(i16, F8E4M3), where_i32_fp8_e4m3(i32, F8E4M3), where_i64_fp8_e4m3(i64, F8E4M3), where_u32_fp8_e4m3(u32, F8E4M3), where_u8_fp8_e4m3(u8, F8E4M3),
    where_i64_f16(i64, F16), where_u32_f16(u32, F16), where_u8_f16(u8, F16),
    where_i64_f32(i64, f32), where_i64_f64(i64, f64), where_i64_u8(i64, u8), where_i64_u32(i64, u32), where_i64_i64(i64, i64),
    where_u32_f32(u32, f32), where_u32_f64(u32, f64), where_u32_u8(u32, u8), where_u32_u32(u32, u32), where_u32_i64(u32, i64),
    where_u8_f32(u8, f32), where_u8_f64(u8, f64), where_u8_u8(u8, u8), where_u8_u32(u8, u32), where_u8_i64(u8, i64),
    }
}

pub mod quant {
    use super::c_void;

    dequant_vec_fns! {
    dequantize_mul_mat_vec_q4_0_cuda, dequantize_mul_mat_vec_q4_1_cuda, dequantize_mul_mat_vec_q5_0_cuda,
    dequantize_mul_mat_vec_q5_1_cuda, dequantize_mul_mat_vec_q8_0_cuda, dequantize_mul_mat_vec_q2_k,
    dequantize_mul_mat_vec_q3_k, dequantize_mul_mat_vec_q4_k, dequantize_mul_mat_vec_q5_k, dequantize_mul_mat_vec_q6_k,
    }

    mat_vec_q8_1_fns! {
    mul_mat_vec_q4_0_q8_1_cuda1, mul_mat_vec_q4_1_q8_1_cuda1, mul_mat_vec_q5_0_q8_1_cuda1, mul_mat_vec_q5_1_q8_1_cuda1, mul_mat_vec_q8_0_q8_1_cuda1, mul_mat_vec_q2_K_q8_1_cuda1, mul_mat_vec_q3_K_q8_1_cuda1, mul_mat_vec_q4_K_q8_1_cuda1, mul_mat_vec_q5_K_q8_1_cuda1, mul_mat_vec_q6_K_q8_1_cuda1,
    mul_mat_vec_q4_0_q8_1_cuda2, mul_mat_vec_q4_1_q8_1_cuda2, mul_mat_vec_q5_0_q8_1_cuda2, mul_mat_vec_q5_1_q8_1_cuda2, mul_mat_vec_q8_0_q8_1_cuda2, mul_mat_vec_q2_K_q8_1_cuda2, mul_mat_vec_q3_K_q8_1_cuda2, mul_mat_vec_q4_K_q8_1_cuda2, mul_mat_vec_q5_K_q8_1_cuda2, mul_mat_vec_q6_K_q8_1_cuda2,
    mul_mat_vec_q4_0_q8_1_cuda3, mul_mat_vec_q4_1_q8_1_cuda3, mul_mat_vec_q5_0_q8_1_cuda3, mul_mat_vec_q5_1_q8_1_cuda3, mul_mat_vec_q8_0_q8_1_cuda3, mul_mat_vec_q2_K_q8_1_cuda3, mul_mat_vec_q3_K_q8_1_cuda3, mul_mat_vec_q4_K_q8_1_cuda3, mul_mat_vec_q5_K_q8_1_cuda3, mul_mat_vec_q6_K_q8_1_cuda3,
    mul_mat_vec_q4_0_q8_1_cuda4, mul_mat_vec_q4_1_q8_1_cuda4, mul_mat_vec_q5_0_q8_1_cuda4, mul_mat_vec_q5_1_q8_1_cuda4, mul_mat_vec_q8_0_q8_1_cuda4, mul_mat_vec_q2_K_q8_1_cuda4, mul_mat_vec_q3_K_q8_1_cuda4, mul_mat_vec_q4_K_q8_1_cuda4, mul_mat_vec_q5_K_q8_1_cuda4, mul_mat_vec_q6_K_q8_1_cuda4,
    mul_mat_vec_q4_0_q8_1_cuda5, mul_mat_vec_q4_1_q8_1_cuda5, mul_mat_vec_q5_0_q8_1_cuda5, mul_mat_vec_q5_1_q8_1_cuda5, mul_mat_vec_q8_0_q8_1_cuda5, mul_mat_vec_q2_K_q8_1_cuda5, mul_mat_vec_q3_K_q8_1_cuda5, mul_mat_vec_q4_K_q8_1_cuda5, mul_mat_vec_q5_K_q8_1_cuda5, mul_mat_vec_q6_K_q8_1_cuda5,
    mul_mat_vec_q4_0_q8_1_cuda6, mul_mat_vec_q4_1_q8_1_cuda6, mul_mat_vec_q5_0_q8_1_cuda6, mul_mat_vec_q5_1_q8_1_cuda6, mul_mat_vec_q8_0_q8_1_cuda6, mul_mat_vec_q2_K_q8_1_cuda6, mul_mat_vec_q3_K_q8_1_cuda6, mul_mat_vec_q4_K_q8_1_cuda6, mul_mat_vec_q5_K_q8_1_cuda6, mul_mat_vec_q6_K_q8_1_cuda6,
    mul_mat_vec_q4_0_q8_1_cuda7, mul_mat_vec_q4_1_q8_1_cuda7, mul_mat_vec_q5_0_q8_1_cuda7, mul_mat_vec_q5_1_q8_1_cuda7, mul_mat_vec_q8_0_q8_1_cuda7, mul_mat_vec_q2_K_q8_1_cuda7, mul_mat_vec_q3_K_q8_1_cuda7, mul_mat_vec_q4_K_q8_1_cuda7, mul_mat_vec_q5_K_q8_1_cuda7, mul_mat_vec_q6_K_q8_1_cuda7,
    mul_mat_vec_q4_0_q8_1_cuda8, mul_mat_vec_q4_1_q8_1_cuda8, mul_mat_vec_q5_0_q8_1_cuda8, mul_mat_vec_q5_1_q8_1_cuda8, mul_mat_vec_q8_0_q8_1_cuda8, mul_mat_vec_q2_K_q8_1_cuda8, mul_mat_vec_q3_K_q8_1_cuda8, mul_mat_vec_q4_K_q8_1_cuda8, mul_mat_vec_q5_K_q8_1_cuda8, mul_mat_vec_q6_K_q8_1_cuda8,
    }

    unsafe extern "C" {
        pub fn quantize_q8_1(x: *const f32, y: *mut c_void, kx: i32, kx_padded: i32);

        pub fn indexed_moe_forward_q2k_q8_1(x: *const c_void, w: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32, dst: *mut f32, ncols: i32, nrows: i32, ne00: i32, ne11: i32);
        pub fn indexed_moe_forward_q3k_q8_1(x: *const c_void, w: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32, dst: *mut f32, ncols: i32, nrows: i32, ne00: i32, ne11: i32);
        pub fn indexed_moe_forward_q4k_q8_1(x: *const c_void, w: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32, dst: *mut f32, ncols: i32, nrows: i32, ne00: i32, ne11: i32);
        pub fn indexed_moe_forward_q5k_q8_1(x: *const c_void, w: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32, dst: *mut f32, ncols: i32, nrows: i32, ne00: i32, ne11: i32);
        pub fn indexed_moe_forward_q6k_q8_1(x: *const c_void, w: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32, dst: *mut f32, ncols: i32, nrows: i32, ne00: i32, ne11: i32);
        pub fn indexed_moe_forward_q8_0_q8_1(x: *const c_void, w: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32, dst: *mut f32, ncols: i32, nrows: i32, ne00: i32, ne11: i32);
    }
}
