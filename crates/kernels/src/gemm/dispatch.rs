use runtime::DeviceProps;

/// CUDA SM architecture families with GEMM specializations in this crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmArch {
    /// Ada Lovelace/Hopper-compatible SM 8.9 kernels.
    Sm89,
    /// Blackwell SM 10.0 kernels.
    Sm100,
}

impl SmArch {
    /// Converts packed compute capability values such as `89` and `100` into
    /// dispatchable architecture families.
    pub fn from_packed(sm_arch: i32) -> Option<Self> {
        match sm_arch {
            89 => Some(Self::Sm89),
            100 => Some(Self::Sm100),
            _ => None,
        }
    }
}

/// GEMM kernel variants selected by runtime device capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmKernel {
    /// BF16 GEMM compiled for SM 8.9.
    Bf16Sm89,
    /// BF16 GEMM compiled for SM 10.0.
    Bf16Sm100,
}

/// Selects the BF16 GEMM implementation for a CUDA device.
///
/// Newer architectures use their native kernels. SM 8.9 remains the fallback
/// for nearby architectures until dedicated kernels are added.
pub fn dispatch_bf16_gemm(props: &DeviceProps) -> GemmKernel {
    match SmArch::from_packed(props.sm_arch()) {
        Some(SmArch::Sm100) => GemmKernel::Bf16Sm100,
        Some(SmArch::Sm89) | None => GemmKernel::Bf16Sm89,
    }
}
