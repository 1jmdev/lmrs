use ops::{cast_bf16_to_f32, reshape};
use tensor::Tensor;

use crate::{Greedy, LogitsProcessor, Result, SamplingStrategy};

use super::SampleOutput;

/// Configuration for a `Sampler`.
///
/// # Example
///
/// ```
/// use sampling::{SamplerConfig, TopK};
///
/// # fn main() -> sampling::Result<()> {
/// let config = SamplerConfig::new(Box::new(TopK::new(8)?)).with_seed(123);
/// assert_eq!(config.seed(), 123);
/// # Ok(())
/// # }
/// ```
pub struct SamplerConfig {
    strategy: Box<dyn SamplingStrategy>,
    seed: u64,
}

impl SamplerConfig {
    /// Creates a configuration with the provided strategy.
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::{SamplerConfig, Greedy};
    ///
    /// let config = SamplerConfig::new(Box::new(Greedy));
    /// ```
    pub fn new(strategy: Box<dyn SamplingStrategy>) -> Self {
        Self { strategy, seed: 1 }
    }

    /// Sets the deterministic RNG seed used by stochastic strategies.
    ///
    /// # Example
    ///
    /// ```
    /// # use sampling::{SamplerConfig, Greedy};
    /// # let config = SamplerConfig::new(Box::new(Greedy)).with_seed(42);
    /// assert_eq!(config.seed(), 42);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed.max(1);
        self
    }

    /// Returns the configured seed.
    ///
    /// # Example
    ///
    /// ```
    /// # use sampling::{SamplerConfig, Greedy};
    /// # let config = SamplerConfig::new(Box::new(Greedy));
    /// assert_eq!(config.seed(), 1);
    /// ```
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::new(Box::new(Greedy)).with_seed(1)
    }
}

/// Chains logits processors and selects a token from CUDA BF16 logits.
///
/// Processor tensor math runs entirely on CUDA via the `ops` and `kernels`
/// crates. The final logits are cast from BF16 to host F32 and fed to the
/// sampling strategy which runs on CPU.
///
/// # Example
///
/// ```no_run
/// use ops::cast_bf16_to_f32;
/// use runtime::CudaContext;
/// use sampling::{Sampler, SamplerConfig};
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> sampling::Result<()> {
/// let context = CudaContext::new(0)?;
/// let logits = copy_h2d(
///     &context,
///     Shape::new([3])?,
///     DType::BF16,
///     &[
///         half::bf16::from_f32(1.0).to_bits(),
///         half::bf16::from_f32(5.0).to_bits(),
///         half::bf16::from_f32(2.0).to_bits(),
///     ],
/// )?;
/// let mut sampler = Sampler::new(SamplerConfig::default());
/// let out = sampler.sample(&logits, &[])?;
/// assert_eq!(out.token_id(), 1);
/// # Ok(())
/// # }
/// ```
pub struct Sampler {
    processors: Vec<Box<dyn LogitsProcessor>>,
    strategy: Box<dyn SamplingStrategy>,
    rng: u64,
}

impl Sampler {
    /// Creates a sampler from explicit configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::{Sampler, SamplerConfig, Greedy};
    ///
    /// let sampler = Sampler::new(SamplerConfig::new(Box::new(Greedy)));
    /// ```
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            processors: Vec::new(),
            strategy: config.strategy,
            rng: config.seed,
        }
    }

    /// Adds a logits processor to the end of the processing chain.
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::{Sampler, Temperature};
    ///
    /// # fn main() -> sampling::Result<()> {
    /// let mut sampler = Sampler::default();
    /// sampler.push_processor(Temperature::new(1.0)?);
    /// assert_eq!(sampler.processor_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn push_processor<P>(&mut self, processor: P)
    where
        P: LogitsProcessor + 'static,
    {
        self.processors.push(Box::new(processor));
    }

    /// Applies processors and samples a token.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ops::cast_bf16_to_f32;
    /// # use runtime::CudaContext;
    /// # use sampling::{Sampler, SamplerConfig};
    /// # use tensor::{DType, Shape, copy_h2d};
    /// # fn main() -> sampling::Result<()> {
    /// # let context = CudaContext::new(0)?;
    /// let logits = copy_h2d(
    ///     &context,
    ///     Shape::new([3])?,
    ///     DType::BF16,
    ///     &[
    ///         half::bf16::from_f32(1.0).to_bits(),
    ///         half::bf16::from_f32(5.0).to_bits(),
    ///         half::bf16::from_f32(2.0).to_bits(),
    ///     ],
    /// )?;
    /// let mut sampler = Sampler::default();
    /// let out = sampler.sample(&logits, &[])?;
    /// assert_eq!(out.token_id(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sample(&mut self, logits: &Tensor, history: &[u32]) -> Result<SampleOutput> {
        let mut processed = reshape(logits, [logits.numel()])?;
        for processor in &self.processors {
            processed = processor.process(&processed, history)?;
            processed = reshape(&processed, [processed.numel()])?;
        }
        let values = cast_bf16_to_f32(&processed)?;
        self.strategy.sample(&values, &mut self.rng)
    }

    /// Returns the number of processors in the chain.
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::Sampler;
    ///
    /// let sampler = Sampler::default();
    /// assert_eq!(sampler.processor_count(), 0);
    /// ```
    pub fn processor_count(&self) -> usize {
        self.processors.len()
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(SamplerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use runtime::CudaContext;
    use tensor::{DType, Shape, copy_h2d};

    use crate::{Result, Sampler};

    #[test]
    fn greedy_sampler_picks_largest_logit() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let logits = copy_h2d(
            &ctx,
            Shape::new([3])?,
            DType::BF16,
            &[
                bf16::from_f32(0.0).to_bits(),
                bf16::from_f32(3.0).to_bits(),
                bf16::from_f32(1.0).to_bits(),
            ],
        )?;
        let mut sampler = Sampler::default();
        assert_eq!(sampler.sample(&logits, &[])?.token_id(), 1);
        Ok(())
    }
}
