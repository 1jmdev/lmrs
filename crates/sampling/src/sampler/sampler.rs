use candle_core::{DType, Device, Result, Tensor};

use crate::{LogitsProcessor, SamplingStrategy, strategies::Greedy};

use super::SampleOutput;

/// Configuration for a `Sampler`.
///
/// # Example
///
/// ```
/// use sampling::{SamplerConfig, TopK};
///
/// # fn main() -> candle_core::Result<()> {
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
    pub fn new(strategy: Box<dyn SamplingStrategy>) -> Self {
        Self { strategy, seed: 1 }
    }

    /// Sets the deterministic RNG seed used by stochastic strategies.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed.max(1);
        self
    }

    /// Returns the configured seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::new(Box::new(Greedy)).with_seed(1)
    }
}

/// Chains logits processors and selects a token from model logits.
///
/// Processor tensor math runs on the input tensor device. Greedy BF16 CUDA
/// logits use the workspace CUDA argmax kernel directly; filtered stochastic
/// strategies materialize the final logits slice on the host only at the scalar
/// selection boundary.
///
/// # Example
///
/// ```
/// use candle_core::{Device, Tensor};
/// use sampling::{Sampler, SamplerConfig, Temperature};
///
/// # fn main() -> candle_core::Result<()> {
/// let logits = Tensor::from_vec(vec![1.0_f32, 5.0, 2.0], 3, &Device::Cpu)?;
/// let mut sampler = Sampler::new(SamplerConfig::default());
/// sampler.push_processor(Temperature::new(1.0)?);
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
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            processors: Vec::new(),
            strategy: config.strategy,
            rng: config.seed,
        }
    }

    /// Adds a logits processor to the end of the processing chain.
    pub fn push_processor<P>(&mut self, processor: P)
    where
        P: LogitsProcessor + 'static,
    {
        self.processors.push(Box::new(processor));
    }

    /// Applies processors and samples a token.
    pub fn sample(&mut self, logits: &Tensor, history: &[u32]) -> Result<SampleOutput> {
        let mut processed = logits.flatten_all()?;
        for processor in &self.processors {
            processed = processor.process(&processed, history)?.flatten_all()?;
        }

        if self.processors.is_empty()
            && processed.dtype() == DType::BF16
            && matches!(processed.device(), Device::Cuda(_))
        {
            let argmax = processed.to_dtype(DType::F32)?.argmax(0)?;
            let token = argmax.to_scalar::<u32>()?;
            return Ok(SampleOutput::new(token, 0.0, f32::NAN));
        }

        let values = processed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        self.strategy.sample(&values, &mut self.rng)
    }

    /// Returns the number of processors in the chain.
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
    use candle_core::{Device, Tensor};

    use crate::{RepetitionPenalty, Sampler, SamplerConfig, Temperature, TopK};

    #[test]
    fn greedy_sampler_picks_largest_logit() -> candle_core::Result<()> {
        let logits = Tensor::from_vec(vec![0.0_f32, 3.0, 1.0], 3, &Device::Cpu)?;
        let mut sampler = Sampler::default();
        assert_eq!(sampler.sample(&logits, &[])?.token_id(), 1);
        Ok(())
    }

    #[test]
    fn processors_are_chained_before_sampling() -> candle_core::Result<()> {
        let logits = Tensor::from_vec(vec![4.0_f32, 3.0], 2, &Device::Cpu)?;
        let mut sampler = Sampler::default();
        sampler.push_processor(Temperature::new(2.0)?);
        sampler.push_processor(RepetitionPenalty::new(10.0)?);
        assert_eq!(sampler.sample(&logits, &[0])?.token_id(), 1);
        Ok(())
    }

    #[test]
    fn top_k_limits_candidates() -> candle_core::Result<()> {
        let logits = Tensor::from_vec(vec![0.0_f32, 5.0, 4.0], 3, &Device::Cpu)?;
        let mut sampler = Sampler::new(SamplerConfig::new(Box::new(TopK::new(2)?)).with_seed(10));
        let token = sampler.sample(&logits, &[])?.token_id();
        assert!(token == 1 || token == 2);
        Ok(())
    }
}
