/// Token selected from a processed logits vector.
///
/// # Example
///
/// ```
/// use sampling::SampleOutput;
///
/// let out = SampleOutput::new(7, -0.25, 4.5);
/// assert_eq!(out.token_id(), 7);
/// assert_eq!(out.log_prob(), -0.25);
/// assert_eq!(out.logit(), 4.5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleOutput {
    token_id: u32,
    log_prob: f32,
    logit: f32,
}

impl SampleOutput {
    /// Creates a sampled token output.
    pub fn new(token_id: u32, log_prob: f32, logit: f32) -> Self {
        Self {
            token_id,
            log_prob,
            logit,
        }
    }

    /// Returns the selected token id.
    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    /// Returns the token log probability under the filtered distribution.
    pub fn log_prob(&self) -> f32 {
        self.log_prob
    }

    /// Returns the processed logit for the selected token.
    pub fn logit(&self) -> f32 {
        self.logit
    }
}
