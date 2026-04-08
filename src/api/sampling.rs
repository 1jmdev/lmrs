#[derive(Clone, Debug)]
pub enum Sampling {
    Greedy,
    Temperature(TemperatureSampling),
}

#[derive(Clone, Debug)]
pub struct TemperatureSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for TemperatureSampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}
