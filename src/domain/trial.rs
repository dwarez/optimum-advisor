#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub parallelism: Parallelism,
    pub memory: MemoryBudget,
    pub scheduler: SchedulerBudget,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parallelism {
    pub tensor: usize,
    pub pipeline: usize,
    pub data: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryBudget {
    pub fraction: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerBudget {
    pub prefill_token_budget: u32,
    pub max_running_requests: u32,
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            parallelism: Parallelism::default(),
            memory: MemoryBudget { fraction: 0.90 },
            scheduler: SchedulerBudget {
                prefill_token_budget: 8192,
                max_running_requests: 256,
            },
        }
    }
}

impl Default for Parallelism {
    fn default() -> Self {
        Self {
            tensor: 1,
            pipeline: 1,
            data: 1,
        }
    }
}

impl Candidate {
    pub fn clamp_to_gpus(&mut self, gpus: usize) {
        self.parallelism.tensor = self.parallelism.tensor.min(gpus).max(1);
        self.parallelism.pipeline = self.parallelism.pipeline.min(gpus).max(1);
        self.parallelism.data = self.parallelism.data.max(1);
    }
}

pub fn next_tensor_parallelism(current: usize, gpus: usize) -> Option<usize> {
    ((current + 1)..=gpus).find(|candidate| gpus % candidate == 0)
}
