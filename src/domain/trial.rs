use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    pub parallelism: Parallelism,
    pub memory: MemoryBudget,
    pub scheduler: SchedulerBudget,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Parallelism {
    pub tensor: usize,
    pub pipeline: usize,
    pub data: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MemoryBudget {
    pub fraction: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SchedulerBudget {
    pub prefill_token_budget: u32,
    pub max_running_requests: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CandidateSweep {
    pub tensor_parallelism: Vec<usize>,
    pub memory_fraction: Vec<f32>,
    pub prefill_token_budget: Vec<u32>,
    pub max_running_requests: Vec<u32>,
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

impl CandidateSweep {
    pub fn candidates(&self, base: &Candidate, gpus: usize) -> Vec<Candidate> {
        let tensor_parallelism = values_or(&self.tensor_parallelism, base.parallelism.tensor);
        let memory_fraction = values_or(&self.memory_fraction, base.memory.fraction);
        let prefill_token_budget = values_or(
            &self.prefill_token_budget,
            base.scheduler.prefill_token_budget,
        );
        let max_running_requests = values_or(
            &self.max_running_requests,
            base.scheduler.max_running_requests,
        );
        let mut candidates = Vec::new();

        for tensor in tensor_parallelism {
            for memory in &memory_fraction {
                for prefill in &prefill_token_budget {
                    for max_running in &max_running_requests {
                        let mut candidate = base.clone();
                        candidate.parallelism.tensor = tensor;
                        candidate.memory.fraction = *memory;
                        candidate.scheduler.prefill_token_budget = *prefill;
                        candidate.scheduler.max_running_requests = *max_running;
                        candidate.clamp_to_gpus(gpus);
                        if !candidates.contains(&candidate) {
                            candidates.push(candidate);
                        }
                    }
                }
            }
        }

        candidates
    }
}

pub fn next_tensor_parallelism(current: usize, gpus: usize) -> Option<usize> {
    ((current + 1)..=gpus).find(|candidate| gpus % candidate == 0)
}

fn values_or<T: Clone>(values: &[T], fallback: T) -> Vec<T> {
    if values.is_empty() {
        vec![fallback]
    } else {
        values.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_builds_candidate_product() {
        let sweep = CandidateSweep {
            tensor_parallelism: vec![1, 2],
            memory_fraction: vec![0.80, 0.90],
            prefill_token_budget: Vec::new(),
            max_running_requests: vec![64],
        };

        let candidates = sweep.candidates(&Candidate::default(), 2);

        assert_eq!(candidates.len(), 4);
        assert!(candidates
            .iter()
            .any(|candidate| candidate.parallelism.tensor == 2
                && candidate.memory.fraction == 0.90
                && candidate.scheduler.max_running_requests == 64));
    }
}
