#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorrectnessTask {
    pub domain: &'static str,
    pub spec: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectnessSuite {
    pub id: &'static str,
    pub threshold: f64,
    pub max_samples: u32,
    pub timeout_secs: u64,
    pub tasks: &'static [CorrectnessTask],
}

impl CorrectnessSuite {
    pub fn task_spec(&self) -> String {
        self.tasks
            .iter()
            .map(|task| task.spec)
            .collect::<Vec<_>>()
            .join(",")
    }
}

pub fn default_suite() -> &'static CorrectnessSuite {
    &OA_FAST_V1
}

// ponytail: owned fast gate; keep endpoint/LiteLLM-compatible generative tasks only.
pub const OA_FAST_V1: CorrectnessSuite = CorrectnessSuite {
    id: "oa-fast-v1",
    threshold: 0.60,
    max_samples: 20,
    timeout_secs: 600,
    tasks: &[
        CorrectnessTask {
            domain: "math",
            spec: "gsm8k|0",
        },
        CorrectnessTask {
            domain: "instruction_following",
            spec: "ifeval|0",
        },
        CorrectnessTask {
            domain: "factual_qa",
            spec: "triviaqa|0",
        },
        CorrectnessTask {
            domain: "reading_comprehension",
            spec: "drop|1",
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_spec_is_lighteval_task_list() {
        assert_eq!(OA_FAST_V1.task_spec(), "gsm8k|0,ifeval|0,triviaqa|0,drop|1");
    }
}
