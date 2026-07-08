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

// ponytail: owned correctness policy; change this list when we revise the gate.
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
            domain: "commonsense",
            spec: "hellaswag|0",
        },
        CorrectnessTask {
            domain: "truthfulness",
            spec: "truthfulqa:mc|0",
        },
        CorrectnessTask {
            domain: "instruction_following",
            spec: "ifeval|0",
        },
        CorrectnessTask {
            domain: "knowledge",
            spec: "mmlu:abstract_algebra|0",
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_spec_is_lighteval_task_list() {
        assert_eq!(
            OA_FAST_V1.task_spec(),
            "gsm8k|0,hellaswag|0,truthfulqa:mc|0,ifeval|0,mmlu:abstract_algebra|0"
        );
    }
}
