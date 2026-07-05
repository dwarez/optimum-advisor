#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Oom,
    KvPressure,
    Ready,
    Unknown,
}

pub fn classify_log(log: &str) -> Outcome {
    let lower = log.to_ascii_lowercase();
    if lower.contains("out of memory")
        || lower.contains("torch.outofmemoryerror")
        || lower.contains("cuda oom")
        || lower.contains("cannot allocate memory")
    {
        return Outcome::Oom;
    }
    if lower.contains("kv cache pool is full")
        || lower.contains("not enough kv cache")
        || lower.contains("preempted")
        || lower.contains("retract requests")
    {
        return Outcome::KvPressure;
    }
    if lower.contains("server is fired up")
        || lower.contains("uvicorn running")
        || lower.contains("application startup complete")
        || lower.contains("available kv cache memory")
    {
        return Outcome::Ready;
    }
    Outcome::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_oom_logs() {
        assert_eq!(
            classify_log("torch.OutOfMemoryError: CUDA out of memory"),
            Outcome::Oom
        );
    }

    #[test]
    fn classifies_kv_pressure() {
        assert_eq!(
            classify_log("KV cache pool is full. Retract requests."),
            Outcome::KvPressure
        );
    }
}
