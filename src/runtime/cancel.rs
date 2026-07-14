use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::error::{Error, ErrorKind, ExecutionStage, Result};

#[derive(Clone, Debug, Default)]
pub(crate) struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    ancestors: Vec<Arc<AtomicBool>>,
}

impl CancellationToken {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn child(&self) -> Self {
        let mut ancestors = Vec::with_capacity(self.ancestors.len() + 1);
        ancestors.push(Arc::clone(&self.cancelled));
        ancestors.extend(self.ancestors.iter().cloned());
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            ancestors,
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
            || self
                .ancestors
                .iter()
                .any(|cancelled| cancelled.load(Ordering::SeqCst))
    }

    pub(crate) fn register_os_signals(&self) -> Result<()> {
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            signal_hook::flag::register(signal, Arc::clone(&self.cancelled)).map_err(|source| {
                Error::new(
                    ErrorKind::Io,
                    Some(ExecutionStage::Preflight),
                    format!("failed to register signal handler for signal {signal}"),
                )
                .with_operation("register signal handler")
                .with_source(source)
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_cancellation_state() {
        let token = CancellationToken::new();
        let clone = token.clone();

        token.cancel();

        assert!(clone.is_cancelled());
    }

    #[test]
    fn child_cancellation_is_local_but_observes_parent_cancellation() {
        let parent = CancellationToken::new();
        let first = parent.child();
        let second = parent.child();

        first.cancel();
        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
        assert!(!parent.is_cancelled());

        parent.cancel();
        assert!(second.is_cancelled());
    }
}
