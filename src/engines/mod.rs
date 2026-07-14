use crate::domain::engine::Engine;

pub(crate) mod managed;
mod sglang;
mod vllm;

pub(crate) fn parameter_introspection_script(engine: Engine) -> &'static str {
    match engine {
        Engine::Vllm => vllm::VLLM_ARGPARSE_INTROSPECTION,
        Engine::Sglang => sglang::SGLANG_ARGPARSE_INTROSPECTION,
    }
}
