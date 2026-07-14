pub(super) const VLLM_ARGPARSE_INTROSPECTION: &str = r#"
import argparse
from vllm.entrypoints.openai.cli_args import make_arg_parser

parser = make_arg_parser(argparse.ArgumentParser(prog="vllm serve"))
flag_types = (
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
    argparse._StoreConstAction,
    argparse.BooleanOptionalAction,
)
for action in parser._actions:
    kind = "flag" if isinstance(action, flag_types) or action.nargs == 0 else "value"
    for option in action.option_strings:
        if option.startswith("--"):
            print(f"{option}\t{kind}")
"#;
