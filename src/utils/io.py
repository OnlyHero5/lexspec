"""Backward-compatibility shim. New code should import from src.utils.jsonl_io / json_io."""
from src.utils.jsonl_io import (
    read_jsonl, read_jsonl_stream, write_jsonl, append_jsonl,
    load_pydantic_list, save_pydantic_list,
)
from src.utils.json_io import read_json, write_json, ensure_dir

__all__ = [
    "read_jsonl", "read_jsonl_stream", "write_jsonl", "append_jsonl",
    "load_pydantic_list", "save_pydantic_list",
    "read_json", "write_json", "ensure_dir",
]
