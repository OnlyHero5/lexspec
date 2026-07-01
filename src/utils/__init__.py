"""
LexSpec 工具包
==========================

LexSpec 全项目共享的工具模块。

公开 API:
  - jsonl_io: JSONL 读写与 Pydantic 模型序列化。
  - json_io:  普通 JSON 读写与目录管理。
  - io:       向后兼容垫片，从 jsonl_io 与 json_io 重新导出
              （现有代码无需修改即可继续工作）。
  - logging:  集中式日志配置，含控制台与文件处理器。所有模块使用
              get_logger(__name__) 以保持格式一致。
"""

from src.utils.jsonl_io import (
    read_jsonl,
    read_jsonl_stream,
    write_jsonl,
    append_jsonl,
    load_pydantic_list,
    save_pydantic_list,
)

from src.utils.json_io import (
    read_json,
    write_json,
    ensure_dir,
)

from src.utils.logging import (
    setup_logging,
    get_logger,
)

__all__ = [
    # I/O —— JSONL
    "read_jsonl",
    "read_jsonl_stream",
    "write_jsonl",
    "append_jsonl",
    "load_pydantic_list",
    "save_pydantic_list",
    # I/O —— JSON
    "read_json",
    "write_json",
    "ensure_dir",
    # 日志
    "setup_logging",
    "get_logger",
]
