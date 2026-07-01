"""
LexSpec JSONL I/O 工具
================================
JSONL 读写与 Pydantic 序列化辅助函数。

本模块提供规范的 JSONL I/O 层。所有 JSONL 文件访问均通过这些函数——
各模块不直接读写 JSONL 文件。确保流水线中一致的错误处理、编码与路径管理。

设计决策:
  - 所有路径接受 str | Path，内部使用 pathlib.Path。
  - JSONL（JSON Lines）为主要序列化格式——每行一个 JSON 对象，
    适合流式与增量处理。
  - Pydantic 序列化使用 model_dump(mode='json')，确保枚举值
    序列化为字符串而非 Python 枚举对象。
  - 写操作自动创建父目录。
  - 读操作抛出带清晰消息的 FileNotFoundError。
"""

import json
from typing import List, Iterator, Any, TypeVar, Type
from pathlib import Path

# Pydantic 模型类的泛型类型变量
T = TypeVar("T")

# 所有文件 I/O 使用的默认编码
_ENCODING = "utf-8"


# ============================================================================
# JSONL —— 按行分隔的 JSON（主要数据格式）
# ============================================================================


def read_jsonl(file_path: str | Path) -> List[dict]:
    """
    从 JSONL 文件读取全部记录为字典列表。

    文件中每行须为有效 JSON 对象。空行静默跳过。本方法将整个文件
    载入内存，适合中小型数据集。大文件请使用 read_jsonl_stream()。

    参数:
        file_path: JSONL 文件路径。

    返回:
        字典列表，每条 JSONL 记录一个。文件仅含空行时返回空列表。

    异常:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: 任一非空行不是有效 JSON。
    """
    file_path = Path(file_path)  # 规范化为 pathlib.Path

    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records: List[dict] = []
    with open(file_path, "r", encoding=_ENCODING) as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                # 跳过空行——手工编辑的 JSONL 文件中常见
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                # 附加文件与行上下文以便调试
                raise json.JSONDecodeError(
                    f"Invalid JSON at {file_path}:{line_num}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
    return records


def read_jsonl_stream(file_path: str | Path) -> Iterator[dict]:
    """
    逐条流式读取 JSONL 文件记录。

    大文件的内存高效生成器——同一时间仅保留一条记录在内存中。
    处理超出可用 RAM 的数据集时使用。

    参数:
        file_path: JSONL 文件路径。

    生成:
        每条 JSONL 记录对应的字典。

    异常:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: 任一非空行不是有效 JSON。
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with open(file_path, "r", encoding=_ENCODING) as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON at {file_path}:{line_num}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc


def write_jsonl(file_path: str | Path, records: List[dict]) -> None:
    """
    将字典列表写入 JSONL 文件。

    按需创建父目录。若文件已存在则覆盖。每个字典序列化为单行 JSON，
    无额外空白。

    参数:
        file_path: 输出 JSONL 文件路径。
        records:   待写入的字典列表。空列表产生空文件。

    异常:
        OSError: 无法创建或写入文件。
        TypeError: 任一记录含不可 JSON 序列化的数据。
    """
    file_path = Path(file_path)
    # 写入前确保父目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=_ENCODING) as fh:
        for record in records:
            # ensure_ascii=False 保留法律文本中的 Unicode 字符
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(file_path: str | Path, record: dict) -> None:
    """
    向 JSONL 文件追加单条记录。

    按需创建文件与父目录。使用追加模式以保留已有数据。
    在 POSIX 系统上，每次写入为单行时具有线程安全性。

    参数:
        file_path: JSONL 文件路径。
        record:    作为新 JSONL 行追加的单个字典。

    异常:
        OSError: 无法以追加模式打开文件。
        TypeError: 记录含不可 JSON 序列化的数据。
    """
    file_path = Path(file_path)
    # 确保目录链存在；mkdir 在目录已存在时不会抛出异常
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a", encoding=_ENCODING) as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================================
# Pydantic 模型序列化 —— 类型安全的 I/O
# ============================================================================


def load_pydantic_list(file_path: str | Path, model_class: Type[T]) -> List[T]:
    """
    读取 JSONL 文件并将每条记录解析为 Pydantic 模型实例。

    反序列化时使用 model_validate（Pydantic v2）进行校验。
    确保进入系统的全部数据符合模式。

    参数:
        file_path:    JSONL 文件路径。
        model_class:  用于解析每条记录的 Pydantic BaseModel 子类。

    返回:
        经过校验的 Pydantic 模型实例列表。

    异常:
        FileNotFoundError: 文件不存在。
        pydantic.ValidationError: 任一记录模式校验失败。
        json.JSONDecodeError: 任一行不是有效 JSON。
    """
    records = read_jsonl(file_path)
    # model_validate 是 Pydantic v2 中 parse_obj 的替代
    return [model_class.model_validate(record) for record in records]


def save_pydantic_list(file_path: str | Path, instances: List[Any]) -> None:
    """
    将 Pydantic 模型列表序列化到 JSONL 文件。

    使用 model_dump(mode='json')（Pydantic v2）将每个实例转换为
    可 JSON 序列化的字典。mode='json' 参数确保：
      - 枚举值序列化为其字符串值，而非枚举对象
      - datetime 对象（若有）转换为 ISO 8601 字符串
      - Path 对象转换为字符串

    参数:
        file_path: 输出 JSONL 文件路径。
        instances: Pydantic BaseModel 实例列表。

    异常:
        OSError: 无法写入文件。
        AttributeError: 任一项缺少 model_dump（即非 Pydantic 模型）。
    """
    # 使用 Pydantic 序列化将每个模型转为普通字典
    records = [instance.model_dump(mode="json") for instance in instances]
    write_jsonl(file_path, records)
