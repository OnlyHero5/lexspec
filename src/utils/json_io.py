"""
LexSpec 普通 JSON 与文件系统工具
================================================
单对象 JSON 读写与目录管理。

本模块处理普通（非按行分隔）JSON 文件——通常用于配置、元数据
及小型结构化数据文件。还包含项目通用的 ensure_dir 文件系统工具。

设计决策:
  - 所有路径接受 str | Path，内部使用 pathlib.Path。
  - JSON 读写使用 ensure_ascii=False 以保留 Unicode。
  - 写操作自动创建父目录。
  - 读操作抛出带清晰消息的 FileNotFoundError。
"""

import json
from pathlib import Path

# 所有文件 I/O 使用的默认编码
_ENCODING = "utf-8"


# ============================================================================
# 普通 JSON —— 用于配置、元数据与小型数据文件
# ============================================================================


def read_json(file_path: str | Path) -> dict:
    """
    从文件读取单个 JSON 对象。

    文件必须包含单个 JSON 对象（非 JSONL / 按行分隔格式）。
    用于配置文件、元数据及其他小型结构化数据。

    参数:
        file_path: JSON 文件路径。

    返回:
        从 JSON 文件解析的字典。

    异常:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: 文件不含有效 JSON。
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding=_ENCODING) as fh:
        return json.load(fh)


def write_json(file_path: str | Path, data: dict, indent: int = 2) -> None:
    """
    将字典以美化格式写入 JSON 文件。

    按需创建父目录。使用 ensure_ascii=False 以保留法律文本中的
    Unicode 字符（如章节符号、非英文当事方名称）。默认 2 空格缩进
    使输出便于人工阅读。

    参数:
        file_path: 输出 JSON 文件路径。
        data:      待序列化的字典（须可 JSON 序列化）。
        indent:    缩进空格数（默认 2）。设为 None 可输出紧凑单行格式。

    异常:
        OSError: 无法写入文件。
        TypeError: data 含不可 JSON 序列化的值。
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=_ENCODING) as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)


# ============================================================================
# 文件系统工具
# ============================================================================


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，必要时创建（含所有父目录）。

    幂等——对已存在目录调用为无操作（不报错、不修改）。
    这是对 Path.mkdir 的薄封装，为所有模块提供一致接口。

    参数:
        path: 目录路径，str 或 Path。

    返回:
        指向已确保存在的目录的 Path 对象。

    异常:
        OSError: 无法创建目录（如权限不足，或给定路径已存在同名文件）。
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
