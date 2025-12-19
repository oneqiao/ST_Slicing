# st_slicer/blocks/render.py
from __future__ import annotations
from typing import List, Set

from .core import FunctionalBlock

#纯渲染，把 line_numbers + line_overrides 变成文本。保持最轻，避免渲染阶段改语义。

def render_block_text(
    block,
    code_lines,
    *,
    normalize_else_only_if: bool = False,
) -> str:
    """
    渲染一个 FunctionalBlock 的源码文本。

    兼容参数：
      - normalize_else_only_if: 旧版接口参数。你现在已经在 pipeline 后处理阶段
        fold_half_empty_ifs_in_block() 做了折叠/清理，因此这里保留参数仅用于兼容，
        默认不再重复改写以避免二次变形。

    同时支持：
      - block.line_overrides: 对特定行进行重写（例如把 IF 头改写成 IF NOT (...) THEN）
    """
    n = len(code_lines)
    lnset = [ln for ln in sorted(set(getattr(block, "line_numbers", []) or [])) if 1 <= ln <= n]
    if not lnset:
        return ""

    overrides = getattr(block, "line_overrides", {}) or {}

    out_lines = []
    for ln in lnset:
        # 优先输出 override 行（如果存在）
        if ln in overrides and isinstance(overrides[ln], str) and overrides[ln].strip():
            out_lines.append(overrides[ln].rstrip("\n"))
        else:
            out_lines.append(code_lines[ln - 1].rstrip("\n"))

    return "\n".join(out_lines).rstrip() + "\n"

