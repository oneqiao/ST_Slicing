# st_slicer/blocks/render.py
from __future__ import annotations
from typing import List, Set

from .core import FunctionalBlock

#纯渲染，把 line_numbers + line_overrides 变成文本。保持最轻，避免渲染阶段改语义。

def render_block_text(block: FunctionalBlock, code_lines: List[str]) -> str:
    n = len(code_lines)
    lnset: Set[int] = {ln for ln in block.line_numbers if 1 <= ln <= n}
    if not lnset:
        return ""

    out_lines: List[str] = []
    for ln in sorted(lnset):
        if ln in block.line_overrides:
            out_lines.append(block.line_overrides[ln].rstrip("\n"))
        else:
            out_lines.append(code_lines[ln - 1].rstrip("\n"))

    return "\n".join(out_lines).rstrip() + "\n"
