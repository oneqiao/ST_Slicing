# st_slicer/block_context.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict
import re

from st_slicer.blocks.types import FunctionalBlock
from st_slicer.blocks.postprocess import collect_vars_in_block
from st_slicer.blocks.render import render_block_text

@dataclass
class CompletedBlock:
    """
    表示一个已经“结构补全”的功能块：
      - code: 完整的 ST 程序文本（含 PROGRAM/VAR/END_PROGRAM）
      - line_numbers: 源程序中涉及到的行号
      - vars_used: 该块中用到的变量名集合
    """
    block_index: int
    code: str
    line_numbers: List[int]
    vars_used: Set[str]


def _classify_var_storage(sym) -> str:
    """
    根据符号表中的 symbol 判断变量的存储类别：
      - VAR_INPUT / VAR_OUTPUT / VAR（或其他）
    你目前的 symbol 好像有 type / role，可以兼容 storage/role 两种字段。
    """
    storage = getattr(sym, "storage", None)
    if storage is None:
        storage = getattr(sym, "role", None)

    if storage is None:
        # 默认按普通 VAR 处理
        return "VAR"

    storage_upper = str(storage).upper()
    if "INPUT" in storage_upper:
        return "VAR_INPUT"
    if "OUTPUT" in storage_upper:
        return "VAR_OUTPUT"
    return "VAR"

def classify_variable(symbol):
    """
    根据 symbol 判断变量应该放在什么声明区。
    返回:
        ("fb_instance", type_name)   → 需要放 VAR 中的 FB 实例
        ("normal_var", type_name)    → 普通变量
        ("ignore", None)             → function，不需要声明
    """
    role = getattr(symbol, "role", None)
    typ  = getattr(symbol, "type", None)

    # FB 实例
    if role in ("FB", "FUNCTION_BLOCK"):
        return ("fb_instance", typ)

    # function 调用：不出现在变量区
    if role in ("FUNCTION",):
        return ("ignore", None)
        
    return ("normal_var", typ)

def build_completed_block(
    block: FunctionalBlock,
    pou_name: str,
    pou_symtab,
    code_lines: List[str],
    block_index: int,
    *,
    normalize_else_only_if: bool = False,
) -> CompletedBlock:
    ...
    # 1) 收集块中使用到的变量名（基于 AST / 语句）
    vars_used: Set[str] = collect_vars_in_block(block.stmts)

    # 2) 构建 name -> symbol
    sym_by_name: Dict[str, object] = {
        sym.name: sym
        for sym in pou_symtab.get_all_symbols()
    }

    fb_instance_decls: List[str] = []
    var_input_decls: List[str] = []
    var_output_decls: List[str] = []
    var_local_decls: List[str] = []

    known_func_like_prefixes = ("MC_", "F_", "REAL_TO_", "UDINT_TO_", "DINT_TO_")
    known_func_like_names = {
        "ESQR",
        "RealAbs",
        "UDINT_TO_REAL",
        "DINT_TO_REAL",
    }

    # 3) 先按 AST 使用情况做一轮粗过滤 + 分类
    local_var_names: Set[str] = set()

    for v in sorted(vars_used):
        sym = sym_by_name.get(v)

        if sym is None:
            if v.upper().startswith(known_func_like_prefixes) or v in known_func_like_names:
                continue
            continue

        storage = (getattr(sym, "storage", "") or "").upper()
        v_type = getattr(sym, "type", "REAL")
        role = (
            (getattr(sym, "role", "") or "")
            or (getattr(sym, "kind", "") or "")
        ).upper()

        if role in ("FB", "FUNCTION_BLOCK", "FB_INSTANCE"):
            fb_instance_decls.append(f"    {sym.name} : {v_type};")
            continue

        if role in ("FUNCTION", "FUNC", "METHOD", "ACTION", "PROGRAM"):
            continue

        if storage == "VAR_INPUT":
            var_input_decls.append(f"    {sym.name} : {v_type};")
        elif storage == "VAR_OUTPUT":
            var_output_decls.append(f"    {sym.name} : {v_type};")
        else:
            local_var_names.add(sym.name)

    # 3bis) 用“最终将输出的 body 文本”做一次真使用过滤（与输出一致）
    body_text = render_block_text(
        block,
        code_lines,
        normalize_else_only_if=normalize_else_only_if,
    )

    name_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    body_used_names: Set[str] = set(name_pattern.findall(body_text))

    for name in sorted(local_var_names):
        if name in body_used_names:
            sym = sym_by_name[name]
            v_type = getattr(sym, "type", "REAL")
            var_local_decls.append(f"    {name} : {v_type};")

    # 4) 组装 PROGRAM 框架
    prog_name = f"{pou_name}_BLOCK_{block_index}"
    out_lines: List[str] = []

    out_lines.append(f"PROGRAM {prog_name}")

    if fb_instance_decls:
        out_lines.append("VAR")
        out_lines.extend(fb_instance_decls)
        out_lines.append("END_VAR")

    if var_input_decls:
        out_lines.append("VAR_INPUT")
        out_lines.extend(var_input_decls)
        out_lines.append("END_VAR")

    if var_output_decls:
        out_lines.append("VAR_OUTPUT")
        out_lines.extend(var_output_decls)
        out_lines.append("END_VAR")

    if var_local_decls:
        out_lines.append("VAR")
        out_lines.extend(var_local_decls)
        out_lines.append("END_VAR")

    out_lines.append("")
    out_lines.append("(* ===== Functional body from original code ===== *)")
    out_lines.append(body_text.rstrip("\n"))
    out_lines.append("END_PROGRAM")

    code = "\n".join(out_lines)

    return CompletedBlock(
        block_index=block_index,
        code=code,
        line_numbers=list(block.line_numbers),
        vars_used=vars_used,
    )
