# st_slicer/block_context.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict
import re

from .functional_blocks import FunctionalBlock, collect_vars_in_block


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
    local_var_names: Set[str] = set()   # 暂存普通 VAR 的名字，待会再按文本剪一刀

    for v in sorted(vars_used):
        sym = sym_by_name.get(v)

        if sym is None:
            if v.upper().startswith(known_func_like_prefixes) or v in known_func_like_names:
                continue
            # 其他未知名字，认为由工程环境提供：直接忽略
            continue

        storage = (getattr(sym, "storage", "") or "").upper()
        v_type = getattr(sym, "type", "REAL")
        role = (
            (getattr(sym, "role", "") or "")
            or (getattr(sym, "kind", "") or "")
        ).upper()

        # FB 实例
        if role in ("FB", "FUNCTION_BLOCK", "FB_INSTANCE"):
            fb_instance_decls.append(f"    {sym.name} : {v_type};")
            continue

        # 函数 / 程序 / 方法等，不在本块声明
        if role in ("FUNCTION", "FUNC", "METHOD", "ACTION", "PROGRAM"):
            continue

        # 普通变量，先暂存，后面按文本再裁一次
        if storage == "VAR_INPUT":
            var_input_decls.append(f"    {sym.name} : {v_type};")
        elif storage == "VAR_OUTPUT":
            var_output_decls.append(f"    {sym.name} : {v_type};")
        else:
            # 普通 VAR 先不直接生成声明，先记下名字
            local_var_names.add(sym.name)

    # 3bis) 基于当前 block 的源码文本，再对局部 VAR 做一次“真使用”过滤
    body_lines: List[str] = []
    for ln in sorted(block.line_numbers):
        if 1 <= ln <= len(code_lines):
            body_lines.append(code_lines[ln - 1])

    body_text = "\n".join(body_lines)

    # 简单标识符提取，用于判断名字是否真的出现在当前块源码中
    name_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    body_used_names: Set[str] = set(name_pattern.findall(body_text))

    for name in sorted(local_var_names):
        if name in body_used_names:
            sym = sym_by_name[name]
            v_type = getattr(sym, "type", "REAL")
            var_local_decls.append(f"    {name} : {v_type};")
        else:
            # 这里只是给调试看，确认有哪些被裁掉
            # print(f"[DEBUG] drop unused local var in block {block_index}: {name}")
            pass

    # 4) 组装 PROGRAM 框架（下面保持你原来的逻辑，只是用新的 var_local_decls）
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
    for ln in sorted(block.line_numbers):
        if 1 <= ln <= len(code_lines):
            out_lines.append(code_lines[ln - 1].rstrip())
    out_lines.append("END_PROGRAM")

    code = "\n".join(out_lines)

    return CompletedBlock(
        block_index=block_index,
        code=code,
        line_numbers=list(block.line_numbers),
        vars_used=vars_used,
    )