# st_slicer/block_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import re

from st_slicer.blocks.core import FunctionalBlock
from st_slicer.blocks.pipeline import collect_vars_in_block
from st_slicer.blocks.render import render_block_text


# --------- 1) 可执行语句判定：只保留“动作”，控制头不算 ----------
_EXEC_PAT = re.compile(
    r"(:=|\bRETURN\b|\bEXIT\b|\bCONTINUE\b|\w+\s*\()",  # assignment / call / return-like
    re.IGNORECASE,
)

_STRUCT_ONLY = {
    "ELSE", "END_IF", "END_PROGRAM", "END_CASE", "END_FOR", "END_WHILE",
    "VAR", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "END_VAR",
    "IF", "ELSIF", "THEN",
}

_IF_RE = re.compile(r"^(\s*)IF\b(.*)\bTHEN\b\s*$", re.IGNORECASE)
_ELSIF_RE = re.compile(r"^(\s*)ELSIF\b(.*)\bTHEN\b\s*$", re.IGNORECASE)
_ELSE_RE = re.compile(r"^(\s*)ELSE\b\s*$", re.IGNORECASE)
_ENDIF_RE = re.compile(r"^(\s*)END_IF\b\s*;?\s*$", re.IGNORECASE)


def is_executable_st_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # full-line comments
    if s.startswith("//"):
        return False
    if s.startswith("(*") and s.endswith("*)"):
        return False

    up = s.upper().rstrip(";")
    if up in _STRUCT_ONLY:
        return False

    # likely var decl: "a : REAL;" (but not "a := ..."
    if ":" in s and ":=" not in s and s.endswith(";"):
        return False

    # 纯控制头不算“执行语句”
    if _IF_RE.match(line) or _ELSIF_RE.match(line) or _ELSE_RE.match(line) or _ENDIF_RE.match(line):
        return False

    return _EXEC_PAT.search(s) is not None


# --------- 2) IF/ELSIF/ELSE 轻量解析与裁剪 + 补全 ----------
@dataclass
class Branch:
    header: str                    # "IF ... THEN" or "ELSIF ... THEN" or "ELSE"
    lines: List[str] = field(default_factory=list)

    def has_exec(self) -> bool:
        return any(is_executable_st_line(ln) for ln in self.lines)


@dataclass
class IfNode:
    indent: str
    branches: List[Branch] = field(default_factory=list)
    raw_end_line: Optional[str] = None  # preserve original END_IF line

    def any_exec(self) -> bool:
        return any(b.has_exec() for b in self.branches)


@dataclass
class CompletedBlock:
    """
    表示一个已经“结构补全”的功能块：
      - code: 完整的 ST 程序文本（VAR区 + Functional body）
      - line_numbers: 源程序中涉及到的行号
      - vars_used: 该块中用到的变量名集合
    """
    block_index: int
    code: str
    line_numbers: List[int]
    vars_used: Set[str]


def _promote_elsif_to_if(header: str) -> str:
    """
    将 'ELSIF cond THEN' 提升为 'IF cond THEN'，保留缩进与 cond。
    """
    m = _ELSIF_RE.match(header)
    if not m:
        return header
    indent = m.group(1) or ""
    cond = (m.group(2) or "").strip()
    return f"{indent}IF {cond} THEN"


def _wrap_else_only(indent: str, else_body_lines: List[str]) -> List[str]:
    """
    只剩 ELSE 分支时，为保证语法可用：
    转为：
        IF TRUE THEN
            ...
        END_IF;
    """
    out: List[str] = []
    out.append(f"{indent}IF TRUE THEN")
    out.extend(else_body_lines)
    out.append(f"{indent}END_IF;")
    return out


def _count_if_end(lines: List[str]) -> Tuple[int, int]:
    n_if = sum(1 for ln in lines if _IF_RE.match(ln))
    n_end = sum(1 for ln in lines if _ENDIF_RE.match(ln))
    return n_if, n_end

def _if_depth_scan(lines: List[str]) -> Tuple[bool, int, int]:
    """
    返回 (ok, min_depth, final_depth)
    ok = True 表示扫描过程中 depth 从未 < 0
    """
    depth = 0
    min_depth = 0
    for ln in lines:
        if _IF_RE.match(ln):
            depth += 1
        elif _ENDIF_RE.match(ln):
            depth -= 1
            if depth < min_depth:
                min_depth = depth
    return (min_depth >= 0, min_depth, depth)


def remove_orphan_control_lines(lines: List[str]) -> List[str]:
    """
    先清理切片导致的“孤儿控制行”：
      - depth==0 时遇到 END_IF/ELSE/ELSIF：丢弃
      - depth>0 时保留 END_IF/ELSE/ELSIF（由后续 simplify 处理）
    说明：这一步只做“删除孤儿”，不做结构补全、不引入 IF TRUE。
    """
    out: List[str] = []
    depth = 0

    for ln in lines:
        # 统一用原始行（保留缩进与分号风格）
        if _IF_RE.match(ln):
            depth += 1
            out.append(ln)
            continue

        if _ENDIF_RE.match(ln):
            if depth == 0:
                # 孤儿 END_IF，直接丢弃
                continue
            depth -= 1
            out.append(ln)
            continue

        if _ELSIF_RE.match(ln) or _ELSE_RE.match(ln):
            if depth == 0:
                # 孤儿 ELSIF/ELSE，直接丢弃
                continue
            out.append(ln)
            continue

        out.append(ln)

    # 注意：这里不主动补 END_IF。补全会改变语义；我们只负责“清理孤儿”
    return out

def flatten_synthetic_if_true(lines: List[str]) -> List[str]:
    """
    展开切片器注入的 'IF TRUE THEN' 包裹层：
      - 遇到 IF TRUE THEN：入栈标记为 synthetic，不输出该行
      - 遇到 END_IF：若匹配的是 synthetic，则不输出该 END_IF
      - 遇到 ELSE/ELSIF：若当前 depth==0（孤儿），丢弃
      - 遇到 END_IF：若当前 depth==0（孤儿），丢弃
    目的：消除大量 IF TRUE 包裹导致的缺闭合/多闭合连锁错误
    """
    out: List[str] = []
    stack: List[bool] = []  # True=synthetic IF TRUE, False=normal IF

    for ln in lines:
        if _IF_RE.match(ln):
            # 判断是否为合成 IF TRUE THEN（允许缩进）
            if ln.strip().upper().startswith("IF TRUE THEN"):
                stack.append(True)
                continue  # 不输出 synthetic opener
            else:
                stack.append(False)
                out.append(ln)
                continue

        if _ELSIF_RE.match(ln) or _ELSE_RE.match(ln):
            if not stack:
                continue  # 孤儿 ELSE/ELSIF
            out.append(ln)
            continue

        if _ENDIF_RE.match(ln):
            if not stack:
                continue  # 孤儿 END_IF
            is_syn = stack.pop()
            if is_syn:
                continue  # 不输出 synthetic closer
            out.append(ln)
            continue

        out.append(ln)

    # 注意：这里不自动补 END_IF（避免改变真实语义）
    # 但 synthetic 的 IF TRUE 如果没闭合，会在 stack 中残留，
    # 它本来就不应存在，直接忽略即可（相当于“把包裹层彻底移除”）。
    return out


def simplify_st_if_skeleton(lines: List[str]) -> List[str]:
    """
    目标：
      - 删除空分支（THEN/ELSIF/ELSE 内无可执行语句）
      - 删除整块空 IF（所有分支都空）
      - 补全：避免裁剪后出现孤儿 ELSE/ELSIF
          * 若首分支是 ELSIF -> 提升为 IF
          * 若只剩 ELSE -> 包一层 IF TRUE THEN
    注意：
      - 该函数假设输入 IF 结构“完整”，不负责处理明显不配对的片段；
        片段场景应由 preprocess_slice_block_text 的 guard 跳过本函数。
    """
    out: List[str] = []
    stack: List[IfNode] = []

    def emit_if(node: IfNode) -> List[str]:
        # 1) 裁剪空分支
        kept = [b for b in node.branches if b.has_exec()]
        if not kept:
            return []

        # 2) 补全：修复开头不是 IF 的情况
        # 2.1 只剩 ELSE
        if len(kept) == 1 and _ELSE_RE.match(kept[0].header):
            return _wrap_else_only(node.indent, kept[0].lines)

        # 2.2 首分支是 ELSIF：提升为 IF
        if kept and _ELSIF_RE.match(kept[0].header):
            kept[0].header = _promote_elsif_to_if(kept[0].header)

        # 2.3 极端：首分支是 ELSE（理论上不应发生，但为安全补一层）
        if kept and _ELSE_RE.match(kept[0].header):
            return _wrap_else_only(node.indent, kept[0].lines)

        # 3) 输出
        emitted: List[str] = []
        for b in kept:
            emitted.append(b.header)
            emitted.extend(b.lines)

        end_ln = node.raw_end_line if node.raw_end_line is not None else (node.indent + "END_IF;")
        # END_IF 规范化：确保有分号（不强制改变原风格）
        emitted.append(end_ln)
        return emitted

    for ln in lines:
        s = ln.rstrip("\n")
        m_if = _IF_RE.match(s)
        m_els = _ELSIF_RE.match(s)
        m_else = _ELSE_RE.match(s)
        m_end = _ENDIF_RE.match(s)

        if m_if:
            indent = m_if.group(1) or ""
            node = IfNode(indent=indent)
            node.branches.append(Branch(header=s))
            stack.append(node)
            continue

        if stack:
            node = stack[-1]

            if m_els:
                node.branches.append(Branch(header=s))
                continue
            if m_else:
                node.branches.append(Branch(header=s))
                continue
            if m_end:
                node.raw_end_line = s
                stack.pop()
                simplified = emit_if(node)

                if stack:
                    # nested: append to parent's current branch body
                    stack[-1].branches[-1].lines.extend(simplified)
                else:
                    out.extend(simplified)
                continue

            # normal line inside IF: append to current branch
            node.branches[-1].lines.append(s)
            continue

        # outside IF
        out.append(s)

    # 若仍有未闭合 IF：保守原样输出（不做裁剪/提升），避免结构进一步损坏
    while stack:
        node = stack.pop(0)
        for b in node.branches:
            out.append(b.header)
            out.extend(b.lines)
        out.append(node.raw_end_line or (node.indent + "END_IF;"))

    return out


# --------- 3) 可选：去掉 VAR 声明区，只保留 Functional body ----------
_VAR_START_RE = re.compile(r"^\s*VAR(_INPUT|_OUTPUT|_IN_OUT)?\b", re.IGNORECASE)
_ENDVAR_RE = re.compile(r"^\s*END_VAR\b", re.IGNORECASE)


def strip_var_sections(lines: List[str]) -> List[str]:
    res: List[str] = []
    in_var = False
    for ln in lines:
        s = ln.rstrip("\n")
        if _VAR_START_RE.match(s):
            in_var = True
            continue
        if in_var and _ENDVAR_RE.match(s):
            in_var = False
            continue
        if not in_var:
            res.append(s)
    return res


# --------- 4) 总入口：对一个 block 文本做清洗 ----------
def preprocess_slice_block_text(block_text: str, drop_var: bool = True) -> str:
    lines = block_text.splitlines()

    if drop_var:
        lines = strip_var_sections(lines)

    # 0) 先展开 IF TRUE THEN 包裹层（这是你当前结构崩坏的根因）
    lines = flatten_synthetic_if_true(lines)

    # 1) 再清理孤儿控制行（保险）
    lines = remove_orphan_control_lines(lines)

    # 2) 强 guard：结构闭合才做 IF skeleton 剪裁
    ok, _min_depth, final_depth = _if_depth_scan(lines)
    if ok and final_depth == 0:
        lines = simplify_st_if_skeleton(lines)

    # 3) 合并多余空行
    cleaned: List[str] = []
    last_blank = False
    for ln in lines:
        ln = ln.rstrip("\n")
        blank = (ln.strip() == "")
        if blank and last_blank:
            continue
        cleaned.append(ln)
        last_blank = blank

    return "\n".join(cleaned).strip() + "\n"




def classify_variable(symbol):
    """
    根据 symbol 判断变量应该放在什么声明区。
    返回:
        ("fb_instance", type_name)   → 需要放 VAR 中的 FB 实例
        ("normal_var", type_name)    → 普通变量
        ("ignore", None)             → function，不需要声明
    """
    role = getattr(symbol, "role", None)
    typ = getattr(symbol, "type", None)

    if role in ("FB", "FUNCTION_BLOCK"):
        return ("fb_instance", typ)

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
    simplify_empty_if: bool = True,
) -> CompletedBlock:
    # 1) 收集块中使用到的变量名（基于 AST / 语句）
    vars_used: Set[str] = collect_vars_in_block(block.stmts)

    # 2) 构建 name -> symbol
    sym_by_name: Dict[str, object] = {sym.name: sym for sym in pou_symtab.get_all_symbols()}

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
            # 未知符号：如果像函数/转换，就不声明
            if v.upper().startswith(known_func_like_prefixes) or v in known_func_like_names:
                continue
            continue

        storage = (getattr(sym, "storage", "") or "").upper()
        v_type = getattr(sym, "type", "REAL")
        role = ((getattr(sym, "role", "") or "") or (getattr(sym, "kind", "") or "")).upper()

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

    if simplify_empty_if:
        body_text = preprocess_slice_block_text(body_text, drop_var=False)

    name_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    body_used_names: Set[str] = set(name_pattern.findall(body_text))

    for name in sorted(local_var_names):
        if name in body_used_names:
            sym = sym_by_name[name]
            v_type = getattr(sym, "type", "REAL")
            var_local_decls.append(f"    {name} : {v_type};")

    # 4) 组装输出
    out_lines: List[str] = []

    if var_input_decls:
        out_lines.append("VAR_INPUT")
        out_lines.extend(var_input_decls)
        out_lines.append("END_VAR")

    if var_output_decls:
        out_lines.append("VAR_OUTPUT")
        out_lines.extend(var_output_decls)
        out_lines.append("END_VAR")

    # FB 实例与普通局部变量都放 VAR（你也可以按需拆开）
    if fb_instance_decls or var_local_decls:
        out_lines.append("VAR")
        out_lines.extend(fb_instance_decls)
        out_lines.extend(var_local_decls)
        out_lines.append("END_VAR")

    out_lines.append("")
    out_lines.append("(* ===== Functional body from original code ===== *)")
    out_lines.append(body_text.rstrip("\n"))

    code = "\n".join(out_lines)

    return CompletedBlock(
        block_index=block_index,
        code=code,
        line_numbers=list(block.line_numbers),
        vars_used=vars_used,
    )
