# st_nl/nl/emitter.py
import re
from dataclasses import dataclass
from typing import List, Optional
from st_nl.ast import nodes as N

_END_PUNCT = (".", "!", "?", ":")

@dataclass
class CallSpec:
    callee: str
    pos_args: List[N.Expr]
    named_args: List[N.NamedArg]
    outputs: List[N.Expr]
    loc: N.SourceLocation

def ensure_period(line: str) -> str:
    """
    只在行尾不是 . ! ? 时补一个句号。
    """
    s = (line or "").rstrip()
    if not s:
        return s
    if s.endswith(_END_PUNCT):
        return s
    return s + "."

def normalize_line(line: str) -> str:
    """
    行级规范化：
    1) 行尾多个标点归一：'..' -> '.', '!!' -> '!', '??' -> '?'
    2) 行尾空格清理
    3) （可选）逗号后空格统一为一个空格：'1,  2' -> '1, 2'
    """
    s = (line or "").strip()
    if not s:
        return s

    # 逗号后空格压缩（可选但建议做，保证 WHEN 1, 2 风格稳定）
    s = re.sub(r",\s*", ", ", s)

    # 行尾重复标点归一（只处理末尾连续同类标点）
    s = re.sub(r"\.{2,}$", ".", s)
    s = re.sub(r"!{2,}$", "!", s)
    s = re.sub(r"\?{2,}$", "?", s)

    return s

def finalize_line(line: str) -> str:
    """
    统一收口：先补句号（按需），再规范化，保证不会出现 '..'。
    """
    return normalize_line(ensure_period(line))

def extract_call_spec(stmt: N.Stmt) -> Optional[CallSpec]:
    if isinstance(stmt, N.Assignment) and isinstance(stmt.value, N.CallExpr):
        outs: List[N.Expr]

        # 调试：确认提取的语句
        #print(f"[DEBUG] Found Assignment with CallExpr: {stmt}")
        if isinstance(stmt.target, N.TupleExpr):
            outs = stmt.target.items
        else:
            outs = [stmt.target]

        return CallSpec(
            callee=stmt.value.func,
            pos_args=stmt.value.pos_args,
            named_args=stmt.value.named_args,
            outputs=outs,
            loc=stmt.loc,
        )

    if isinstance(stmt, N.CallStmt):
        # 调试：确认提取的函数块调用语句
        #print(f"[DEBUG] Found CallStmt: {stmt}")
        return CallSpec(
            callee=stmt.fb_name,
            pos_args=stmt.pos_args,
            named_args=stmt.named_args,
            outputs=[],
            loc=stmt.loc,
        )

    return None
