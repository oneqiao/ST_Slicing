# st_nl/nl/ir.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Set

from st_nl.ast import nodes as N
from st_nl.nl.emitter import CallSpec, extract_call_spec


CallKind = Literal["function", "fb"]
ArgDir = Literal["in", "out", "inout", "unknown"]


@dataclass(frozen=True)
class InputIR:
    name: Optional[str]          # 命名参数名；位置参数为 None
    expr: N.Expr                 # 参数表达式（保持 AST Expr）
    direction: ArgDir = "in"     # 新增：默认 in（不破坏旧代码）


@dataclass(frozen=True)
class OutputIR:
    target: N.Expr               # 输出目标（VarRef / ArrayAccess / FieldAccess / Tuple item etc.）


@dataclass(frozen=True)
class CallIR:
    callee: str
    call_kind: CallKind
    inputs: List[InputIR]
    outputs: List[OutputIR]
    loc: N.SourceLocation


def _expr_key(e: N.Expr) -> str:
    """
    用于去重的稳定 key（保守实现）。
    你后续如果希望更精确，可对 ArrayAccess/FieldAccess 做结构化 key。
    """
    if isinstance(e, N.VarRef):
        return f"VarRef:{e.name}"
    if isinstance(e, N.Literal):
        return f"Literal:{e.value}"
    # 兜底：repr 通常对 dataclass 节点稳定
    return f"{type(e).__name__}:{repr(e)}"


def callspec_to_callir(spec: CallSpec, call_kind: CallKind) -> CallIR:
    """
    CallSpec -> CallIR
    - pos_args: name=None, direction=in
    - named_args: name=<argname>, direction=na.direction (若无则 unknown)
    - outputs:
        1) spec.outputs（赋值 LHS）
        2) named_args 中 direction in (out, inout) 的目标变量（FB 调用关键）
       并做保序去重
    """
    inputs: List[InputIR] = []

    # 位置参数：name=None
    for a in spec.pos_args:
        inputs.append(InputIR(name=None, expr=a, direction="in"))

    # 命名参数：保留 name + direction
    for na in spec.named_args:
        d = getattr(na, "direction", None) or "unknown"
        if d not in ("in", "out", "inout"):
            d = "unknown"
        inputs.append(InputIR(name=na.name, expr=na.value, direction=d))

    # ---- outputs: LHS + out/inout 参数 ----
    out_exprs: List[N.Expr] = []
    seen: Set[str] = set()

    def push(e: N.Expr):
        k = _expr_key(e)
        if k not in seen:
            seen.add(k)
            out_exprs.append(e)

    # 1) LHS outputs（函数调用赋值等）
    for o in spec.outputs:
        push(o)

    # 2) out / inout named args（FB 调用的“写集合”）
    for na in spec.named_args:
        d = getattr(na, "direction", None) or "unknown"
        if d in ("out", "inout"):
            # 这里 value 应当是 variable 类（VarRef/FieldAccess/ArrayAccess），直接作为输出目标
            push(na.value)

    outputs: List[OutputIR] = [OutputIR(target=e) for e in out_exprs]

    return CallIR(
        callee=spec.callee,
        call_kind=call_kind,
        inputs=inputs,
        outputs=outputs,
        loc=spec.loc,
    )


def stmt_to_callir(stmt: N.Stmt) -> Optional[CallIR]:
    """
    一步到位：Stmt -> CallIR
    内部先 extract_call_spec，然后根据 stmt 类型决定 call_kind。
    """
    spec = extract_call_spec(stmt)
    if spec is None:
        return None

    if isinstance(stmt, N.CallStmt):
        kind: CallKind = "fb"
    elif isinstance(stmt, N.Assignment) and isinstance(stmt.value, N.CallExpr):
        kind = "function"
    else:
        kind = "function"

    return callspec_to_callir(spec, kind)
