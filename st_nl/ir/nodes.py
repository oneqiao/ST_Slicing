# st_nl/ir/nodes.py
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any

from st_nl.ast import nodes as N
from st_nl.nl.emitter import CallSpec

CallKind = Literal["function", "fb"]

@dataclass
class ArgIR:
    name: Optional[str]   # 命名参数则是 "IN1"，位置参数则 None
    expr: Any             # 先直接挂 AST Expr，后面 expr renderer 再处理

@dataclass
class OutputIR:
    target: Any           # AST Expr（VarRef / FieldAccess / ArrayAccess / Tuple item）
    name: Optional[str] = None  # 可选：target是VarRef时可填 name，便于模板

@dataclass
class CallIR:
    callee: str
    call_kind: CallKind
    inputs: List[ArgIR] = field(default_factory=list)
    outputs: List[OutputIR] = field(default_factory=list)
    loc: Any = None

@dataclass
class AssignIR:
    target: Any
    value: Any
    loc: Any = None

@dataclass
class IfIR:
    cond: Any
    then_body: List[Any]
    elif_branches: List[tuple[Any, List[Any]]] = field(default_factory=list)
    else_body: List[Any] = field(default_factory=list)
    loc: Any = None

@dataclass
class CallArgIR:
    name: Optional[str]   # None for positional
    expr: N.Expr
    direction: Optional[str] = None  # "in"/"out"/None

@dataclass
class CallOutIR:
    name: Optional[str]
    target: N.Expr

@dataclass
class CallIR:
    callee: str
    call_kind: str         # "function" | "fb"
    inputs: List[CallArgIR]
    outputs: List[CallOutIR]
    loc: N.SourceLocation

def call_spec_to_ir(spec: CallSpec) -> CallIR:
    kind = "fb" if spec.outputs == [] and spec.callee and True else "function"
    # 更稳的做法：你在 extract_call_spec 里对 CallStmt 直接标 fb；这里先保持你现有结构
    # 如果你愿意，建议把 CallSpec 也加 call_kind 字段。

    inputs: List[CallArgIR] = []
    outputs: List[CallOutIR] = []

    # 位置参数
    for e in spec.pos_args:
        inputs.append(CallArgIR(name=None, expr=e, direction=None))

    # 命名参数：区分 in/out
    for na in spec.named_args:
        if getattr(na, "direction", None) == "out":
            outputs.append(CallOutIR(name=na.name, target=na.value))
        else:
            inputs.append(CallArgIR(name=na.name, expr=na.value, direction=getattr(na, "direction", None)))

    # Assignment 的 LHS 输出（函数式写法）：优先用 spec.outputs
    if spec.outputs:
        kind = "function"
        outputs = [CallOutIR(name=getattr(o, "name", None), target=o) for o in spec.outputs]  # o 通常是 VarRef

    return CallIR(
        callee=spec.callee,
        call_kind=kind,
        inputs=inputs,
        outputs=outputs,
        loc=spec.loc
    )