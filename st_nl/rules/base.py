from __future__ import annotations

from typing import List, Optional

from st_nl.ast import nodes as N
from st_nl.nl.core import NLLine, NLFragment, EmitContext
from st_nl.nl.templates import tpl_assign,tpl_return, tpl_exit
from st_nl.nl.ir import stmt_to_callir

def emit_return_rule(stmt: N.Stmt, ctx: EmitContext) -> Optional[NLFragment]:
    if type(stmt).__name__ not in ("ReturnStmt", "Return"):
        return None
    return NLFragment([NLLine("Return", raw=False)])

def emit_continue_rule(stmt: N.Stmt, ctx: EmitContext) -> Optional[NLFragment]:
    if not isinstance(stmt, N.ContinueStmt):
        return None
    return NLFragment([NLLine("Continue loop", raw=False)])

def emit_exit_rule(stmt: N.Stmt, ctx: EmitContext) -> Optional[NLFragment]:
    if type(stmt).__name__ not in ("ExitStmt", "Exit", "BreakStmt", "Break"):
        return None
    return NLFragment([NLLine("Exit loop", raw=False)])

def emit_assign_rule(stmt: N.Stmt, ctx: EmitContext) -> Optional[NLFragment]:
    if not isinstance(stmt, N.Assignment):
        return None

    lhs = ctx.rexpr(stmt.target)
    rhs = ctx.rexpr(stmt.value)

    # 统一用模板（避免 := 噪声）
    return NLFragment([NLLine(tpl_assign(lhs, rhs), raw=False)])


def emit_call_rule(stmt: N.Stmt, ctx: EmitContext) -> Optional[NLFragment]:
    """
    处理所有能提取出 CallIR 的语句（函数/FB）。
    返回 None 表示不是 call。
    """
    cir = stmt_to_callir(stmt)
    if cir is None:
        return None

    cfg = ctx.cfg

    ins: List[str] = []
    for inp in cir.inputs:
        nm = inp.name if inp.name is not None else "<pos>"
        expr = ctx.rexpr(inp.expr)
        dir_ = getattr(inp, "direction", "in") or "in"
        ins.append(f"{nm}={expr}({dir_})")

    outs = [ctx.rexpr(o.target) for o in cir.outputs]
    in_s = ", ".join(ins)
    out_s = ", ".join(outs) if outs else "<no_out>"

    if cir.call_kind == "function":
        text = f"Call function: {out_s} <- {cir.callee}({in_s})"
    else:
        text = f"Call FB: {out_s} <- {cir.callee}({in_s})"

    return NLFragment([NLLine(text, raw=False)])

