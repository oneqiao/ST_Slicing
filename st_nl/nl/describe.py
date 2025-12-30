# st_nl/nl/describe.py
from __future__ import annotations
from typing import List, Optional
from st_nl.ast import nodes as N
from st_nl.nl.ir import stmt_to_callir
from st_nl.nl.core import NLLine, NLFragment, EmitContext

def _strip_prefix(s: str) -> str:
    # 把 "Equal comparison: R = (A == B)" -> "R = (A == B)"
    if ":" in s:
        left, right = s.split(":", 1)
        # 只在左侧像标题时裁剪（可按你文档风格调整）
        if len(left) <= 32:
            return right.strip()
    return s.strip()

def _sem_sentence_from_call(stmt: N.Stmt, ctx: EmitContext) -> Optional[str]:
    cir = stmt_to_callir(stmt)
    if cir is None:
        return None

    catalog = getattr(ctx, "catalog", None)
    if catalog is None:
        return None

    ent = catalog.lookup(cir.callee)
    if ent is None or not ent.semantics_template:
        return None

    outs = [ctx.rexpr(o.target) for o in (cir.outputs or [])]
    pos_ins = [ctx.rexpr(i.expr) for i in (cir.inputs or []) if i.name is None]

    named_ins = {}
    named_outs = {}
    for i in (cir.inputs or []):
        if i.name is None:
            continue
        nm = i.name.upper()
        if nm.startswith("IN"):
            named_ins[nm] = ctx.rexpr(i.expr)
        elif nm.startswith("OUT"):
            if getattr(i, "direction", "") in ("out", "inout"):
                named_outs[nm] = ctx.rexpr(i.expr)

    bound = catalog.bind_template(
        ent.semantics_template,
        outs=outs,
        pos_ins=pos_ins,
        named_ins=named_ins,
        named_outs=named_outs,
    )
    return _strip_prefix(bound)

def _describe_block(stmts: List[N.Stmt], ctx: EmitContext, depth: int, *, max_sent: int = 6) -> str:
    # 收集若干关键句，避免变回 FINE
    sents: List[str] = []
    for s in stmts:
        if len(sents) >= max_sent:
            sents.append("...")
            break
        one = describe_stmt_sentence(s, ctx, depth + 1)
        if one:
            sents.append(one)
    return " ".join(sents).strip()

def describe_stmt_sentence(stmt: N.Stmt, ctx: EmitContext, depth: int) -> Optional[str]:
    # 控制深度，防止过长
    if depth >= getattr(ctx.cfg, "fine_max_depth", 7):
        return "..."

    # 1) 优先：语义调用（UEQB/UGEB 等）
    sem = _sem_sentence_from_call(stmt, ctx)
    if sem:
        # 输出如 "R1 = (A == B)"，更像自然语言伪代码
        return f"Set {sem}."

    # 2) 赋值
    if isinstance(stmt, N.Assignment):
        lhs = ctx.rexpr(stmt.target)
        rhs = ctx.rexpr(stmt.value)
        return f"Set {lhs} to {rhs}."

    # 3) IF
    if isinstance(stmt, N.IfStmt):
        cond = ctx.rexpr(stmt.cond)
        then_txt = _describe_block(stmt.then_body, ctx, depth, max_sent=4)
        parts = [f"If {cond}, then {then_txt}"]
        for (c, body) in (stmt.elif_branches or []):
            cc = ctx.rexpr(c)
            bt = _describe_block(body, ctx, depth, max_sent=3)
            parts.append(f"else if {cc}, then {bt}")
        if stmt.else_body:
            et = _describe_block(stmt.else_body, ctx, depth, max_sent=3)
            parts.append(f"otherwise {et}")
        return "; ".join(parts) + "."

    # 4) CASE
    if isinstance(stmt, N.CaseStmt):
        sel = ctx.rexpr(stmt.cond)
        branches: List[str] = []
        for e in stmt.entries:
            conds = ", ".join(c.text for c in e.conds)
            bt = _describe_block(e.body, ctx, depth, max_sent=3)
            branches.append(f"when {sel} is {conds}: {bt}")
        if stmt.else_body:
            et = _describe_block(stmt.else_body, ctx, depth, max_sent=2)
            branches.append(f"otherwise: {et}")
        return f"Branch on {sel}: " + "; ".join(branches) + "."

    # 5) FOR/WHILE/REPEAT
    if isinstance(stmt, N.ForStmt):
        start = ctx.rexpr(stmt.start)
        end = ctx.rexpr(stmt.end)
        step = ctx.rexpr(stmt.step) if stmt.step else "1"
        bt = _describe_block(stmt.body, ctx, depth, max_sent=5)
        return f"Loop {stmt.var} from {start} to {end} step {step}: {bt}"

    if isinstance(stmt, N.WhileStmt):
        cond = ctx.rexpr(stmt.cond)
        bt = _describe_block(stmt.body, ctx, depth, max_sent=5)
        return f"While {cond}: {bt}"

    if isinstance(stmt, N.RepeatStmt):
        until = ctx.rexpr(stmt.until)
        bt = _describe_block(stmt.body, ctx, depth, max_sent=5)
        return f"Repeat: {bt} until {until}"

    return None

def describe_pou(pou: N.ProgramDecl | N.FBDecl, ctx: EmitContext) -> NLFragment:
    # 把整个 POU 汇总为 1-3 行
    body_txt = _describe_block(pou.body, ctx, depth=0, max_sent=10)
    line1 = f"Program {pou.name}: {body_txt}"
    return NLFragment([NLLine(line1, raw=False)])
