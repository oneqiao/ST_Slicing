# st_nl/nl/generate.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum, auto

from st_nl.ast import nodes as N
from st_nl.nl.ir import stmt_to_callir
from st_nl.nl.render import render_expr, RenderCfg

from st_nl.nl.core import NLLine, NLFragment, EmitContext, finalize_fragment

from st_nl.rules.control_flow import (
    emit_if_rule,
    emit_case_rule,
    emit_for_rule,
    emit_while_rule,
    emit_repeat_rule,
)
from st_nl.rules.base import (
    emit_assign_rule,
    emit_continue_rule,
    emit_call_rule,
)

from st_nl.nl.templates import tpl_assign

# -----------------------
# NL Level / Config
# -----------------------
class NLLevel(Enum):
    COARSE = auto()
    MEDIUM = auto()
    FINE = auto()


@dataclass(frozen=True)
class NLCfg:
    nl_level: NLLevel = NLLevel.MEDIUM
    enable_enriched: bool = True

    summary_max_depth: int = 2
    summary_max_items: int = 3
    summary_joiner: str = "; "

    fine_max_depth: int = 7
    fine_max_stmts: int = 50
    fine_indent: str = "  "

    render: RenderCfg = RenderCfg(expr_max_len=80)


@dataclass(frozen=True)
class DocEntry:
    summary: str


_PLACEHOLDER_RE = re.compile(r"\b(IN|OUT)(\d+)\b")


def _strip_end_punct(s: str) -> str:
    return (s or "").strip().rstrip(".!?").strip()


# -----------------------
# Action summarizer
# -----------------------
def summarize_block(stmts: List[N.Stmt], cfg: NLCfg, depth: int) -> str:
    if depth > cfg.summary_max_depth:
        return "..."

    items: List[str] = []
    for s in stmts:
        if len(items) >= cfg.summary_max_items:
            items.append("...")
            break

        cir = stmt_to_callir(s)
        if cir is not None:
            items.append(cir.callee)
            continue

        if isinstance(s, N.Assignment):
            items.append(f"assign {render_expr(s.target, cfg.render)}")
            continue

        if isinstance(s, N.IfStmt):
            items.append("if(...)")
            items.append(summarize_block(s.then_body, cfg, depth + 1))
            continue

        if isinstance(s, N.CaseStmt):
            items.append("case(...)")
            continue

        if isinstance(s, (N.ForStmt, N.WhileStmt, N.RepeatStmt)):
            items.append(type(s).__name__.replace("Stmt", "").lower())
            continue

        items.append(type(s).__name__)

    return cfg.summary_joiner.join([x for x in items if x])


# -----------------------
# Generic emitters
# -----------------------
def emit_generic_call(cir, cfg: NLCfg) -> str:
    ins: List[str] = []
    for inp in cir.inputs:
        nm = inp.name if inp.name is not None else "<pos>"
        expr = render_expr(inp.expr, cfg.render)
        dir_ = getattr(inp, "direction", "in") or "in"
        ins.append(f"{nm}={expr}({dir_})")

    outs = [render_expr(o.target, cfg.render) for o in cir.outputs]
    in_s = ", ".join(ins)
    out_s = ", ".join(outs) if outs else "<no_out>"

    if cir.call_kind == "function":
        return f"Call function: {out_s} <- {cir.callee}({in_s})"
    else:
        return f"Call FB: {out_s} <- {cir.callee}({in_s})"


def emit_generic_stmt(stmt: N.Stmt, ctx: EmitContext) -> NLFragment:
    cfg: NLCfg = ctx.cfg

    cir = stmt_to_callir(stmt)
    if cir is not None:
        return NLFragment([NLLine(emit_generic_call(cir, cfg), raw=False)])

    if isinstance(stmt, N.Assignment):
        lhs = render_expr(stmt.target, cfg.render)
        rhs = render_expr(stmt.value, cfg.render)
        # 注意：tpl_assign 输出的是 "="，你后续想保留 ":=" 可再改模板
        return NLFragment([NLLine(tpl_assign(lhs, rhs), raw=False)])

    if isinstance(stmt, N.ContinueStmt):
        return NLFragment([NLLine("Continue loop", raw=False)])

    return NLFragment([NLLine(f"Stmt {type(stmt).__name__}", raw=False)])


# -----------------------
# Semantics
# -----------------------
def _pos_inputs(cir) -> List[N.Expr]:
    return [i.expr for i in cir.inputs if i.name is None]


def enrich_ushlw(cir, cfg: NLCfg) -> str:
    outs = cir.outputs or []
    pos = _pos_inputs(cir)

    if len(outs) < 1 or len(pos) < 2:
        return "Semantics: Logical left shift"

    lhs = render_expr(outs[0].target, cfg.render)
    a0  = render_expr(pos[0], cfg.render)
    a1  = render_expr(pos[1], cfg.render)
    return f"Semantics: Logical left shift: {lhs} = {a0} << {a1}"


def maybe_enrich(stmt: N.Stmt, docs: Dict[str, DocEntry], cfg: NLCfg) -> Optional[str]:
    if not cfg.enable_enriched:
        return None

    cir = stmt_to_callir(stmt)
    if cir is None:
        return None

    if cir.callee == "USHLW" and cir.call_kind == "function":
        return enrich_ushlw(cir, cfg)

    doc = docs.get(cir.callee)
    if doc is None:
        return None
    return f"Semantics: {doc.summary}"


# -----------------------
# Dispatcher (always returns NLFragment)
# -----------------------
def emit_stmt(stmt: N.Stmt, ctx: EmitContext, depth: int = 0) -> NLFragment:
    cfg: NLCfg = ctx.cfg
    docs: Dict[str, DocEntry] = ctx.docs

    if isinstance(stmt, N.IfStmt):
        return emit_if_rule(stmt, ctx, depth, emit_stmt, summarize_block)

    if isinstance(stmt, N.CaseStmt):
        return emit_case_rule(stmt, ctx, depth, emit_stmt, summarize_block)

    if isinstance(stmt, N.ForStmt):
        return emit_for_rule(stmt, ctx, depth, emit_stmt, summarize_block)

    if isinstance(stmt, N.WhileStmt):
        return emit_while_rule(stmt, ctx, depth, emit_stmt, summarize_block)

    if isinstance(stmt, N.RepeatStmt):
        return emit_repeat_rule(stmt, ctx, depth, emit_stmt, summarize_block)

    # ---- 普通语句：Call / Assign / Continue / fallback + (可选) Enriched ----
    frag = (
        emit_call_rule(stmt, ctx)
        or emit_assign_rule(stmt, ctx)
        or emit_continue_rule(stmt, ctx)
    )

    if frag is None:
        frag = NLFragment([NLLine(f"Stmt {type(stmt).__name__}", raw=False)])

    extra = maybe_enrich(stmt, docs, cfg)
    if extra:
        frag = NLFragment(frag.lines + [NLLine(extra, raw=False)])

    return frag


def emit_pou(pou: N.ProgramDecl | N.FBDecl, cfg: NLCfg, docs: Dict[str, DocEntry]) -> List[str]:
    ctx = EmitContext(cfg=cfg, docs=docs)

    lines: List[NLLine] = [NLLine(f"POU {pou.name}", raw=False)]
    for s in pou.body:
        lines.extend(emit_stmt(s, ctx, depth=0).lines)
    lines.append(NLLine(f"END_POU {pou.name}", raw=False))

    return finalize_fragment(NLFragment(lines=lines))
