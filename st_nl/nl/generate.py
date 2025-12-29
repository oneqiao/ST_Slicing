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
    emit_exit_rule,
    emit_return_rule,
)

from st_nl.nl.templates import tpl_assign
from st_nl.rules.semantic_catalog import SemanticCatalog, norm_name

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

# -----------------------
# Action summarizer
# -----------------------
def summarize_block(stmts: List[N.Stmt], ctx: EmitContext, depth: int) -> str:
    cfg: NLCfg = ctx.cfg

    if depth > cfg.summary_max_depth:
        return "..."

    items: List[str] = []
    for s in stmts:
        if len(items) >= cfg.summary_max_items:
            items.append("...")
            break

        cir = stmt_to_callir(s)
        if cir is not None:
            # 1) 先拿 callee 名
            name = cir.callee

            # 2) 尝试把语义模板绑定进去（只在摘要里体现，不额外多行）
            sem = None
            catalog = getattr(ctx, "catalog", None)
            if catalog is not None:
                ent = catalog.lookup(name)
                if ent is not None and ent.semantics_template:
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
                                # 这里 OUT 参数 expr 通常是变量引用
                                named_outs[nm] = ctx.rexpr(i.expr)

                    bound = catalog.bind_template(
                        ent.semantics_template,
                        outs=outs,
                        pos_ins=pos_ins,
                        named_ins=named_ins,
                        named_outs=named_outs,
                    )
                    sem = bound.strip()

            # 3) 摘要串：带语义就更“可学习”
            if sem:
                items.append(f"{name}[{sem}]")
            else:
                items.append(name)
            continue

        if isinstance(s, N.Assignment):
            lhs = ctx.rexpr(s.target)
            rhs = ctx.rexpr(s.value)
            items.append(f"{lhs}={rhs}")
            continue

        if isinstance(s, N.IfStmt):
            items.append(f"if({ctx.rexpr(s.cond)})")
            continue

        if isinstance(s, N.CaseStmt):
            items.append(f"case({ctx.rexpr(s.cond)})")
            continue

        if isinstance(s, N.ForStmt):
            items.append("for(...)")
            continue
        if isinstance(s, N.WhileStmt):
            items.append("while(...)")
            continue
        if isinstance(s, N.RepeatStmt):
            items.append("repeat(...)")
            continue

        items.append(type(s).__name__)

    return cfg.summary_joiner.join([x for x in items if x])

def maybe_enrich(stmt: N.Stmt, ctx: EmitContext) -> Optional[NLFragment]:
    if not getattr(ctx.cfg, "enable_enriched", False):
        return None

    cir = stmt_to_callir(stmt)
    if cir is None:
        return None

    catalog = ctx.catalog
    ent = catalog.lookup(cir.callee) if catalog else None
    if ent is None or not ent.semantics_template:
        return None

    # ---- 绑定材料 ----
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

    return NLFragment([NLLine(f"Semantics: {bound}", raw=False)])


# Dispatcher (always returns NLFragment)

def emit_stmt(stmt: N.Stmt, ctx: EmitContext, depth: int = 0) -> NLFragment:
    # 1) 控制流规则（你已经迁移好了）
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

    # 2) 普通语句规则：call/assign/return/exit/continue...
    frag: Optional[NLFragment] = None

    # 你如果把 emit_call_rule/emit_assign_rule 放在 control_flow.py 里也没问题
    frag = frag or emit_call_rule(stmt, ctx)
    frag = frag or emit_assign_rule(stmt, ctx)
    frag = frag or emit_return_rule(stmt, ctx)   # 若你已实现
    frag = frag or emit_exit_rule(stmt, ctx)     # 若你已实现
    frag = frag or emit_continue_rule(stmt, ctx)

    if frag is None:
        frag = NLFragment([NLLine(f"Stmt {type(stmt).__name__}", raw=False)])

    # 3) Enriched：统一在 dispatcher 追加（保证 Call 后紧跟 Semantics）
    extra = maybe_enrich(stmt, ctx)
    if extra:
        frag = NLFragment(frag.lines + extra.lines)

    return frag

def emit_pou(
    pou: N.ProgramDecl | N.FBDecl,
    cfg: NLCfg,
    catalog,   # SemanticCatalog
) -> List[str]:
    ctx = EmitContext(cfg=cfg, catalog=catalog)

    lines: List[NLLine] = [NLLine(f"POU {pou.name}", raw=False)]
    for s in pou.body:
        lines.extend(emit_stmt(s, ctx, depth=0).lines)
    lines.append(NLLine(f"END_POU {pou.name}", raw=False))

    return finalize_fragment(NLFragment(lines=lines))
