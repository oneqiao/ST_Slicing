# st_nl/nl/rules/control_flow.py
from __future__ import annotations

from typing import Callable, List, Optional, Dict, Any

from st_nl.ast import nodes as N
from st_nl.nl.core import NLLine, NLFragment, EmitContext, indent_fragment
from st_nl.nl.ir import stmt_to_callir
from st_nl.nl.templates import (
    tpl_if,
    tpl_elsif,
    tpl_end_if,
    tpl_then_actions,
    tpl_elsif_actions,
    tpl_else_actions,

    tpl_case,
    tpl_end_case,
    tpl_when_actions,

    tpl_for,
    tpl_end_for,
    tpl_loop_actions,

    tpl_while,
    tpl_end_while,

    tpl_repeat,
    tpl_until,
    tpl_end_repeat,

    tpl_assign,
)

EmitStmtFn = Callable[[N.Stmt, EmitContext, int], NLFragment]
SummFn = Callable[[List[N.Stmt], object, int], str]


def _fine_expand_block(
    stmts: List[N.Stmt],
    ctx: EmitContext,
    depth: int,
    emit_stmt: EmitStmtFn,
) -> NLFragment:
    """
    FINE 级别：展开 block 子语句（受 depth/max_stmts 控制），并统一缩进。
    depth 表示“当前 block 所在的控制结构嵌套层级”。
    """
    cfg = ctx.cfg
    max_depth = getattr(cfg, "fine_max_depth", 6)   # 建议默认 >= 6
    max_n = getattr(cfg, "fine_max_stmts", 20)
    indent = getattr(cfg, "fine_indent", "  ")

    # 关键：这里用 depth+1 判上限更直观（因为 emit_stmt 递归会传 depth+1）
    if depth + 1 >= max_depth:
        return indent_fragment(NLFragment([NLLine("...", raw=False)]), indent)

    # 关键：空块显式输出 <empty>，避免出现 “actions:” 后没有内容
    if not stmts:
        return indent_fragment(NLFragment([NLLine("<empty>", raw=False)]), indent)

    out: List[NLLine] = []
    count = 0
    for s in stmts:
        if count >= max_n:
            out.append(NLLine("...", raw=False))
            break
        sub = emit_stmt(s, ctx, depth + 1)
        out.extend(sub.lines)
        count += 1

    return indent_fragment(NLFragment(out), indent)

# -----------------------
# IF rule
# -----------------------
def emit_if_rule(
    stmt: N.IfStmt,
    ctx: EmitContext,
    depth: int,
    emit_stmt: EmitStmtFn,
    summarize_block: SummFn,
) -> NLFragment:
    lines: List[NLLine] = []
    cond = ctx.rexpr(stmt.cond)

    # COARSE：只结构（注意：actions 不输出）
    if ctx.is_coarse():
        lines.append(NLLine(tpl_if(cond), raw=False))
        for (c, _body) in (stmt.elif_branches or []):
            lines.append(NLLine(tpl_elsif(ctx.rexpr(c)), raw=False))
        if stmt.else_body:
            # COARSE 下保留一个 ELSE actions:（或你也可以只保留 ELSE）
            lines.append(NLLine("ELSE", raw=False))
        lines.append(NLLine(tpl_end_if(), raw=False))
        return NLFragment(lines)

    # MEDIUM：结构 + 摘要（IF/ELSIF/ELSE 各自一行 actions）
    if ctx.is_medium():
        lines.append(NLLine(tpl_if(cond), raw=False))
        lines.append(NLLine(tpl_then_actions(summarize_block(stmt.then_body, ctx.cfg, depth + 1)), raw=False))

        for (c, body) in (stmt.elif_branches or []):
            lines.append(NLLine(tpl_elsif(ctx.rexpr(c)), raw=False))
            lines.append(NLLine(tpl_elsif_actions(summarize_block(body, ctx.cfg, depth + 1)), raw=False))

        if stmt.else_body:
            # 不单独输出 ELSE 行，直接用 ELSE actions: ...
            lines.append(NLLine(tpl_else_actions(summarize_block(stmt.else_body, ctx.cfg, depth + 1)), raw=False))

        lines.append(NLLine(tpl_end_if(), raw=False))
        return NLFragment(lines)

    # FINE：结构 + “actions:”头 + 展开
    lines.append(NLLine(tpl_if(cond), raw=False))
    # 末尾是 ":"，不能补句号
    lines.append(NLLine(tpl_then_actions(), raw=True))
    lines.extend(_fine_expand_block(stmt.then_body, ctx, depth, emit_stmt).lines)

    for (c, body) in (stmt.elif_branches or []):
        lines.append(NLLine(tpl_elsif(ctx.rexpr(c)), raw=False))
        lines.append(NLLine(tpl_elsif_actions(), raw=True))
        lines.extend(_fine_expand_block(body, ctx, depth, emit_stmt).lines)

    if stmt.else_body:
        lines.append(NLLine(tpl_else_actions(), raw=True))
        lines.extend(_fine_expand_block(stmt.else_body, ctx, depth, emit_stmt).lines)

    lines.append(NLLine(tpl_end_if(), raw=False))
    return NLFragment(lines)


# -----------------------
# CASE rule
# -----------------------
def emit_case_rule(
    stmt: N.CaseStmt,
    ctx: EmitContext,
    depth: int,
    emit_stmt: EmitStmtFn,
    summarize_block: SummFn,
) -> NLFragment:
    lines: List[NLLine] = []
    sel = ctx.rexpr(stmt.cond)

    lines.append(NLLine(tpl_case(sel), raw=False))

    # 1) 先输出所有 WHEN 分支（不要在这里输出 ELSE）
    for e in stmt.entries:
        conds = ", ".join(c.text for c in e.conds)

        if ctx.is_coarse():
            # "WHEN ...:" 末尾 ":"，不能补句号
            lines.append(NLLine(f"WHEN {conds}:", raw=True))
            continue

        if ctx.is_medium():
            actions = summarize_block(e.body, ctx.cfg, depth + 1)
            lines.append(NLLine(tpl_when_actions(conds, actions), raw=False))
            continue

        # FINE：WHEN ...: actions:（空 actions => 末尾 ":"，raw=True）
        lines.append(NLLine(tpl_when_actions(conds), raw=True))
        lines.extend(_fine_expand_block(e.body, ctx, depth, emit_stmt).lines)

    # 2) 再输出 ELSE 分支（只输出一次，并且在所有 WHEN 之后）
    if stmt.else_body:
        if ctx.is_coarse():
            # 你当前 COARSE 的设计：不输出 actions，只输出结构也可以
            lines.append(NLLine("ELSE", raw=False))
        elif ctx.is_medium():
            actions = summarize_block(stmt.else_body, ctx.cfg, depth + 1)
            lines.append(NLLine(tpl_else_actions(actions), raw=False))
        else:
            # FINE：ELSE actions:（空 => raw=True）+ 展开块
            lines.append(NLLine(tpl_else_actions(), raw=True))
            lines.extend(_fine_expand_block(stmt.else_body, ctx, depth, emit_stmt).lines)

    lines.append(NLLine(tpl_end_case(), raw=False))
    return NLFragment(lines)

# -----------------------
# FOR rule
# -----------------------
def emit_for_rule(
    stmt: N.ForStmt,
    ctx: EmitContext,
    depth: int,
    emit_stmt: EmitStmtFn,
    summarize_block: SummFn,
) -> NLFragment:
    lines: List[NLLine] = []

    start = ctx.rexpr(stmt.start)
    end = ctx.rexpr(stmt.end)
    step = ctx.rexpr(stmt.step) if stmt.step else "1"

    lines.append(NLLine(tpl_for(stmt.var, start, end, step), raw=False))

    if ctx.is_medium():
        actions = summarize_block(stmt.body, ctx.cfg, depth + 1)
        lines.append(NLLine(tpl_loop_actions(actions), raw=False))
    elif ctx.is_fine():
        lines.append(NLLine(tpl_loop_actions(), raw=True))
        lines.extend(_fine_expand_block(stmt.body, ctx, depth, emit_stmt).lines)

    lines.append(NLLine(tpl_end_for(), raw=False))
    return NLFragment(lines)


# -----------------------
# WHILE rule
# -----------------------
def emit_while_rule(
    stmt: N.WhileStmt,
    ctx: EmitContext,
    depth: int,
    emit_stmt: EmitStmtFn,
    summarize_block: SummFn,
) -> NLFragment:
    lines: List[NLLine] = []

    cond = ctx.rexpr(stmt.cond)
    lines.append(NLLine(tpl_while(cond), raw=False))

    if ctx.is_medium():
        actions = summarize_block(stmt.body, ctx.cfg, depth + 1)
        lines.append(NLLine(tpl_loop_actions(actions), raw=False))
    elif ctx.is_fine():
        lines.append(NLLine(tpl_loop_actions(), raw=True))
        lines.extend(_fine_expand_block(stmt.body, ctx, depth, emit_stmt).lines)

    lines.append(NLLine(tpl_end_while(), raw=False))
    return NLFragment(lines)


# -----------------------
# REPEAT rule
# -----------------------
def emit_repeat_rule(
    stmt: N.RepeatStmt,
    ctx: EmitContext,
    depth: int,
    emit_stmt: EmitStmtFn,
    summarize_block: SummFn,
) -> NLFragment:
    lines: List[NLLine] = []

    until = ctx.rexpr(stmt.until)

    lines.append(NLLine(tpl_repeat(), raw=False))

    if ctx.is_medium():
        actions = summarize_block(stmt.body, ctx.cfg, depth + 1)
        lines.append(NLLine(tpl_loop_actions(actions), raw=False))
    elif ctx.is_fine():
        lines.append(NLLine(tpl_loop_actions(), raw=True))
        lines.extend(_fine_expand_block(stmt.body, ctx, depth, emit_stmt).lines)

    lines.append(NLLine(tpl_until(until), raw=False))
    lines.append(NLLine(tpl_end_repeat(), raw=False))
    return NLFragment(lines)


