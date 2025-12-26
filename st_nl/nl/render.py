# st_nl/nl/render.py
from __future__ import annotations
from dataclasses import dataclass
from st_nl.ast import nodes as N

@dataclass(frozen=True)
class RenderCfg:
    expr_max_len: int = 80

def clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."

def render_expr(e: N.Expr, cfg: RenderCfg | None = None) -> str:
    cfg = cfg or RenderCfg()
    t = type(e).__name__

    if t == "VarRef":
        return e.name
    if t == "Literal":
        return str(e.value)
    if t == "FieldAccess":
        return clip(f"{render_expr(e.base, cfg)}.{e.field}", cfg.expr_max_len)
    if t == "ArrayAccess":
        return clip(f"{render_expr(e.base, cfg)}[{render_expr(e.index, cfg)}]", cfg.expr_max_len)
    if t == "TupleExpr":
        inner = ", ".join(render_expr(x, cfg) for x in e.items)
        return clip(f"({inner})", cfg.expr_max_len)
    if t == "UnaryOp":
        return clip(f"{e.op} {render_expr(e.operand, cfg)}", cfg.expr_max_len)
    if t == "BinOp":
        return clip(f"{render_expr(e.left, cfg)} {e.op} {render_expr(e.right, cfg)}", cfg.expr_max_len)

    if t == "CallExpr":
        args = []
        for a in e.pos_args:
            args.append(render_expr(a, cfg))
        for na in e.named_args:
            op = "=>" if getattr(na, "direction", None) == "out" else ":="
            args.append(f"{na.name} {op} {render_expr(na.value, cfg)}")
        return clip(f"{e.func}(" + ", ".join(args) + ")", cfg.expr_max_len)

    # fallback（保证不崩）
    return clip(getattr(e, "name", "") or str(e), cfg.expr_max_len)
