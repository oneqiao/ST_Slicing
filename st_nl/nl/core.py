# st_nl/nl/core.py
from __future__ import annotations
import re

from dataclasses import dataclass
from typing import Dict, List, Any

from st_nl.ast import nodes as N
from st_nl.nl.emitter import finalize_line
from st_nl.nl.render import render_expr

_LEAD_WS = re.compile(r"^(\s*)(.*)$")

@dataclass(frozen=True)
class NLLine:
    """
    raw=True  => 不做 finalize_line（不补句号、不做行尾标点归一）
    raw=False => 交给 finalize_line 做全局句号规范化
    """
    text: str
    raw: bool = False


@dataclass(frozen=True)
class NLFragment:
    lines: List[NLLine]


@dataclass
class EmitContext:
    """
    duck-typed cfg：只要求有 cfg.nl_level / cfg.render / cfg.fine_* 等字段
    """
    cfg: Any
    docs: Dict[str, Any]

    def rexpr(self, e: N.Expr) -> str:
        return render_expr(e, self.cfg.render)

    def level_name(self) -> str:
        lv = getattr(self.cfg, "nl_level", None)
        return getattr(lv, "name", str(lv))

    def is_coarse(self) -> bool:
        return self.level_name() == "COARSE"

    def is_medium(self) -> bool:
        return self.level_name() == "MEDIUM"

    def is_fine(self) -> bool:
        return self.level_name() == "FINE"


def indent_fragment(frag: NLFragment, indent: str) -> NLFragment:
    if not indent:
        return frag
    out = [NLLine(indent + ln.text, raw=ln.raw) for ln in frag.lines]
    return NLFragment(out)

def finalize_fragment(frag: NLFragment) -> List[str]:
    out: List[str] = []
    for ln in frag.lines:
        raw_text = ln.text or ""
        if not raw_text.strip():
            continue

        # 保留缩进
        raw_text = raw_text.rstrip("\n")
        m = _LEAD_WS.match(raw_text)
        indent, body = (m.group(1), m.group(2)) if m else ("", raw_text)

        if ln.raw:
            out.append(raw_text.rstrip())
        else:
            out.append(indent + finalize_line(body.rstrip()))
    return out
