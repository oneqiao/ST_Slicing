#st_nl/rules/semantic_catalog.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

PLACEHOLDER_RE = re.compile(r"\{([A-Z]+)(\d+)\}")  # {IN1} {OUT2}


def norm_name(s: str) -> str:
    """统一规范化：大小写、下划线、空格等"""
    s = (s or "").strip()
    s = s.replace(" ", "").replace("-", "_")
    return s.upper()


@dataclass(frozen=True)
class SemanticEntry:
    name: str
    kind: str                       # function | fb
    summary: str
    signature: Dict[str, Any]        # inputs/outputs schema
    semantics_template: Optional[str]
    semantics_bind: str              # positional | named | mixed
    control_text_template: Optional[str]
    aliases: Tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SemanticEntry":
        sem = d.get("semantics") or {}
        ctl = d.get("control_text") or {}
        return SemanticEntry(
            name=norm_name(d["name"]),
            kind=(d.get("kind") or "function").lower(),
            summary=d.get("summary") or "",
            signature=d.get("signature") or {},
            semantics_template=sem.get("template"),
            semantics_bind=(sem.get("bind") or "mixed").lower(),
            control_text_template=ctl.get("template"),
            aliases=tuple(norm_name(x) for x in (d.get("aliases") or [])),
        )


class SemanticCatalog:
    def __init__(self, entries: List[SemanticEntry]):
        self._by_name: Dict[str, SemanticEntry] = {}
        self._alias: Dict[str, str] = {}

        for e in entries:
            self._by_name[e.name] = e
            for a in e.aliases:
                self._alias[a] = e.name

    @staticmethod
    def load_yaml(path: str) -> "SemanticCatalog":
        if yaml is None:
            raise RuntimeError("pyyaml not installed. Please pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)

        items = obj.get("entries") or []
        entries = [SemanticEntry.from_dict(x) for x in items]
        return SemanticCatalog(entries)

    def lookup(self, callee: str) -> Optional[SemanticEntry]:
        key = norm_name(callee)
        if key in self._by_name:
            return self._by_name[key]
        if key in self._alias:
            return self._by_name[self._alias[key]]
        return None

    # --------- 绑定：把 {IN1}/{OUT1} 换成当前语句实参/输出 ----------
    def bind_template(
        self,
        template: str,
        *,
        outs: List[str],
        pos_ins: List[str],
        named_ins: Dict[str, str],
        named_outs: Dict[str, str],
    ) -> str:
        """
        outs:  输出列表（按顺序 OUT1/OUT2...）
        pos_ins: 位置输入列表（按顺序 IN1/IN2...）
        named_ins/named_outs: FB 命名参数映射（IN1->expr, OUT1->var）
        """
        def repl(m: re.Match) -> str:
            base = m.group(1)  # IN / OUT
            idx = int(m.group(2))
            key = f"{base}{idx}"

            # 1) 优先命名映射（适合 FB）
            if base == "IN" and key in named_ins:
                return named_ins[key]
            if base == "OUT" and key in named_outs:
                return named_outs[key]

            # 2) 其次按顺序位置映射（适合 function）
            if base == "OUT":
                return outs[idx - 1] if 1 <= idx <= len(outs) else key
            else:
                return pos_ins[idx - 1] if 1 <= idx <= len(pos_ins) else key

        return PLACEHOLDER_RE.sub(repl, template)
