# st_slicer/blocks/core.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# 所有基础设施（数据结构 + 文本清洗 + 正则 + 控制深度）。任何模块只要需要“识别 ST 关键字/去注释/判断行类型”，都从这里取
# -----------------------------
# Data structures
# -----------------------------
from ..ast.nodes import Stmt

@dataclass(frozen=True)
class SlicingCriterion:
    node_id: int
    kind: str
    variable: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FunctionalBlock:
    criteria: List[SlicingCriterion] = field(default_factory=list)
    node_ids: Set[int] = field(default_factory=set)
    stmts: List[Stmt] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)
    line_overrides: Dict[int, str] = field(default_factory=dict)

# -----------------------------
# ST text utilities
# -----------------------------
_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)

_BLOCK_INLINE = re.compile(r"\(\*.*?\*\)")
_LINE_INLINE = re.compile(r"//.*$")

def strip_st_comments(s: str) -> str:
    s = _BLOCK_COMMENT_RE.sub("", s)
    s = _LINE_COMMENT_RE.sub("", s)
    return s

def clean_st_line(line: str) -> str:
    line = _LINE_INLINE.sub("", line)
    line = _BLOCK_INLINE.sub("", line)
    return line

def norm_line(code_line: str) -> str:
    return strip_st_comments(code_line).strip().upper()

# heads
RE_IF_HEAD     = re.compile(r"^\s*IF\b", re.IGNORECASE)
RE_ELSIF_HEAD  = re.compile(r"^\s*ELSIF\b", re.IGNORECASE)
RE_ELSE_HEAD   = re.compile(r"^\s*ELSE\b", re.IGNORECASE)

RE_FOR_HEAD    = re.compile(r"^\s*FOR\b", re.IGNORECASE)
RE_CASE_HEAD   = re.compile(r"^\s*CASE\b", re.IGNORECASE)
RE_WHILE_HEAD  = re.compile(r"^\s*WHILE\b", re.IGNORECASE)
RE_REPEAT_HEAD = re.compile(r"^\s*REPEAT\b", re.IGNORECASE)

RE_THEN  = re.compile(r"\bTHEN\b", re.IGNORECASE)
RE_ELSIF = re.compile(r"\bELSIF\b", re.IGNORECASE)

# ends
RE_END_IF     = re.compile(r"\bEND_IF\b", re.IGNORECASE)
RE_END_FOR    = re.compile(r"\bEND_FOR\b", re.IGNORECASE)
RE_END_CASE   = re.compile(r"\bEND_CASE\b", re.IGNORECASE)
RE_END_WHILE  = re.compile(r"\bEND_WHILE\b", re.IGNORECASE)
RE_END_REPEAT = re.compile(r"\bEND_REPEAT\b", re.IGNORECASE)

# CASE label
RE_COLON  = re.compile(r":(?!\=)")
RE_ASSIGN = re.compile(r":=")

RE_ELSE_LINE = re.compile(r"^\s*ELSE\b", re.IGNORECASE)

def is_if_start(u: str) -> bool:
    t = u.strip().upper()
    return t.startswith("IF ") and (not t.startswith("ELSIF "))

def is_elsif(u: str) -> bool:
    return u.strip().upper().startswith("ELSIF ")

def is_else(u: str) -> bool:
    t = u.strip().upper()
    return t == "ELSE" or t.startswith("ELSE ")

def is_end_if(u: str) -> bool:
    return u.strip().upper().startswith("END_IF")

def is_case_label_line(text: str) -> bool:
    if RE_ASSIGN.search(text):
        return False
    return RE_COLON.search(text) is not None

def is_substantive_line(line: str) -> bool:
    return strip_st_comments(line).strip() != ""

def update_ctrl_depth(text: str, depth: int, *, clamp_negative: bool = True) -> int:
    u = clean_st_line(text).upper()

    depth -= len(RE_END_IF.findall(u))
    depth -= len(RE_END_FOR.findall(u))
    depth -= len(RE_END_CASE.findall(u))
    depth -= len(RE_END_WHILE.findall(u))
    depth -= len(RE_END_REPEAT.findall(u))

    tmp = RE_ELSIF.sub("", u)
    if RE_IF_HEAD.search(tmp):
        depth += 1
    if RE_FOR_HEAD.search(u):
        depth += 1
    if RE_CASE_HEAD.search(u):
        depth += 1
    if RE_WHILE_HEAD.search(u):
        depth += 1
    if RE_REPEAT_HEAD.search(u):
        depth += 1

    if clamp_negative and depth < 0:
        depth = 0
    return depth
