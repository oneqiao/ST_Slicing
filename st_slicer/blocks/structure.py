# st_slicer/blocks/structure.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

from .core import (
    clean_st_line, norm_line, strip_st_comments,
    is_if_start, is_elsif, is_else, is_end_if,
    is_case_label_line, is_substantive_line,
    RE_ELSIF, RE_THEN,
    RE_IF_HEAD, RE_END_IF,
    RE_FOR_HEAD, RE_END_FOR,
    RE_CASE_HEAD, RE_END_CASE,
    RE_WHILE_HEAD, RE_END_WHILE,
    RE_REPEAT_HEAD, RE_END_REPEAT,
    RE_ELSE_LINE,
    FunctionalBlock,
)

# 控制结构层面的逻辑（扫描 END、扫描 IF 头、补齐 IF/CASE 结构、折叠半空 IF）。避免这些逻辑散落在渲染、行映射、后处理等位置。
# -----------------------------
# Generic end matching scanners
# -----------------------------
def scan_matching_end_generic(
    line_start: int,
    code_lines: List[str],
    head_re: re.Pattern,
    end_re: re.Pattern,
) -> Optional[int]:
    n = len(code_lines)
    if line_start < 1 or line_start > n:
        return None

    depth = 0
    for ln in range(line_start, n + 1):
        txt = clean_st_line(code_lines[ln - 1]).strip()
        if not txt:
            continue
        u = txt.upper()

        tmp = RE_ELSIF.sub("", u) if head_re is RE_IF_HEAD else u
        if head_re.search(tmp):
            depth += 1

        if end_re.search(u):
            depth -= len(end_re.findall(u))
            if depth == 0:
                return ln

    return None

def scan_matching_end_if(line_start: int, code_lines: List[str]) -> int:
    return scan_matching_end_generic(line_start, code_lines, RE_IF_HEAD, RE_END_IF) or line_start

def scan_matching_end_for(line_start: int, code_lines: List[str]) -> int:
    return scan_matching_end_generic(line_start, code_lines, RE_FOR_HEAD, RE_END_FOR) or line_start

def scan_matching_end_case(line_start: int, code_lines: List[str]) -> int:
    return scan_matching_end_generic(line_start, code_lines, RE_CASE_HEAD, RE_END_CASE) or line_start

def scan_matching_end_while(line_start: int, code_lines: List[str]) -> int:
    return scan_matching_end_generic(line_start, code_lines, RE_WHILE_HEAD, RE_END_WHILE) or line_start

def scan_matching_end_repeat(line_start: int, code_lines: List[str]) -> int:
    return scan_matching_end_generic(line_start, code_lines, RE_REPEAT_HEAD, RE_END_REPEAT) or line_start

# -----------------------------
# IF header (multiline) scan
# -----------------------------
def scan_if_header_end(line_start: int, code_lines: List[str]) -> int:
    n = len(code_lines)
    ln = line_start
    while ln <= n:
        if RE_THEN.search(clean_st_line(code_lines[ln - 1])):
            return ln
        ln += 1
    return line_start

def _extract_if_condition(if_ln: int, header_end: int, code_lines: List[str]) -> str:
    head = " ".join(code_lines[if_ln - 1: header_end])
    head = strip_st_comments(head).strip()
    head = re.sub(r"^\s*IF\s+", "", head, flags=re.IGNORECASE)
    head = re.sub(r"\bTHEN\b\s*;?\s*$", "", head, flags=re.IGNORECASE)
    return head.strip() or "TRUE"

def _find_else_line_same_level(if_ln: int, end_if_ln: int, code_lines: List[str]) -> Optional[int]:
    depth = 0
    for ln in range(if_ln, end_if_ln + 1):
        u = norm_line(code_lines[ln - 1])
        if not u:
            continue
        if is_if_start(u) and not is_elsif(u):
            depth += 1
            continue
        if is_end_if(u):
            depth -= 1
            continue
        if depth == 1 and is_else(u):
            return ln
    return None

# -----------------------------
# Fold half-empty IFs (postprocess)
# -----------------------------
def fold_half_empty_ifs_in_block(block: FunctionalBlock, code_lines: List[str]) -> None:
    if not block.line_numbers:
        return
    n = len(code_lines)
    ln_set = set(block.line_numbers)
    to_remove: Set[int] = set()

    for ln in sorted(block.line_numbers):
        if not (1 <= ln <= n):
            continue
        u = strip_st_comments(code_lines[ln - 1]).strip().upper()
        if not is_if_start(u):
            continue

        end_ln = scan_matching_end_if(ln, code_lines)
        if not (1 <= end_ln <= n) or end_ln <= ln:
            continue

        header_end = scan_if_header_end(ln, code_lines)
        else_ln = _find_else_line_same_level(ln, end_ln, code_lines)
        if else_ln is None:
            continue

        then_src_has = any(is_substantive_line(code_lines[k - 1]) for k in range(header_end + 1, else_ln))
        else_src_has = any(is_substantive_line(code_lines[k - 1]) for k in range(else_ln + 1, end_ln))

        then_blk_has = any((header_end + 1) <= k <= (else_ln - 1) for k in ln_set)
        else_blk_has = any((else_ln + 1) <= k <= (end_ln - 1) for k in ln_set)

        # A) THEN empty, ELSE non-empty
        if (not then_src_has) and else_src_has and else_blk_has:
            cond = _extract_if_condition(ln, header_end, code_lines)
            block.line_overrides[ln] = f"IF NOT ({cond}) THEN"
            for k in range(ln + 1, header_end + 1):
                if k in ln_set:
                    to_remove.add(k)
            if else_ln in ln_set:
                to_remove.add(else_ln)
            continue

        # B) ELSE empty, THEN non-empty
        if then_src_has and (not else_src_has) and then_blk_has:
            if else_ln in ln_set:
                to_remove.add(else_ln)

    if to_remove:
        block.line_numbers = sorted(x for x in block.line_numbers if x not in to_remove)

# -----------------------------
# Patch IF structure
# -----------------------------
@dataclass
class _IfFrame:
    if_line: int
    current_branch_head: int
    branches: List[Tuple[int, int, int]]  # (branch_head, body_start, body_end)

def patch_if_structure(
    sliced_lines: Iterable[int],
    code_lines: List[str],
    *,
    ensure_end_if: bool = True,
) -> Set[int]:
    n = len(code_lines)
    patched: Set[int] = {int(x) for x in sliced_lines if 1 <= int(x) <= n}
    stack: List[_IfFrame] = []

    def _branch_body_start(head_ln: int) -> int:
        t = norm_line(code_lines[head_ln - 1])
        if is_else(t):
            return head_ln + 1
        head_end = scan_if_header_end(head_ln, code_lines)
        return head_end + 1

    for ln in range(1, n + 1):
        t = norm_line(code_lines[ln - 1])
        if not t:
            continue

        if is_if_start(t) and not is_elsif(t):
            stack.append(_IfFrame(if_line=ln, current_branch_head=ln, branches=[]))
            continue

        if is_elsif(t) or is_else(t):
            if not stack:
                continue
            frame = stack[-1]
            prev_head = frame.current_branch_head
            frame.branches.append((prev_head, _branch_body_start(prev_head), ln - 1))
            frame.current_branch_head = ln
            continue

        if is_end_if(t):
            if not stack:
                continue
            frame = stack.pop()
            prev_head = frame.current_branch_head
            frame.branches.append((prev_head, _branch_body_start(prev_head), ln - 1))

            touched_any = False
            for branch_head, b_start, b_end in frame.branches:
                if b_start <= b_end:
                    touched = any((b_start <= L <= b_end) for L in patched)
                    if touched:
                        patched.add(branch_head)
                        touched_any = True

            if ensure_end_if and touched_any:
                patched.add(ln)

    return patched

# -----------------------------
# Patch CASE structure
# -----------------------------
def _scan_case_start(ln: int, code_lines: List[str]) -> int:
    for i in range(ln, 0, -1):
        if RE_CASE_HEAD.search(clean_st_line(code_lines[i - 1]).strip()):
            return i
    return ln

def patch_case_structure(
    line_numbers: Iterable[int],
    code_lines: List[str],
    ensure_end_case: bool = True,
    include_branch_headers: bool = True,
) -> Set[int]:
    n = len(code_lines)
    lines: Set[int] = {int(ln) for ln in line_numbers if 1 <= int(ln) <= n}
    if not lines:
        return lines

    case_regions: Set[Tuple[int, int]] = set()

    for ln in sorted(lines):
        txt = clean_st_line(code_lines[ln - 1]).strip()
        u = txt.upper()

        if RE_CASE_HEAD.search(u):
            cs = ln
            ce = scan_matching_end_case(cs, code_lines)
            if 1 <= cs <= ce <= n:
                case_regions.add((cs, ce))
            continue

        if RE_END_CASE.search(u):
            cs = _scan_case_start(ln, code_lines)
            if cs != ln:
                case_regions.add((cs, ln))
            continue

        if is_case_label_line(txt) or RE_ELSE_LINE.search(txt):
            cs = _scan_case_start(ln, code_lines)
            if cs != ln and RE_CASE_HEAD.search(clean_st_line(code_lines[cs - 1]).strip()):
                ce = scan_matching_end_case(cs, code_lines)
                if 1 <= cs <= ce <= n:
                    case_regions.add((cs, ce))

    if not case_regions:
        return lines

    for (cs, ce) in sorted(case_regions):
        inner_hits = sorted([x for x in lines if cs < x < ce])
        if not inner_hits:
            continue

        # source-level empty CASE check (no label/ELSE)
        has_label = False
        for k in range(cs + 1, ce):
            t = clean_st_line(code_lines[k - 1]).strip()
            if is_case_label_line(t) or RE_ELSE_LINE.search(t):
                has_label = True
                break
        if not has_label:
            lines.discard(cs)
            lines.discard(ce)
            continue

        lines.add(cs)
        if ensure_end_case:
            lines.add(ce)

        if include_branch_headers:
            for hit in inner_hits:
                hit_txt = clean_st_line(code_lines[hit - 1]).strip()
                if is_case_label_line(hit_txt) or RE_ELSE_LINE.search(hit_txt) or RE_CASE_HEAD.search(hit_txt):
                    continue
                for up in range(hit - 1, cs, -1):
                    up_txt = clean_st_line(code_lines[up - 1]).strip()
                    if is_case_label_line(up_txt) or RE_ELSE_LINE.search(up_txt):
                        lines.add(up)
                        break

    return lines
