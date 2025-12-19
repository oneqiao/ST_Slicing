# st_slicer/blocks/pipeline.py
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from ..slicer import backward_slice
from ..ast.nodes import (
    Stmt, IfStmt, ForStmt, CaseStmt, WhileStmt, RepeatStmt,
    Assignment, CallStmt,
    Expr, VarRef, ArrayAccess, FieldAccess, Literal, BinOp, CallExpr,
)

from .core import (
    FunctionalBlock, SlicingCriterion,
    clean_st_line, strip_st_comments, norm_line,
    is_if_start, is_case_label_line,
    RE_CASE_HEAD, RE_ELSE_LINE,
    RE_FOR_HEAD, RE_WHILE_HEAD, RE_REPEAT_HEAD,
    update_ctrl_depth,
)

from .structure import (
    scan_if_header_end,
    scan_matching_end_if, scan_matching_end_case,
    scan_matching_end_for, scan_matching_end_while, scan_matching_end_repeat,
    patch_if_structure, patch_case_structure,
    patch_loop_structures,
    fold_half_empty_ifs_in_block,
)

# 端到端流水线所需的功能：行号映射、PDG 切片/聚类、node→stmt→line、stage/size 切分、变量收集、后处理（删空结构/去重/质量判定）。上层主流程基本只需要 import 这里的函数。
# =========================================================
# 1) AST stmts -> line numbers
# =========================================================
def _scan_stmt_end(line_start: int, code_lines: List[str]) -> int:
    n = len(code_lines)
    if line_start < 1 or line_start > n:
        return line_start

    paren = 0
    for ln in range(line_start, n + 1):
        t = clean_st_line(code_lines[ln - 1])
        if not t.strip():
            continue
        paren += t.count("(")
        paren -= t.count(")")
        if ";" in t and paren <= 0:
            return ln
    return line_start

def stmts_to_line_numbers(stmts: List[Stmt], code_lines: List[str]) -> List[int]:
    sliced: Set[int] = set()
    n = len(code_lines)

    for st in stmts:
        line_no = getattr(st.loc, "line", None)
        if line_no is None or not (1 <= line_no <= n):
            continue

        if isinstance(st, IfStmt):
            header_end = scan_if_header_end(line_no, code_lines)
            for ln in range(line_no, min(header_end, n) + 1):
                sliced.add(ln)
            end_if = scan_matching_end_if(line_no, code_lines)
            if 1 <= end_if <= n:
                sliced.add(end_if)

        elif isinstance(st, CaseStmt):
            sliced.add(line_no)
            end_ln = scan_matching_end_case(line_no, code_lines)
            if 1 <= end_ln <= n:
                sliced.add(end_ln)

        elif isinstance(st, ForStmt):
            sliced.add(line_no)
            end_ln = scan_matching_end_for(line_no, code_lines)
            if 1 <= end_ln <= n:
                sliced.add(end_ln)

        elif isinstance(st, WhileStmt):
            sliced.add(line_no)
            end_ln = scan_matching_end_while(line_no, code_lines)
            if 1 <= end_ln <= n:
                sliced.add(end_ln)

        elif isinstance(st, RepeatStmt):
            sliced.add(line_no)
            end_ln = scan_matching_end_repeat(line_no, code_lines)
            if 1 <= end_ln <= n:
                sliced.add(end_ln)

        else:
            end_ln = _scan_stmt_end(line_no, code_lines)
            for ln in range(line_no, min(end_ln, n) + 1):
                sliced.add(ln)

    return sorted(sliced)

# =========================================================
# 2) PDG slice + clustering
# =========================================================
def compute_slice_nodes(
    prog_pdg,
    start_nodes: Union[int, Iterable[int], None],
    *,
    var_sensitive: bool = False,
    seed_vars: Optional[Iterable[str]] = None,
    use_data: bool = True,
    use_control: bool = True,
) -> Set[int]:
    if start_nodes is None:
        starts: List[int] = []
    elif isinstance(start_nodes, int):
        starts = [start_nodes]
    else:
        starts = list(start_nodes)

    if not starts:
        return set()

    return backward_slice(
        prog_pdg,
        starts,
        use_data=use_data,
        use_control=use_control,
        var_sensitive=var_sensitive,
        seed_vars=seed_vars,
    )


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union if union else 0.0

def cluster_slices(
    all_slices: List[Tuple[SlicingCriterion, Set[int]]],
    overlap_threshold: float = 0.35,
) -> List[dict]:
    n = len(all_slices)
    if n == 0:
        return []

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    node_sets = [ns for _, ns in all_slices]
    for i in range(n):
        si = node_sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = node_sets[j]
            if not sj:
                continue
            if _jaccard(si, sj) >= overlap_threshold:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters: List[dict] = []
    for idxs in groups.values():
        merged_nodes: Set[int] = set()
        merged_criteria: List[SlicingCriterion] = []
        for k in idxs:
            crit, ns = all_slices[k]
            merged_nodes |= ns
            merged_criteria.append(crit)
        clusters.append({"nodes": merged_nodes, "criteria": merged_criteria})

    return clusters

# =========================================================
# 3) Node ids -> AST stmts with parent closure
# =========================================================
def build_parent_map_from_ir2ast(ir2ast_stmt: List[Stmt]) -> Dict[Stmt, Optional[Stmt]]:
    uniq: Set[Stmt] = {st for st in ir2ast_stmt if st is not None}
    parent: Dict[Stmt, Optional[Stmt]] = {}
    referenced: Set[Stmt] = set()

    def children_of(stmt: Stmt) -> List[Stmt]:
        out: List[Stmt] = []
        if isinstance(stmt, IfStmt):
            out.extend(stmt.then_body or [])
            for _c, body in (stmt.elif_branches or []):
                out.extend(body or [])
            out.extend(stmt.else_body or [])
        elif isinstance(stmt, ForStmt):
            out.extend(stmt.body or [])
        elif isinstance(stmt, CaseStmt):
            for entry in (stmt.entries or []):
                out.extend(entry.body or [])
            out.extend(stmt.else_body or [])
        elif isinstance(stmt, WhileStmt):
            out.extend(stmt.body or [])
        elif isinstance(stmt, RepeatStmt):
            out.extend(stmt.body or [])
        return out

    for st in list(uniq):
        for ch in children_of(st):
            if ch in uniq:
                referenced.add(ch)

    roots = [st for st in uniq if st not in referenced]

    def visit(st: Stmt, p: Optional[Stmt]) -> None:
        if st not in uniq:
            return
        if st not in parent:
            parent[st] = p
        elif parent[st] is None and p is not None:
            parent[st] = p
        for ch in children_of(st):
            visit(ch, st)

    for r in roots:
        visit(r, None)

    for st in uniq:
        parent.setdefault(st, None)

    return parent

def close_with_control_structures(stmt_set: Set[Stmt], parent_map: Dict[Stmt, Optional[Stmt]]) -> Set[Stmt]:
    closed: Set[Stmt] = set(stmt_set)
    work = list(stmt_set)

    while work:
        st = work.pop()
        p = parent_map.get(st)
        while p is not None:
            if isinstance(p, (IfStmt, ForStmt, CaseStmt, WhileStmt, RepeatStmt)) and p not in closed:
                closed.add(p)
                work.append(p)
            p = parent_map.get(p)

    return closed

def nodes_to_sorted_ast_stmts(
    node_ids: Set[int],
    ir2ast_stmt: List[Stmt],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[Stmt]:
    stmt_set: Set[Stmt] = set()
    for nid in node_ids:
        if 0 <= nid < len(ir2ast_stmt):
            st = ir2ast_stmt[nid]
            if st is not None:
                stmt_set.add(st)

    stmt_set = close_with_control_structures(stmt_set, parent_map)
    return sorted(stmt_set, key=lambda s: (getattr(s.loc, "line", 0), getattr(s.loc, "column", 0)))

# =========================================================
# 4) Splitters (stage + size)
# =========================================================
_STAGE_IF_RE = re.compile(r"\bIF\b.*\b{NAME}\b\s*=\s*[\w\.#]+", re.IGNORECASE)
_STAGE_CASE_RE = re.compile(r"^\s*CASE\b.*\b{NAME}\b.*\bOF\b", re.IGNORECASE)
_STAGE_ASSIGN_RE = re.compile(r"\b{NAME}\b\s*:=\s*[\w\.#]+", re.IGNORECASE)

def _build_block_from_lines(
    parent_block: FunctionalBlock,
    seg_lines: List[int],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> FunctionalBlock:
    seg_set = set(seg_lines)

    sub_node_ids: Set[int] = set()
    for nid in parent_block.node_ids:
        if 0 <= nid < len(ir2ast_stmt):
            st = ir2ast_stmt[nid]
            if st is None:
                continue
            st_lines = set(stmts_to_line_numbers([st], code_lines))
            if st_lines & seg_set:
                sub_node_ids.add(nid)

    sub_stmts = nodes_to_sorted_ast_stmts(sub_node_ids, ir2ast_stmt, parent_map)

    base_lines = stmts_to_line_numbers(sub_stmts, code_lines)
    fixed = patch_if_structure(base_lines, code_lines, ensure_end_if=True)
    fixed = patch_case_structure(fixed, code_lines, ensure_end_case=True, include_branch_headers=True)
    fixed = patch_loop_structures(fixed, code_lines, include_header_span=True, include_until_span=True)

    return FunctionalBlock(
        criteria=list(parent_block.criteria),
        node_ids=sub_node_ids,
        stmts=sub_stmts,
        line_numbers=sorted(fixed),
    )

def _split_block_by_stage(
    block: FunctionalBlock,
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
    min_lines: int,
    stage_var_names: Tuple[str, ...],
) -> List[FunctionalBlock]:
    lines = sorted(block.line_numbers)
    if not lines:
        return [block]

    def is_stage_marker(line_text: str) -> bool:
        lt = line_text.strip()
        if not lt:
            return False
        for name in stage_var_names:
            pat_if = _STAGE_IF_RE.pattern.format(NAME=re.escape(name))
            pat_case = _STAGE_CASE_RE.pattern.format(NAME=re.escape(name))
            pat_asg = _STAGE_ASSIGN_RE.pattern.format(NAME=re.escape(name))
            if re.search(pat_if, lt, re.IGNORECASE):
                return True
            if re.search(pat_case, lt, re.IGNORECASE):
                return True
            if re.search(pat_asg, lt, re.IGNORECASE):
                return True
        return False

    segments: List[List[int]] = []
    current: List[int] = []
    saw_marker = False

    for ln in lines:
        text = code_lines[ln - 1] if 1 <= ln <= len(code_lines) else ""
        if is_stage_marker(text):
            saw_marker = True
            if current:
                segments.append(current)
            current = [ln]
        else:
            (current.append(ln) if current else (current := [ln]))  # py3.8+ ok

    if current:
        segments.append(current)

    if not saw_marker:
        return [block]

    merged: List[List[int]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        elif len(seg) < min_lines:
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    out: List[FunctionalBlock] = []
    for seg in merged:
        seg = sorted(set(seg))
        if len(seg) < max(1, min_lines // 2):
            continue
        out.append(_build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map))
    return out or [block]

def split_blocks_by_stage(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
    min_lines: int = 5,
    stage_var_names: Tuple[str, ...] = ("stage", "Stage", "state", "State"),
) -> List[FunctionalBlock]:
    out: List[FunctionalBlock] = []
    for b in blocks:
        out.extend(_split_block_by_stage(b, ir2ast_stmt, code_lines, parent_map, min_lines, stage_var_names))
    return out

def _split_block_by_size(
    block: FunctionalBlock,
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    lines = sorted(block.line_numbers)
    if len(lines) <= max_lines:
        return [block]

    segments: List[List[int]] = []
    current: List[int] = []
    ctrl_depth = 0

    for ln in lines:
        if not current:
            current = [ln]
        else:
            if ln != current[-1] + 1 and len(current) >= min_lines and ctrl_depth == 0:
                segments.append(current)
                current = [ln]
            else:
                current.append(ln)

        if 1 <= ln <= len(code_lines):
            ctrl_depth = update_ctrl_depth(code_lines[ln - 1], ctrl_depth, clamp_negative=True)

        if len(current) >= max_lines and ctrl_depth == 0:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    merged: List[List[int]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        elif len(seg) < min_lines:
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    out: List[FunctionalBlock] = []
    for seg in merged:
        seg = sorted(set(seg))
        if len(seg) < max(1, min_lines // 2):
            continue
        out.append(_build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map))
    return out or [block]

def normalize_block_sizes(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    out: List[FunctionalBlock] = []
    for b in blocks:
        if len(b.line_numbers) > max_lines:
            out.extend(_split_block_by_size(b, ir2ast_stmt, code_lines, min_lines, max_lines, parent_map))
        else:
            out.append(b)
    return out

def normalize_and_split_blocks(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    stage_blocks = split_blocks_by_stage(blocks, ir2ast_stmt, code_lines, parent_map, min_lines=min_lines)
    return normalize_block_sizes(stage_blocks, ir2ast_stmt, code_lines, min_lines, max_lines, parent_map)

# =========================================================
# 5) Variable collection (declaration closure)
# =========================================================
def collect_vars_in_expr(expr: Optional[Expr], vars_used: Set[str], funcs_used: Optional[Set[str]] = None) -> None:
    if expr is None:
        return
    if isinstance(expr, VarRef):
        vars_used.add(expr.name); return
    if isinstance(expr, ArrayAccess):
        collect_vars_in_expr(expr.base, vars_used, funcs_used)
        collect_vars_in_expr(expr.index, vars_used, funcs_used); return
    if isinstance(expr, FieldAccess):
        collect_vars_in_expr(expr.base, vars_used, funcs_used); return
    if isinstance(expr, BinOp):
        collect_vars_in_expr(expr.left, vars_used, funcs_used)
        collect_vars_in_expr(expr.right, vars_used, funcs_used); return
    if isinstance(expr, CallExpr):
        if funcs_used is not None:
            funcs_used.add(expr.func)
        for a in expr.args:
            collect_vars_in_expr(a, vars_used, funcs_used)
        return
    if isinstance(expr, Literal):
        return

def collect_vars_in_stmt(stmt: Stmt, vars_used: Set[str], funcs_used: Optional[Set[str]] = None) -> None:
    if isinstance(stmt, Assignment):
        collect_vars_in_expr(stmt.target, vars_used, funcs_used)
        collect_vars_in_expr(stmt.value, vars_used, funcs_used)
        return
    if isinstance(stmt, IfStmt):
        collect_vars_in_expr(stmt.cond, vars_used, funcs_used)
        for s in stmt.then_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        for cond, body in stmt.elif_branches:
            collect_vars_in_expr(cond, vars_used, funcs_used)
            for s in body:
                collect_vars_in_stmt(s, vars_used, funcs_used)
        for s in stmt.else_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        return
    if isinstance(stmt, ForStmt):
        vars_used.add(stmt.var)
        collect_vars_in_expr(stmt.start, vars_used, funcs_used)
        collect_vars_in_expr(stmt.end, vars_used, funcs_used)
        collect_vars_in_expr(stmt.step, vars_used, funcs_used)
        for s in stmt.body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        return
    if isinstance(stmt, WhileStmt):
        collect_vars_in_expr(stmt.cond, vars_used, funcs_used)
        for s in stmt.body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        return
    if isinstance(stmt, RepeatStmt):
        for s in stmt.body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        collect_vars_in_expr(stmt.until, vars_used, funcs_used)
        return
    if isinstance(stmt, CaseStmt):
        collect_vars_in_expr(stmt.cond, vars_used, funcs_used)
        for entry in stmt.entries:
            for s in entry.body:
                collect_vars_in_stmt(s, vars_used, funcs_used)
        for s in stmt.else_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        return
    if isinstance(stmt, CallStmt):
        fb = getattr(stmt, "fb_name", None)
        if fb:
            vars_used.add(fb)
        for a in stmt.args:
            collect_vars_in_expr(a, vars_used, funcs_used)
        return

def collect_vars_in_block(stmts: List[Stmt]) -> Set[str]:
    out: Set[str] = set()
    for s in stmts:
        collect_vars_in_stmt(s, out)
    return out

# =========================================================
# 6) Postprocess (empty structures, dedup, meaningful)
# =========================================================
def is_meaningful_block(block, code_lines, min_len=8):
    lines = sorted(set(block.line_numbers or []))
    if len(lines) < min_len:
        return False

    # 统计非空、非注释行
    def is_code(s: str) -> bool:
        t = s.strip()
        if not t:
            return False
        if t.startswith("//"):
            return False
        if t.startswith("(*") and t.endswith("*)"):
            return False
        return True

    texts = [code_lines[i-1] for i in lines if 1 <= i <= len(code_lines)]
    texts = [t for t in texts if is_code(t)]

    # 关键：全是结构性关键字而没有“动作语句”，判为无意义
    structural = ("IF", "THEN", "ELSE", "ELSIF", "END_IF", "CASE", "OF", "END_CASE",
                  "FOR", "TO", "DO", "END_FOR", "WHILE", "END_WHILE", "REPEAT", "UNTIL", "END_REPEAT")
    action_hits = 0
    for t in texts:
        tt = t.strip()
        if ":=" in tt:
            action_hits += 1
        elif "(" in tt and ")" in tt and not any(tt.startswith(k) for k in structural):
            # 粗略认为函数/FB 调用
            action_hits += 1
        elif "stage :=" in tt or "stage :=" in tt.replace(" ", ""):
            action_hits += 1

    if action_hits == 0:
        return False

    return True


def _has_nonblank_inside(ln_set: Set[int], code_lines: List[str], a: int, b: int) -> bool:
    for ln in ln_set:
        if a < ln < b:
            if strip_st_comments(code_lines[ln - 1]).strip() != "":
                return True
    return False

def remove_empty_ifs_in_blocks(blocks: List[FunctionalBlock], code_lines: List[str]) -> List[FunctionalBlock]:
    n = len(code_lines)
    for b in blocks:
        if not b.line_numbers:
            continue
        ln_set = set(b.line_numbers)
        to_remove: Set[int] = set()
        for ln in sorted(b.line_numbers):
            if not (1 <= ln <= n):
                continue
            u = strip_st_comments(code_lines[ln - 1]).strip().upper()
            if not is_if_start(u):
                continue
            end_ln = scan_matching_end_if(ln, code_lines)
            if not (1 <= end_ln <= n) or end_ln <= ln:
                continue
            if not _has_nonblank_inside(ln_set, code_lines, ln, end_ln):
                to_remove.add(ln)
                if end_ln in ln_set:
                    to_remove.add(end_ln)
        if to_remove:
            b.line_numbers = sorted(x for x in b.line_numbers if x not in to_remove)
    return blocks

def remove_empty_loops_in_blocks(blocks: List[FunctionalBlock], code_lines: List[str]) -> List[FunctionalBlock]:
    n = len(code_lines)
    for b in blocks:
        if not b.line_numbers:
            continue
        ln_set = set(b.line_numbers)
        to_remove: Set[int] = set()
        for ln in sorted(b.line_numbers):
            if not (1 <= ln <= n):
                continue
            u = norm_line(code_lines[ln - 1])
            if RE_FOR_HEAD.search(u):
                end_ln = scan_matching_end_for(ln, code_lines)
                if 1 <= end_ln <= n and not _has_nonblank_inside(ln_set, code_lines, ln, end_ln):
                    to_remove.add(ln); 
                    if end_ln in ln_set: to_remove.add(end_ln)
            if RE_WHILE_HEAD.search(u):
                end_ln = scan_matching_end_while(ln, code_lines)
                if 1 <= end_ln <= n and not _has_nonblank_inside(ln_set, code_lines, ln, end_ln):
                    to_remove.add(ln); 
                    if end_ln in ln_set: to_remove.add(end_ln)
            if RE_REPEAT_HEAD.search(u):
                end_ln = scan_matching_end_repeat(ln, code_lines)
                if 1 <= end_ln <= n and not _has_nonblank_inside(ln_set, code_lines, ln, end_ln):
                    to_remove.add(ln); 
                    if end_ln in ln_set: to_remove.add(end_ln)
        if to_remove:
            b.line_numbers = sorted(x for x in b.line_numbers if x not in to_remove)
    return blocks

def remove_empty_cases_in_blocks(blocks: List[FunctionalBlock], code_lines: List[str]) -> List[FunctionalBlock]:
    n = len(code_lines)

    def src_case_has_label(cs: int, ce: int) -> bool:
        for ln in range(cs + 1, ce):
            t = clean_st_line(code_lines[ln - 1]).strip()
            if is_case_label_line(t) or RE_ELSE_LINE.search(t):
                return True
        return False

    for b in blocks:
        lnset = set(x for x in b.line_numbers if 1 <= x <= n)
        case_pairs: List[Tuple[int, int]] = []
        for ln in sorted(list(lnset)):
            if RE_CASE_HEAD.search(clean_st_line(code_lines[ln - 1]).strip()):
                cs = ln
                ce = scan_matching_end_case(cs, code_lines)
                if ce in lnset and cs < ce:
                    case_pairs.append((cs, ce))
        for cs, ce in case_pairs:
            if not src_case_has_label(cs, ce):
                lnset.discard(cs); lnset.discard(ce)
        b.line_numbers = sorted(lnset)

    return blocks

def dedup_blocks_by_code(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
    *,
    overlap_jaccard: float = 0.0,   # 0.0 表示不启用 overlap 过滤
    prefer_larger: bool = True,     # overlap 冲突时优先保留更大的块
) -> List[FunctionalBlock]:
    """
    集中处理块去重/去冗余：
    1) 严格去重：按“归一化文本”去重（你原来的逻辑）
    2) 可选 overlap 过滤：按 line_numbers 的 Jaccard 相似度去冗余（全量生成后收敛）
    """
    n = len(code_lines)
    if not blocks:
        return []

    # ---- 1) 严格去重（沿用你原本的 clean_st_line 方案）----
    seen: Set[str] = set()
    uniq: List[FunctionalBlock] = []

    for b in blocks:
        if not b.line_numbers:
            continue
        body = []
        for ln in sorted(set(b.line_numbers)):
            if 1 <= ln <= n:
                t = clean_st_line(code_lines[ln - 1]).strip()
                if t:
                    body.append(t)
        key = "\n".join(body).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(b)

    # ---- 2) overlap 去冗余（全量生成后过滤重复度过高的块）----
    if overlap_jaccard and overlap_jaccard > 0.0 and len(uniq) > 1:
        def jaccard(a: Set[int], b: Set[int]) -> float:
            if not a and not b:
                return 1.0
            inter = len(a & b)
            if inter == 0:
                return 0.0
            union = len(a | b)
            return inter / union if union else 0.0

        # 排序：默认优先保留更大的块（覆盖更完整）
        uniq.sort(key=lambda b: len(b.line_numbers or []), reverse=prefer_larger)

        kept: List[FunctionalBlock] = []
        kept_sets: List[Set[int]] = []

        for b in uniq:
            s = set(b.line_numbers or [])
            redundant = False
            for ks in kept_sets:
                if jaccard(s, ks) >= overlap_jaccard:
                    redundant = True
                    break
            if not redundant:
                kept.append(b)
                kept_sets.append(s)

        uniq = kept

    return uniq


def postprocess_blocks(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
    *,
    remove_empty_structures: bool = True,
    fold_half_empty_ifs: bool = True,
    dedup: bool = True,
) -> List[FunctionalBlock]:
    out = blocks
    if remove_empty_structures:
        out = remove_empty_ifs_in_blocks(out, code_lines)
        out = remove_empty_loops_in_blocks(out, code_lines)
        out = remove_empty_cases_in_blocks(out, code_lines)
    if fold_half_empty_ifs:
        for b in out:
            fold_half_empty_ifs_in_block(b, code_lines)
    if dedup:
        out = dedup_blocks_by_code(out, code_lines)
    return out
