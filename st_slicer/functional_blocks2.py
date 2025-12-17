# st_slicer/functional_blocks.py

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import  Any, List, Set, Tuple, Iterable, Dict, Optional
from .slicer import backward_slice   # 如果已经在上面 import 过就不用重复
from .criteria import SlicingCriterion
from .ast.nodes import Stmt, VarRef, ArrayAccess, FieldAccess, Literal, BinOp, CallExpr, CaseStmt, WhileStmt, RepeatStmt
import re

from .ast.nodes import (
    Expr,
    VarRef,
    ArrayAccess,
    FieldAccess,
    Literal,
    BinOp,
    Stmt,
    Assignment,
    IfStmt,
    ForStmt,
    CallStmt,
)
from .slicer import backward_slice
from .criteria import SlicingCriterion

# --- 轻量级：去掉 ST 注释，便于识别关键字 ---
_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)

# 清理：去掉行内 (*...*) 注释，去掉 // 后内容
_BLOCK_INLINE = re.compile(r"\(\*.*?\*\)")
_LINE_COMMENT = re.compile(r"//.*$")

def _clean_st_line(line: str) -> str:
    line = _LINE_COMMENT.sub("", line)
    line = _BLOCK_INLINE.sub("", line)
    return line

# 头部正则（词边界）
_RE_IF_HEAD     = re.compile(r"^\s*IF\b", re.IGNORECASE)
_RE_ELSIF_HEAD  = re.compile(r"^\s*ELSIF\b", re.IGNORECASE)
_RE_ELSE_HEAD   = re.compile(r"^\s*ELSE\b", re.IGNORECASE)

_RE_FOR_HEAD    = re.compile(r"^\s*FOR\b", re.IGNORECASE)
_RE_CASE_HEAD   = re.compile(r"^\s*CASE\b", re.IGNORECASE)
_RE_WHILE_HEAD  = re.compile(r"^\s*WHILE\b", re.IGNORECASE)
_RE_REPEAT_HEAD = re.compile(r"^\s*REPEAT\b", re.IGNORECASE)

_RE_ELSIF = re.compile(r"\bELSIF\b", re.IGNORECASE)
_RE_THEN = re.compile(r"\bTHEN\b", re.IGNORECASE)

# END_* 正则
_RE_END_IF = re.compile(r"\bEND_IF\b", re.IGNORECASE)
_RE_END_FOR = re.compile(r"\bEND_FOR\b", re.IGNORECASE)
_RE_END_CASE = re.compile(r"\bEND_CASE\b", re.IGNORECASE)
# 分支 label（形如： 1: / 3,4: / 3(*...*), 4(*...*) : / 1..5: 等）
# 只要行里有 ":" 且不像赋值 ":="，我们就认为它是 label 行（保守）
_RE_COLON = re.compile(r":(?!\=)")
_RE_ASSIGN = re.compile(r":=")
# ELSE 分支头（可能是 "ELSE" 或 "ELSE:" 或 "ELSE :"）
_RE_ELSE = re.compile(r"^\s*ELSE\b", re.IGNORECASE)

_RE_END_WHILE = re.compile(r"\bEND_WHILE\b", re.IGNORECASE)
_RE_END_REPEAT = re.compile(r"\bEND_REPEAT\b", re.IGNORECASE)

def _strip_st_comments(s: str) -> str:
    s = _BLOCK_COMMENT_RE.sub("", s)
    s = _LINE_COMMENT_RE.sub("", s)
    return s


def _norm_line(code_line: str) -> str:
    """去注释、去空白、统一大写，便于关键字识别。"""
    t = _strip_st_comments(code_line).strip()
    return t.upper()


def _is_if_start(t: str) -> bool:
    """
    判断 IF 头开始（排除 ELSIF）。
    兼容：IF ... THEN（THEN 可以在本行出现；多行 THEN 的情况由 _scan_if_header_end 处理）
    这里只要识别到以 IF 开头即可入栈，END/THEN 细节不在此处强依赖。
    """
    if not t.startswith("IF "):
        return False
    if t.startswith("ELSIF "):
        return False
    return True


def _is_elsif(t: str) -> bool:
    return t.startswith("ELSIF ")


def _is_else(t: str) -> bool:
    # ST 中 ELSE 通常是单词，后面可能有注释
    return t == "ELSE" or t.startswith("ELSE ")


def _is_end_if(t: str) -> bool:
    # 允许 END_IF; 或 END_IF
    return t.startswith("END_IF")

def _is_if_head(line: str) -> bool:
    """Return True if a source line is an IF header (IF ... THEN), excluding ELSIF."""
    u = _clean_st_line(line).upper()
    # 排除 ELSIF 作为 IF 头
    u2 = _RE_ELSIF.sub("", u)
    return _RE_IF_HEAD.search(u2) is not None


def _is_case_head(line: str) -> bool:
    """Return True if a source line is a CASE header (CASE ... OF)."""
    u = _clean_st_line(line).upper()
    return _RE_CASE_HEAD.search(u) is not None

@dataclass
class _IfFrame:
    if_line: int                          # IF 起始行（IF 关键行）
    current_branch_head: int              # 当前分支头行（IF/ELSIF/ELSE 的行号）
    branches: List[tuple]                 # (branch_head, body_start, body_end)


@dataclass
class FunctionalBlock:
    """
    一个功能块的抽象：由若干切片准则 + 节点集合 + AST 语句 + 源码行号组成。
    后续你可以在这里加：vars_used, var_decls, block_program_ast 等字段。
    """
    criteria: List[SlicingCriterion] = field(default_factory=list)
    node_ids: Set[int] = field(default_factory=set)
    stmts: List[Stmt] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)


# 低层工具函数

def compute_slice_nodes(prog_pdg, start_node_id: int) -> Set[int]:
    """
    对给定起始节点做一次后向切片。
    如需按变量过滤，可在这里扩展；目前直接复用 backward_slice。
    """
    return backward_slice(prog_pdg, [start_node_id])


def cluster_slices(
    all_slices: List[Tuple[SlicingCriterion, Set[int]]],
    overlap_threshold: float = 0.5,
) -> List[dict]:
    """
    输入: all_slices = [(criterion, node_set), ...]
    输出: clusters = [
        {
            "nodes": set[int],                 # 该簇中所有节点的并集
            "criteria": [criterion, ...],      # 属于这个簇的所有准则
        },
        ...
    ]
    overlap_threshold: 两个切片的重叠比例 >= 此阈值时归为同一簇。
    """
    clusters: List[dict] = []

    for crit, node_set in all_slices:
        placed = False
        for cluster in clusters:
            cluster_nodes: Set[int] = cluster["nodes"]
            inter = len(cluster_nodes & node_set)
            denom = min(len(cluster_nodes), len(node_set))
            if denom == 0:
                continue
            overlap = inter / denom
            if overlap >= overlap_threshold:
                cluster["nodes"] |= node_set
                cluster["criteria"].append(crit)
                placed = True
                break

        if not placed:
            clusters.append(
                {
                    "nodes": set(node_set),
                    "criteria": [crit],
                }
            )

    return clusters

def close_with_control_structures(
    stmt_set: Set[Stmt],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> Set[Stmt]:
    """
    在原有语句集合基础上，补全所有必要的控制语句（IfStmt / ForStmt 的外层骨架）。

    做法：
      - 对集合里的每个语句，沿 parent_map 一路向上找；
      - 遇到 IfStmt 或 ForStmt，就加入 closed 集合；
      - 继续往上，直到 None。
    """
    closed: Set[Stmt] = set(stmt_set)
    worklist: List[Stmt] = list(stmt_set)

    while worklist:
        st = worklist.pop()
        p = parent_map.get(st)
        while p is not None:
            if isinstance(p, (IfStmt, ForStmt, CaseStmt, WhileStmt, RepeatStmt)) and p not in closed:
                closed.add(p)
                worklist.append(p)
            p = parent_map.get(p)

    return closed


def nodes_to_sorted_ast_stmts(
    cluster_nodes: Set[int],
    ir2ast_stmt: List[Stmt],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[Stmt]:
    """
    从节点集合映射到 AST 语句集合，并按源码行号排序。
    会自动补全必要的 IfStmt / ForStmt 之类控制结构。
    """
    stmt_set: Set[Stmt] = set()
    for nid in cluster_nodes:
        if 0 <= nid < len(ir2ast_stmt):
            ast_stmt = ir2ast_stmt[nid]
            if ast_stmt is not None:
                stmt_set.add(ast_stmt)

    # 结构闭包：补全所需的控制结构骨架
    stmt_set = close_with_control_structures(stmt_set, parent_map)

    stmts = sorted(
        stmt_set,
        key=lambda s: (getattr(s.loc, "line", 0), getattr(s.loc, "column", 0)),
    )
    return stmts


def build_parent_map_from_ir2ast(ir2ast_stmt: List[Stmt]) -> Dict[Stmt, Optional[Stmt]]:
    """
    基于 ir2ast_stmt 粗略构造一个 parent_map: child_stmt -> parent_stmt (IfStmt / ForStmt / None)。
    """
    uniq_stmts: Set[Stmt] = {st for st in ir2ast_stmt if st is not None}
    parent: Dict[Stmt, Optional[Stmt]] = {}

    def visit(stmt: Stmt, parent_stmt: Optional[Stmt]) -> None:
        # 只关心出现在 uniq_stmts 里的语句
        if stmt not in uniq_stmts:
            return

        # 允许从 None 升级到一个真实的 parent
        if stmt not in parent or (parent[stmt] is None and parent_stmt is not None):
            parent[stmt] = parent_stmt

        # 向下递归处理控制结构
        if isinstance(stmt, IfStmt):
            for child in stmt.then_body:
                visit(child, stmt)
            for _cond, body in stmt.elif_branches:
                for child in body:
                    visit(child, stmt)
            for child in stmt.else_body:
                visit(child, stmt)

        elif isinstance(stmt, ForStmt):
            for child in stmt.body:
                visit(child, stmt)
        
        elif isinstance(stmt, CaseStmt):
            for entry in stmt.entries:
                for child in entry.body:
                    visit(child, stmt)
            for child in stmt.else_body:
                visit(child, stmt)

        elif isinstance(stmt, WhileStmt):
            for child in stmt.body:
                visit(child, stmt)

        elif isinstance(stmt, RepeatStmt):
            for child in stmt.body:
                visit(child, stmt)


    # 把所有语句都当作「潜在根」跑一遍 visit
    for st in uniq_stmts:
        visit(st, None)

    return parent

def _scan_if_header_end(line_start: int, code_lines: List[str]) -> int:
    """
    从 IF/ELSIF 语句起始行向下扫描，直到遇到包含 THEN 的行，返回该行号。
    改进点：
      - 去除行内注释与 // 注释，避免注释中的 THEN 误触发
      - 使用词边界匹配 THEN，减少误判
    """
    n = len(code_lines)
    ln = line_start
    while ln <= n:
        raw = code_lines[ln - 1]
        text = _clean_st_line(raw)  # 需要你已有的 _clean_st_line
        if _RE_THEN.search(text):
            return ln
        ln += 1
    return line_start

def _scan_matching_end_generic(
    line_start: int,
    code_lines: List[str],
    head_re: re.Pattern,
    end_re: re.Pattern,
) -> Optional[int]:
    """
    从 line_start(1-based) 开始向下扫描，找到与 head_re 对应的 end_re。
    支持同类嵌套（例如 IF 内嵌套 IF；FOR 内嵌套 FOR 等）。
    为稳健起见：
      - depth 初始为 0
      - 扫描过程中先检测头部 depth += k，再检测 end depth -= k
      - 当 depth 回到 0 时返回 end 行
    """
    n = len(code_lines)
    if line_start < 1 or line_start > n:
        return None

    depth = 0
    for ln in range(line_start, n + 1):
        raw = code_lines[ln - 1]
        txt = _clean_st_line(raw).strip()
        if not txt:
            continue
        u = txt.upper()

        # 头部（IF 要排除 ELSIF）
        tmp = _RE_ELSIF.sub("", u) if head_re is _RE_IF_HEAD else u
        if head_re.search(tmp):
            depth += 1

        # 尾部
        if end_re.search(u):
            depth -= len(end_re.findall(u))
            if depth == 0:
                return ln

    return None


# def _scan_matching_end_if(line_start: int, code_lines: List[str]) -> int:
#     """
#     从 IF 语句起始行向下扫描，使用简单深度计数找到匹配的 END_IF 行号。
#     支持嵌套 IF，并且不会把 ELSIF 误算成新的 IF。
#     """
#     n = len(code_lines)
#     depth = 0
#     for ln in range(line_start, n + 1):
#         text = code_lines[ln - 1].upper().strip()

#         # 先处理 END_IF，防止负深度
#         if "END_IF" in text:
#             depth -= text.count("END_IF")
#             if depth == 0:
#                 return ln

#         # 去掉 ELSIF 再判断 IF ... THEN
#         tmp = text.replace("ELSIF", "")
#         if "IF" in tmp and "THEN" in tmp:
#             depth += 1

#     return line_start

def _scan_matching_end_if(line_start: int, code_lines: List[str]) -> int:
    end_ln = _scan_matching_end_generic(line_start, code_lines, _RE_IF_HEAD, _RE_END_IF)
    return end_ln if end_ln is not None else line_start

def _scan_matching_end_for(line_start: int, code_lines: List[str]) -> int:
    end_ln = _scan_matching_end_generic(line_start, code_lines, _RE_FOR_HEAD, _RE_END_FOR)
    return end_ln if end_ln is not None else line_start


def _scan_matching_end_case(line_start: int, code_lines: List[str]) -> int:
    end_ln = _scan_matching_end_generic(line_start, code_lines, _RE_CASE_HEAD, _RE_END_CASE)
    return end_ln if end_ln is not None else line_start


def _scan_matching_end_while(line_start: int, code_lines: List[str]) -> int:
    end_ln = _scan_matching_end_generic(line_start, code_lines, _RE_WHILE_HEAD, _RE_END_WHILE)
    return end_ln if end_ln is not None else line_start


def _scan_matching_end_repeat(line_start: int, code_lines: List[str]) -> int:
    end_ln = _scan_matching_end_generic(line_start, code_lines, _RE_REPEAT_HEAD, _RE_END_REPEAT)
    return end_ln if end_ln is not None else line_start

def _scan_case_start(ln: int, code_lines: List[str]) -> int:
    """
    从 ln(1-based) 向上找最近的 CASE ... OF 头。
    采用简单匹配：遇到 'CASE' 行即认为是 CASE 头。
    """
    for i in range(ln, 0, -1):
        t = _clean_st_line(code_lines[i - 1]).strip()
        if _RE_CASE_HEAD.search(t):
            return i
    return ln

def _is_case_label_line(text: str) -> bool:
    """
    判断是否是 CASE 分支 label 行。
    策略：包含 ':' 且不包含 ':='（赋值）。
    """
    if _RE_ASSIGN.search(text):
        return False
    return _RE_COLON.search(text) is not None

def _scan_elsif_else_lines(if_start: int, if_end: int, code_lines: List[str]) -> Set[int]:
    """
    在 [if_start, if_end]（闭区间，1-based）范围内扫描：
      - 每个 ELSIF 头行 + 其多行条件头（直到 THEN）
      - 每个 ELSE 行
    返回需要补入 slice 的行号集合。
    """
    n = len(code_lines)
    if_start = max(1, if_start)
    if_end = min(n, if_end)

    out: Set[int] = set()

    ln = if_start
    while ln <= if_end:
        raw = code_lines[ln - 1]
        text = _clean_st_line(raw)  # 若你已有 _clean_st_line，换成它
        if not text.strip():
            ln += 1
            continue

        if _RE_ELSIF_HEAD.search(text):
            header_end = _scan_if_header_end(ln, code_lines)
            header_end = min(header_end, if_end)
            for k in range(ln, header_end + 1):
                out.add(k)
            ln = header_end + 1
            continue

        if _RE_ELSE_HEAD.search(text):
            out.add(ln)
            ln += 1
            continue

        ln += 1

    return out

def patch_if_structure(
    sliced_lines: Set[int],
    code_lines: List[str],
    *,
    ensure_end_if: bool = True,
    include_if_header_when_branch_touched: bool = True,
) -> Set[int]:
    """
    在 sliced_lines 基础上做 IF 结构补全（尽量不膨胀 body）：

    规则：
      1) 若命中某个 IF 的任一分支 body（IF/ELSIF/ELSE 分支体任意一行），则补该分支头行；
      2) 若命中的是 ELSIF/ELSE 分支（即非首分支），则（可选）同时补该 IF 的 IF 头行（否则 ELSIF/ELSE 在片段中会成为孤儿）；
      3) 可选：若 IF 任一分支 body 被触及，则确保 END_IF 也被纳入；
      4) 支持多行 IF 头（从 IF 行向下扫描，直到 THEN）。

    注意：本函数不把整个 IF body 拉进来，只补“结构行”，以控制块大小。
    """
    n = len(code_lines)
    base = set(ln for ln in sliced_lines if 1 <= ln <= n)
    if not base:
        return base

    patched = set(base)

    # -------- helpers --------
    def _scan_if_header_end_local(if_line: int) -> int:
        # 复用你已有的 _scan_if_header_end；若不存在就退化为单行
        try:
            return _scan_if_header_end(if_line, code_lines)
        except Exception:
            return if_line

    # -------- stack frame --------
    @dataclass
    class _IfFrame:
        if_line: int
        header_end: int
        end_if_line: Optional[int]
        # 每个分支：head_line, body_start, body_end
        branches: List[Tuple[int, int, int]]
        current_branch_head: int

    stack: List[_IfFrame] = []

    for ln in range(1, n + 1):
        t = _clean_st_line(code_lines[ln - 1]).strip()
        if not t:
            continue

        # IF head
        if _is_if_head(t):
            header_end = _scan_if_header_end_local(ln)
            frame = _IfFrame(
                if_line=ln,
                header_end=header_end,
                end_if_line=None,
                branches=[],
                current_branch_head=ln,
            )
            stack.append(frame)
            continue

        # branch switch
        if _is_elsif(t) or _is_else(t):
            if not stack:
                continue
            frame = stack[-1]
            body_start = frame.current_branch_head + 1
            body_end = ln - 1
            frame.branches.append((frame.current_branch_head, body_start, body_end))
            frame.current_branch_head = ln
            continue

        # END_IF
        if _is_end_if(t):
            if not stack:
                continue
            frame = stack.pop()
            # close last branch
            body_start = frame.current_branch_head + 1
            body_end = ln - 1
            frame.branches.append((frame.current_branch_head, body_start, body_end))
            frame.end_if_line = ln

            # determine which branches are touched
            touched_heads: List[int] = []
            nonfirst_touched = False
            if_touched = False

            for idx, (head, bs, be) in enumerate(frame.branches):
                if bs <= be and any((bs <= x <= be) for x in base):
                    if_touched = True
                    touched_heads.append(head)
                    if idx > 0:
                        nonfirst_touched = True

            # if the slice already contains ELSIF/ELSE head lines directly, treat that as touch
            for idx, (head, _, _) in enumerate(frame.branches):
                if head in base:
                    if_touched = True
                    touched_heads.append(head)
                    if idx > 0:
                        nonfirst_touched = True

            if if_touched:
                # 1) add touched branch heads
                for head in touched_heads:
                    patched.add(head)

                # 2) ensure IF header (IF ... THEN) is present when non-first branch is used
                if include_if_header_when_branch_touched and nonfirst_touched:
                    for hln in range(frame.if_line, min(frame.header_end, n) + 1):
                        patched.add(hln)

                # 3) ensure END_IF
                if ensure_end_if and frame.end_if_line:
                    patched.add(frame.end_if_line)

            continue

    return patched

def patch_case_structure(
    line_numbers: Iterable[int],
    code_lines: List[str],
    ensure_end_case: bool = True,
    include_branch_headers: bool = True,
    include_case_header_when_nested_touched: bool = True,
) -> Set[int]:
    """
    CASE 结构补全（轻量、保语法优先）：

    - 若 slice 命中 CASE 区域内部任意行，则补齐 CASE 头 + END_CASE
    - 若 slice 命中某分支体但缺 label/ELSE 头，则向上补齐最近分支头（label 或 ELSE）
    - 若命中的是分支体且分支头为 label/ELSE（而 CASE 头未命中），可选补齐 CASE 头，
      避免片段中出现“孤儿分支头/孤儿 END_CASE”。

    注意：尽量只补结构行，不扩张整个分支体。
    """
    n = len(code_lines)
    lines = set(ln for ln in line_numbers if 1 <= ln <= n)
    if not lines:
        return lines




def prune_orphan_control_lines(
    selected_lines: Iterable[int],
    code_lines: List[str],
) -> List[int]:
    """
    对“按行号抽取”的片段做一轮防御性修剪，避免出现明显的结构孤儿导致语法崩坏：
      - 删除没有对应 IF 的 ELSIF/ELSE；
      - 删除多余的 ELSE（同一 IF 内超过 1 个）；
      - 删除没有对应 CASE 的 END_CASE（或 CASE 空壳，仅由头/尾构成且没有任何分支/语句命中时）。

    该函数只删除行，不新增行；目标是让输出更“可编译”，宁可略丢信息也不要语法错误。
    """
    n = len(code_lines)
    lines = sorted({ln for ln in selected_lines if 1 <= ln <= n})
    if not lines:
        return []

    # Pass 1: prune IF orphans by simulating on selected lines only
    kept: List[int] = []
    if_stack: List[dict] = []  # each: {'has_else': bool}

    for ln in lines:
        t = _clean_st_line(code_lines[ln - 1]).strip()
        u = t.upper()

        if _is_if_head(t):
            if_stack.append({'has_else': False})
            kept.append(ln)
            continue

        if _is_elsif(t):
            if not if_stack:
                continue
            kept.append(ln)
            continue

        if _is_else(t):
            if not if_stack:
                continue
            if if_stack[-1]['has_else']:
                # duplicate ELSE within same IF fragment
                continue
            if_stack[-1]['has_else'] = True
            kept.append(ln)
            continue

        if _is_end_if(t):
            if not if_stack:
                continue
            if_stack.pop()
            kept.append(ln)
            continue

        kept.append(ln)

    # Pass 2: prune obvious CASE empty shells (CASE + END_CASE but nothing else selected inside)
    lines2 = sorted(set(kept))
    # locate CASE starts within selected set
    case_starts = [ln for ln in lines2 if _is_case_head(_clean_st_line(code_lines[ln - 1]).strip())]
    to_remove = set()
    for cs in case_starts:
        ce = _scan_matching_end_case(cs, code_lines)
        if not ce or ce not in lines2:
            continue
        inner = [x for x in lines2 if cs < x < ce]
        if not inner:
            # empty shell in the slice; remove both
            to_remove.add(cs); to_remove.add(ce)

    return [ln for ln in lines2 if ln not in to_remove]

    case_regions: List[Tuple[int, int]] = []

    # 1) 对每个命中行：向上找 CASE 头，向下找匹配 END_CASE
    for ln in sorted(lines):
        cs = _scan_case_start(ln, code_lines)
        if cs is None or cs < 1 or cs > n:
            continue
        ce = _scan_matching_end_case(cs, code_lines)
        if ce is None or ce < cs:
            continue
        # 只要命中行落在 (cs,ce] 或正好命中 cs/ce，就认为触及该 CASE
        if cs <= ln <= ce:
            case_regions.append((cs, ce))

    # 去重 + 合并重叠区域（避免重复补导致奇怪拼接）
    case_regions = sorted(set(case_regions))
    merged: List[Tuple[int, int]] = []
    for cs, ce in case_regions:
        if not merged or cs > merged[-1][1]:
            merged.append((cs, ce))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ce))
    case_regions = merged

    if not case_regions:
        return lines

    for (cs, ce) in case_regions:
        # 补 CASE 头/尾
        if include_case_header_when_nested_touched:
            lines.add(cs)
        if ensure_end_case:
            lines.add(ce)

        if not include_branch_headers:
            continue

        # 2) 分支头补全：对 region 内每个命中行向上找最近 label/ELSE
        region_hits = [x for x in lines if cs < x < ce]
        for hit in region_hits:
            hit_text = _clean_st_line(code_lines[hit - 1]).strip()
            if _is_case_label_line(hit_text) or _RE_ELSE.search(hit_text):
                continue

            # backtrack to nearest label/ELSE (do not cross cs)
            for up in range(hit - 1, cs, -1):
                up_text = _clean_st_line(code_lines[up - 1]).strip()
                if not up_text:
                    continue
                if _is_case_label_line(up_text) or _RE_ELSE.search(up_text):
                    lines.add(up)

                    # 多行 label 逗号续行（最多补 3 行）
                    k = up - 1
                    steps = 0
                    while k > cs and steps < 3:
                        prev = _clean_st_line(code_lines[k - 1]).strip()
                        if prev.endswith(","):
                            lines.add(k)
                            k -= 1
                            steps += 1
                        else:
                            break
                    break

    return lines

def stmts_to_line_numbers(stmts: List[Stmt], code_lines: List[str]) -> List[int]:
    """
    把语句集合映射为源码行号集合，并按行号排序。
    针对 IfStmt，会额外加入：
        - 多行条件头部所有行（直到 THEN）
        - 匹配的 END_IF 行
    """
    sliced_lines: Set[int] = set()
    n = len(code_lines)

    for st in stmts:
        line_no = getattr(st.loc, "line", None)
        if line_no is None or not (1 <= line_no <= n):
            continue

        if isinstance(st, IfStmt):
            # 1) IF 头部（支持多行条件）
            header_end = _scan_if_header_end(line_no, code_lines)
            for ln in range(line_no, min(header_end, n) + 1):
                sliced_lines.add(ln)

            # 2) 匹配 END_IF 行
            end_if_ln = _scan_matching_end_if(line_no, code_lines)
            if 1 <= end_if_ln <= n:
                sliced_lines.add(end_if_ln)

                # 3) 补齐 IF 区间内的 ELSIF / ELSE 行（含 ELSIF 多行头）
                extra = _scan_elsif_else_lines(line_no, end_if_ln, code_lines)
                sliced_lines.update(extra)
        elif isinstance(st, CaseStmt):
            # CASE 头
            sliced_lines.add(line_no)
            # END_CASE
            end_ln = _scan_matching_end_case(line_no, code_lines)
            if end_ln:
                sliced_lines.add(end_ln)

        elif isinstance(st, ForStmt):
            sliced_lines.add(line_no)
            end_ln = _scan_matching_end_for(line_no, code_lines)
            if end_ln:
                sliced_lines.add(end_ln)

        elif isinstance(st, WhileStmt):
            sliced_lines.add(line_no)
            end_ln = _scan_matching_end_while(line_no, code_lines)
            if end_ln:
                sliced_lines.add(end_ln)

        elif isinstance(st, RepeatStmt):
            sliced_lines.add(line_no)
            end_ln = _scan_matching_end_repeat(line_no, code_lines)
            if end_ln:
                sliced_lines.add(end_ln)
        else:
            # 普通语句，使用自身所在行
            sliced_lines.add(line_no)

    return sorted(sliced_lines)

# -------------------------------------------------
# 从 parent_block 中按行号子集构造子块
# -------------------------------------------------
def _build_block_from_lines(
    parent_block: FunctionalBlock,
    seg_lines: List[int],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> FunctionalBlock:
    """
    给定父功能块 + 一段连续/非连续行号，构造一个新的子功能块：
      - 从父块的 node_ids 中筛出落在这些行里的节点；
      - 基于这些节点重新生成 stmts / line_numbers。
    """
    seg_line_set = set(seg_lines)

    # 1) 找出属于这些行号的 node_ids
    sub_node_ids: Set[int] = set()
    for nid in parent_block.node_ids:
        if 0 <= nid < len(ir2ast_stmt):
            st = ir2ast_stmt[nid]
            if st is None:
                continue
            ln = getattr(st.loc, "line", None)
            if ln is not None and ln in seg_line_set:
                sub_node_ids.add(nid)

    # 2) 由子 node_ids 重新生成 stmts（带控制结构闭包）
    sub_stmts = nodes_to_sorted_ast_stmts(sub_node_ids, ir2ast_stmt, parent_map)

    base_lines = stmts_to_line_numbers(sub_stmts, code_lines)
    fixed_lines = patch_if_structure(base_lines, code_lines, ensure_end_if=True)
    fixed_lines = patch_case_structure(fixed_lines, code_lines, ensure_end_case=True, include_branch_headers=True)
    sub_lines = sorted(fixed_lines)

    return FunctionalBlock(
        criteria=list(parent_block.criteria),
        node_ids=sub_node_ids,
        stmts=sub_stmts,
        line_numbers=sub_lines,
    )

def update_ctrl_depth(text: str, depth: int, *, clamp_negative: bool = True) -> int:
    u = _clean_st_line(text).upper()

    # END_* 先减
    depth -= len(_RE_END_IF.findall(u))
    depth -= len(_RE_END_FOR.findall(u))
    depth -= len(_RE_END_CASE.findall(u))
    depth -= len(_RE_END_WHILE.findall(u))
    depth -= len(_RE_END_REPEAT.findall(u))

    # 头部后加（IF 排除 ELSIF）
    tmp = _RE_ELSIF.sub("", u)
    if _RE_IF_HEAD.search(tmp):
        depth += 1
    if _RE_FOR_HEAD.search(u):
        depth += 1
    if _RE_CASE_HEAD.search(u):
        depth += 1
    if _RE_WHILE_HEAD.search(u):
        depth += 1
    if _RE_REPEAT_HEAD.search(u):
        depth += 1

    if clamp_negative and depth < 0:
        depth = 0
    return depth


# 拆分“过大”的块
def _split_block_by_size(
    block: FunctionalBlock,
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    """
    如果一个功能块太大，就按行号大致切分成若干子块。
    约束：尽量不要在未闭合的 IF/END_IF 等控制结构中间切断。
    """
    lines = sorted(block.line_numbers)
    if len(lines) <= max_lines:
        return [block]

    segments: List[List[int]] = []
    current: List[int] = []
    ctrl_depth = 0  # 控制结构深度（IF/END_IF, FOR/END_FOR, CASE/END_CASE）

    for ln in lines:
        if not current:
            current = [ln]
        else:
            if ln != current[-1] + 1 and len(current) >= min_lines and ctrl_depth == 0:
                segments.append(current)
                current = [ln]
            else:
                current.append(ln)

        # 先更新深度（基于当前行）
        if 1 <= ln <= len(code_lines):
            ctrl_depth = update_ctrl_depth(code_lines[ln - 1], ctrl_depth, clamp_negative=True)

        # 再判断是否强制切
        if len(current) >= max_lines and ctrl_depth == 0:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    # 对太小的段做一次简单合并，避免出现大量极短小块
    merged_segments: List[List[int]] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        if len(seg) < min_lines:
            # 合并到前一个 segment
            merged_segments[-1].extend(seg)
        else:
            merged_segments.append(seg)

    sub_blocks: List[FunctionalBlock] = []
    for seg in merged_segments:
        seg = sorted(set(seg))
        if len(seg) < max(1, min_lines // 2):
            # 仍然太小的段可以选择丢弃
            continue
        b = _build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map)
        sub_blocks.append(b)

    # 如果全被丢弃了，至少保留原 block，避免返回空列表
    return sub_blocks or [block]



def split_blocks_by_stage(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
    min_lines: int = 5,
    stage_var_names: Tuple[str, ...] = ("stage", "Stage"),
) -> List[FunctionalBlock]:
    """
    在 normalize_block_sizes 之前调用：
      - 先按 Stage 切分一个 block，避免 0–4 状态混在一起；
      - 再交给 normalize_block_sizes 做按行数二次切分。

    如果某个 block 中根本没有 stage 相关标记，则原样返回该 block。
    """
    new_blocks: List[FunctionalBlock] = []

    for block in blocks:
        stage_sub_blocks = _split_block_by_stage(
            block,
            ir2ast_stmt=ir2ast_stmt,
            code_lines=code_lines,
            parent_map=parent_map,
            min_lines=min_lines,
            stage_var_names=stage_var_names,
        )
        new_blocks.extend(stage_sub_blocks)

    return new_blocks


def _split_block_by_stage(
    block: FunctionalBlock,
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
    min_lines: int,
    stage_var_names: Tuple[str, ...] = ("stage", "Stage"),
) -> List[FunctionalBlock]:
    """
    针对单个功能块做基于状态机的切分：
      - 利用 'IF stage = N' / 'stage := N' 作为切分点；
      - 在不同 Stage 之间形成多个子段；
      - 保留原有 block 的 stmts / node_ids 子集，通过 _build_block_from_lines 重建。
    """
    lines = sorted(block.line_numbers)
    if not lines:
        return [block]

    # 标记每一行是否是 stage 切分点
    def is_stage_marker(line_text: str) -> bool:
        lt = line_text.strip()
        if not lt:
            return False
        upper = lt.upper()

        # 统一把变量名也转成大写来匹配
        stage_patterns = []
        for name in stage_var_names:
            uname = name.upper()
            # IF stage = N THEN
            stage_patterns.append(rf"\bIF\b.*\b{uname}\b\s*=\s*[\w\.]+")
            # stage := N
            stage_patterns.append(rf"\b{uname}\b\s*:?=\s*[\w\.]+")

        for pat in stage_patterns:
            if re.search(pat, upper):
                return True
        return False

    segments: List[List[int]] = []
    current: List[int] = []
    saw_stage_marker = False

    for ln in lines:
        text = code_lines[ln - 1] if 0 < ln <= len(code_lines) else ""
        if is_stage_marker(text):
            saw_stage_marker = True
            # 如果当前已经积累了一些行，则先把当前段收尾
            if current:
                segments.append(current)
            # 从当前行起开启新段（确保标记行出现在某个 segment 里）
            current = [ln]
        else:
            # 普通行：累积到 current 段中
            if not current:
                current = [ln]
            else:
                current.append(ln)

    if current:
        segments.append(current)

    # 没有识别到任何 Stage 标记，直接返回原 block
    if not saw_stage_marker:
        return [block]

    # 针对 segments 做一个简单的“最小行数”处理：
    #   - 如果某段 < min_lines，则尝试和前一段合并；
    #   - 再不行就在最后兜底保留原 block。
    merged_segments: List[List[int]] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        if len(seg) < min_lines:
            # 合并到前一个 segment
            merged_segments[-1].extend(seg)
        else:
            merged_segments.append(seg)

    sub_blocks: List[FunctionalBlock] = []
    for seg in merged_segments:
        seg = sorted(set(seg))
        if len(seg) < max(1, min_lines // 2):
            # 太小的段，如果你希望继续丢弃也可以，这里默认还是保留一点
            continue
        b = _build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map)
        sub_blocks.append(b)

    return sub_blocks or [block]

def normalize_and_split_blocks(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    """
    推荐的两级规范化：
      1. 基于状态机的语义切分（split_blocks_by_stage）
      2. 基于行数的尺寸规范化（normalize_block_sizes）
    """
    # 1) Stage-based semantic splitting
    stage_blocks = split_blocks_by_stage(
        blocks=blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        parent_map=parent_map,
        min_lines=min_lines,
        stage_var_names=("stage", "Stage"),
    )

    # 2) Size-based splitting
    normalized_blocks = normalize_block_sizes(
        blocks=stage_blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )

    return normalized_blocks

# -------------------------------------------------
# 对所有块做大小规范化
# -------------------------------------------------
def normalize_block_sizes(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    """
    对功能块大小做规范化：
      - 太大的块拆分；
      - 太小的块可以选择合并或丢弃（当前实现只做简单拆分）。
    """
    normalized: List[FunctionalBlock] = []

    for block in blocks:
        line_count = len(block.line_numbers)
        if line_count > max_lines:
            sub_blocks = _split_block_by_size(
                block, ir2ast_stmt, code_lines, min_lines, max_lines, parent_map
            )
            normalized.extend(sub_blocks)
        else:
            normalized.append(block)

    return normalized



# --------（可选）变量收集，后面用于构造小 PROGRAM --------

def collect_vars_in_expr(
    expr: Expr,
    vars_used: Set[str],
    funcs_used: Set[str] | None = None,
) -> None:
    if expr is None:
        return

    # 1) 变量引用
    if isinstance(expr, VarRef):
        vars_used.add(expr.name)

    # 2) 数组访问：递归 base 和 index
    elif isinstance(expr, ArrayAccess):
        collect_vars_in_expr(expr.base, vars_used, funcs_used)
        collect_vars_in_expr(expr.index, vars_used, funcs_used)

    # 3) 结构体字段访问：递归 base
    elif isinstance(expr, FieldAccess):
        collect_vars_in_expr(expr.base, vars_used, funcs_used)

    # 4) 二元运算：递归左右
    elif isinstance(expr, BinOp):
        collect_vars_in_expr(expr.left, vars_used, funcs_used)
        collect_vars_in_expr(expr.right, vars_used, funcs_used)

    # 5) 函数/FB 调用表达式：只递归参数，不把 func 算变量
    elif isinstance(expr, CallExpr):
        # 如需统计函数名，可以写入 funcs_used
        if funcs_used is not None:
            funcs_used.add(expr.func)
        for arg in expr.args:
            collect_vars_in_expr(arg, vars_used, funcs_used)

    # 6) 字面量：忽略
    elif isinstance(expr, Literal):
        return

    else:
        # 以后有新的 Expr 子类，再在这里补分支即可
        return


def collect_vars_in_stmt(stmt: Stmt,
                         vars_used: Set[str],
                         funcs_used: Set[str] | None = None):
    if isinstance(stmt, Assignment):
        collect_vars_in_expr(stmt.target, vars_used, funcs_used)
        collect_vars_in_expr(stmt.value, vars_used, funcs_used)

    elif isinstance(stmt, IfStmt):
        collect_vars_in_expr(stmt.cond, vars_used, funcs_used)
        for s in stmt.then_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        for cond, body in stmt.elif_branches:
            collect_vars_in_expr(cond, vars_used, funcs_used)
            for s in body:
                collect_vars_in_stmt(s, vars_used, funcs_used)
        for s in stmt.else_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)

    elif isinstance(stmt, ForStmt):
        # 循环变量本身也要声明，所以放进 vars_used
        vars_used.add(stmt.var)
        collect_vars_in_expr(stmt.start, vars_used, funcs_used)
        collect_vars_in_expr(stmt.end, vars_used, funcs_used)
        if stmt.step is not None:
            collect_vars_in_expr(stmt.step, vars_used, funcs_used)
        for s in stmt.body:
            collect_vars_in_stmt(s, vars_used, funcs_used)

    elif isinstance(stmt, CallStmt):
        # 这里只遍历参数，不把 stmt.fb_name 加到 vars_used 里
        for arg in stmt.args:
            collect_vars_in_expr(arg, vars_used, funcs_used)



def collect_vars_in_block(stmts: List[Stmt]) -> Set[str]:
    vars_used: Set[str] = set()
    for s in stmts:
        collect_vars_in_stmt(s, vars_used)
    return vars_used

def is_meaningful_block(block: FunctionalBlock, code_lines: list[str]) -> bool:
    lines = [
        code_lines[ln - 1]
        for ln in sorted(block.line_numbers)
        if 1 <= ln <= len(code_lines)
    ]

    depth = 0
    went_negative = False
    for t in lines:
        depth = update_ctrl_depth(t, depth, clamp_negative=False)
        if depth < 0:
            went_negative = True

    balanced = (depth == 0) and (not went_negative)

    # 你原本可能还有其它规则（空块/注释块/只有声明等），这里保持你的其它判断即可
    return balanced


    # 1) 赋值语句数
    assign_count = sum(
        (":=" in t) and not t.upper().startswith("IF ") and not t.upper().startswith("ELSIF ")
        for t in lines
    )

    # 2) 控制结构平衡检查（简易）
    depth = 0
    for t in lines:
        u = t.upper()
        if "END_IF" in u:
            depth -= u.count("END_IF")
        tmp = u.replace("ELSIF", "")
        if "IF" in tmp and "THEN" in tmp:
            depth += 1
    balanced = (depth == 0)

    # 3) 最小行数
    min_len_ok = (len(lines) >= 10)

    return assign_count >= 3 and balanced and min_len_ok


# -------------------------------------------------
# 后处理 1：删除块中的“空 IF”
# -------------------------------------------------

def _remove_empty_ifs_in_block(block: FunctionalBlock, code_lines: List[str]) -> None:
    """
    在单个 FunctionalBlock 内删除“空 IF”结构：
        IF <cond> THEN
        END_IF;
    判定是基于该 block 的 line_numbers：
      - 只要在 IF 和 END_IF 之间，在本 block 中没有任何非空、非注释的行，
        则视为“空 IF”，从 block.line_numbers 中移除 IF 和 END_IF 这两行。

    注意：
      - 不修改 code_lines，只是让这个 block 不再引用这些行号。
      - 如果 IF 内有嵌套 IF 头出现在本 block 中，会被视为“有内容”，不会删除外层 IF。
    """
    if not block.line_numbers:
        return

    ln_set = set(block.line_numbers)
    to_remove: Set[int] = set()
    n = len(code_lines)

    for ln in sorted(block.line_numbers):
        if not (1 <= ln <= n):
            continue

        raw = code_lines[ln - 1]
        stripped = _strip_st_comments(raw).strip()
        upper = stripped.upper()

        # 只处理 IF 头（排除 ELSIF）
        if not _is_if_start(upper):
            continue

        end_ln = _scan_matching_end_if(ln, code_lines)
        if not (1 <= end_ln <= n):
            continue

        # 在 [ln+1, end_ln-1] 范围内、当前 block 的行号中，是否有“实质内容”
        has_content = False
        for inner_ln in ln_set:
            if ln < inner_ln < end_ln:
                inner_raw = code_lines[inner_ln - 1]
                inner_text = _strip_st_comments(inner_raw).strip()
                if inner_text != "":
                    # 任何非空文本都算内容（包括嵌套 IF/ELSIF/ELSE）
                    has_content = True
                    break

        if not has_content:
            # 视为“空 IF”，移除 IF 头和匹配的 END_IF 行（如果在本 block 中）
            to_remove.add(ln)
            if end_ln in ln_set:
                to_remove.add(end_ln)

    if to_remove:
        block.line_numbers = sorted(ln for ln in block.line_numbers if ln not in to_remove)


def remove_empty_ifs_in_blocks(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
) -> List[FunctionalBlock]:
    """
    对所有功能块执行 _remove_empty_ifs_in_block。
    只修改每个 block 的 line_numbers，不改变 code_lines 本身。
    """
    for b in blocks:
        _remove_empty_ifs_in_block(b, code_lines)
    return blocks


def _find_uncovered_line_ranges(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
) -> List[Tuple[int, int]]:
    """
    找出“未被任何块覆盖”的源码行号区间（闭区间 [start, end]）列表。
    只考虑非空行、非纯注释行。
    """
    used: Set[int] = set()
    for b in blocks:
        used.update(b.line_numbers)

    # 只考虑有实际代码的行
    def is_code_line(idx: int) -> bool:
        line = code_lines[idx - 1]
        stripped = line.strip()
        if not stripped:
            return False
        # 简单处理 ST 注释：(* ... *) 或 // ...
        if stripped.startswith("(*") and stripped.endswith("*)"):
            return False
        if stripped.startswith("//"):
            return False
        return True

    uncovered: List[int] = []
    for i in range(1, len(code_lines) + 1):
        if not is_code_line(i):
            continue
        if i not in used:
            uncovered.append(i)

    if not uncovered:
        return []

    # 把离散行聚集成连续区间
    ranges: List[Tuple[int, int]] = []
    start = uncovered[0]
    prev = start
    for ln in uncovered[1:]:
        if ln == prev + 1:
            prev = ln
        else:
            ranges.append((start, prev))
            start = ln
            prev = ln
    ranges.append((start, prev))

    return ranges


def _attach_uncovered_ranges_to_blocks(
    blocks: List[FunctionalBlock],
    uncovered_ranges: List[Tuple[int, int]],
    code_lines: List[str],
    min_range_len: int = 2,
) -> List[FunctionalBlock]:
    """
    把未覆盖的代码区间“挂靠”到最近的功能块上。

    策略：
      - 对每个未覆盖区间 [s, e]：
        * 若长度 < min_range_len，可选地直接丢弃（太小的片段对微调价值有限）
        * 否则：
            - 找到一个与它“最近”的块（按行号距离衡量）
            - 把 [s, e] 范围内的行号加入该块的 line_numbers
    """
    if not uncovered_ranges or not blocks:
        return blocks

    # 为了不破坏原对象，这里浅拷贝一份列表即可
    new_blocks = list(blocks)

    for (s, e) in uncovered_ranges:
        if e < s:
            continue
        length = e - s + 1
        if length < min_range_len:
            # 太短的未覆盖片段可以忽略，或者你也可以设为 1 保证尽量全覆盖
            continue

        # 计算每个块与此区间的“距离”
        best_block = None
        best_dist = None

        for b in new_blocks:
            if not b.line_numbers:
                continue
            # 该块的最小/最大行号
            b_min = min(b.line_numbers)
            b_max = max(b.line_numbers)
            # 如果区间和块有交叠，则距离视为 0
            if not (e < b_min or s > b_max):
                dist = 0
            else:
                # 否则按最近端点之间的绝对距离
                if e < b_min:
                    dist = b_min - e
                else:
                    dist = s - b_max

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_block = b

        if best_block is None:
            continue

        # 把这个区间的行号加入 best_block
        extra_lines = set(range(s, e + 1))
        best_block.line_numbers = sorted(set(best_block.line_numbers) | extra_lines)

    return new_blocks

# -------------------------------------------------
# 后处理 2：对完全重复的块去重（按源码文本）
# -------------------------------------------------

def dedup_blocks_by_code(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
) -> List[FunctionalBlock]:
    """
    按“切出来的源码文本是否完全相同”对功能块去重：
      - 对每个 block，将其 line_numbers 对应的源码行（去掉行尾空白）拼成字符串作为 key；
      - key 完全相同的 block 只保留出现的第一份，其余丢弃。

    这可以去掉类似 BLOCK 1 / BLOCK 5 这种 PROGRAM 主体完全相同、
    只是名字不同的重复块。
    """
    seen_keys: Set[str] = set()
    unique_blocks: List[FunctionalBlock] = []

    n = len(code_lines)

    for b in blocks:
        if not b.line_numbers:
            continue

        # 用本 block 的源码片段作为去重 key
        body_lines: List[str] = []
        for ln in sorted(set(b.line_numbers)):
            if 1 <= ln <= n:
                body_lines.append(code_lines[ln - 1].rstrip())

        key = "\n".join(body_lines)

        if key in seen_keys:
            # 完全重复，丢弃
            continue

        seen_keys.add(key)
        unique_blocks.append(b)

    return unique_blocks


# -----------------------
# 高层封装：一键“功能块划分”
# -----------------------

def extract_functional_blocks(
    prog_pdg,
    criteria: List[SlicingCriterion],
    ir2ast_stmt: List,
    code_lines: List[str],
    overlap_threshold: float = 0.75,
    min_lines: int = 12,
    max_lines: int = 150,
    min_lines_stage: int = 8,
) -> List[FunctionalBlock]:
    """
    基于多种切片准则挖掘功能块。

    相比旧版本的主要变化：
      1. 按 criterion.kind 分组后分别调用 cluster_slices，避免不同类型的切片彼此“吃掉”；
      2. 默认 overlap_threshold 调高到 0.75，min_lines 降到 12，偏向生成更多的块；
      3. split_blocks_by_stage 的 stage_var_names 更通用：包含 stage/state/step/phase/mode。

    其它 pipeline（补 IF 结构、按 stage 切分、尺寸归一、去重）保持不变。
    """

    # 0) 基于 ir2ast_stmt 构造 parent_map（只做一次）
    parent_map = build_parent_map_from_ir2ast(ir2ast_stmt)

    # 1) 先对每个准则做一次 PDG 切片
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        nodes = compute_slice_nodes(prog_pdg, crit.node_id)
        if not nodes:
            continue
        all_slices.append((crit, nodes))

    if not all_slices:
        return []

    # 1.1 按 kind 分组切片，避免过度混合
    grouped: Dict[str, List[Tuple[SlicingCriterion, Set[int]]]] = defaultdict(list)
    for crit, nodes in all_slices:
        kind = crit.kind or "unknown"
        grouped[kind].append((crit, nodes))

    # 1.2 对每一类 kind 分别调用 cluster_slices，再合并结果
    clusters: List[Dict[str, Any]] = []
    for kind, slices in grouped.items():
        # 你现有的 cluster_slices 接口：cluster_slices(all_slices, overlap_threshold)
        kind_clusters = cluster_slices(slices, overlap_threshold=overlap_threshold)
        # 可以在 cluster 里标记一下 kind，便于 debug
        for c in kind_clusters:
            c.setdefault("kind", kind)
        clusters.extend(kind_clusters)

    # 2) 根据每个 cluster 构建 FunctionalBlock（先不按 stage 切割）
    blocks: List[FunctionalBlock] = []
    for cluster in clusters:
        node_ids: Set[int] = cluster["nodes"]
        crits: List[SlicingCriterion] = cluster["criteria"]

        stmts = nodes_to_sorted_ast_stmts(node_ids, ir2ast_stmt, parent_map)

        # (2.1) 基本行号
        base_lines = stmts_to_line_numbers(stmts, code_lines)

        # (2.2) 补 IF 结构（补 ELSIF/ELSE/END_IF）
        fixed_lines = patch_if_structure(base_lines, code_lines, ensure_end_if=True)
        fixed_lines = patch_case_structure(fixed_lines, code_lines, ensure_end_case=True, include_branch_headers=True)
        # (2.3) 防御性修剪：删除明显的结构孤儿，优先保证语法可编译
        fixed_lines = prune_orphan_control_lines(fixed_lines, code_lines)

        line_numbers = sorted(fixed_lines)

        block = FunctionalBlock()
        block.criteria = crits
        block.node_ids = set(node_ids)
        block.stmts = stmts
        block.line_numbers = line_numbers
        blocks.append(block)

    # 3) 按 stage/state/step/phase/mode 划分子块
    blocks = split_blocks_by_stage(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        parent_map=parent_map,
        min_lines=min_lines_stage,
        stage_var_names=("stage", "Stage", "state", "State", "step", "Step", "phase", "Phase", "mode", "Mode"),
    )

    # 4) 尺寸归一：控制块的长度范围在 [min_lines, max_lines]
    blocks = normalize_block_sizes(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )

        # 先做一次“是否有意义”的过滤
    blocks = [b for b in blocks if is_meaningful_block(b, code_lines)]

    # 后处理 1：删除块中的空 IF（只改 block.line_numbers，不改 code_lines）
    blocks = remove_empty_ifs_in_blocks(blocks, code_lines)

    # 后处理 2：根据源码内容对完全重复的块去重
    blocks = dedup_blocks_by_code(blocks, code_lines)

    # ========= 新增步骤：按行修补覆盖率 =========
    # 1) 找出未覆盖的源码区间
    uncovered_ranges = _find_uncovered_line_ranges(blocks, code_lines)
    # 2) 挂靠到最近的功能块上（可调 min_range_len）
    blocks = _attach_uncovered_ranges_to_blocks(
        blocks,
        uncovered_ranges,
        code_lines,
        min_range_len=2,  # 可以先用 2 或 3，避免一行一行的噪声
    )

    # 修补后再做一次去重（以防某些块变得完全一样）
    blocks = dedup_blocks_by_code(blocks, code_lines)

    return blocks


