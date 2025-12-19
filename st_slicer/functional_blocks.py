# st_slicer/functional_blocks.py

from __future__ import annotations

from collections import defaultdict, Counter
from typing import Any, Dict, List, Set, Tuple, Optional,  Iterable

from .policy import PolicyResolver, PolicyModelContext
from .blocks.core import SlicingCriterion, FunctionalBlock

from .blocks.pipeline import (
    compute_slice_nodes,
    cluster_slices,
    nodes_to_sorted_ast_stmts,
    build_parent_map_from_ir2ast,
    stmts_to_line_numbers,
    split_blocks_by_stage,
    normalize_block_sizes,
    remove_empty_ifs_in_blocks,
    remove_empty_loops_in_blocks,
    remove_empty_cases_in_blocks,
    is_meaningful_block,
    dedup_blocks_by_code,
)

from .blocks.structure import (
    patch_if_structure,
    patch_case_structure,
    patch_loop_structures,
    fold_half_empty_ifs_in_block,
)

from .debug_stats import SliceDiag, diag_enabled


# =========================================================
# Region slicing: 为兼容 slice_strategy="region"
# =========================================================
def _invert_pred_map(pred: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    """pred_map: node -> predecessors. Return succ_map: node -> successors."""
    succ: Dict[int, Set[int]] = {}
    for n, ps in pred.items():
        for p in ps:
            succ.setdefault(p, set()).add(n)
    return succ

def _pick_control_root(seed: int,
                       ctrl_pred: Dict[int, Set[int]],
                       ctrl_succ: Dict[int, Set[int]],
                       *,
                       max_up_steps: int = 32) -> int:
    """
    Choose a control predicate root for a seed node.
    Strategy:
      - If seed itself controls others (has ctrl successors), use it.
      - Otherwise walk upward via ctrl_pred and pick the first node that has ctrl successors.
      - Fallback to seed if none found.
    """
    if seed in ctrl_succ and ctrl_succ[seed]:
        return seed

    visited = set()
    frontier = [seed]
    steps = 0

    while frontier and steps < max_up_steps:
        steps += 1
        nxt: List[int] = []
        for x in frontier:
            if x in visited:
                continue
            visited.add(x)
            preds = ctrl_pred.get(x) or set()
            for p in preds:
                if p in ctrl_succ and ctrl_succ[p]:
                    return p
                nxt.append(p)
        frontier = nxt

    return seed

def compute_region_nodes(seed_node_id: int,
                         pdg,
                         ir2ast_stmt=None,
                         *,
                         max_nodes: int = 5000) -> Set[int]:
    """
    Compute 'control region' nodes for a given seed.
    Definition used here:
      - Find the nearest control predicate root (a node that controls others via PDG control edges)
      - Collect all nodes reachable from that root following ONLY control-dependence edges (forward)
    This turns your current region_size==1 degeneration into meaningful regions.
    """
    # pdg is expected to be ProgramDependenceGraph (build_program_dependence_graph output)
    ctrl_pred = getattr(pdg, "ctrl_pred", None)
    if not isinstance(ctrl_pred, dict):
        return {seed_node_id}

    # build ctrl_succ from ctrl_pred (robust even if pdg doesn't store succ)
    ctrl_succ = getattr(pdg, "ctrl_succ", None)
    if not isinstance(ctrl_succ, dict):
        ctrl_succ = _invert_pred_map(ctrl_pred)

    root = _pick_control_root(seed_node_id, ctrl_pred, ctrl_succ)

    # BFS forward on control edges
    region: Set[int] = set()
    q = [root]
    while q and len(region) < max_nodes:
        cur = q.pop()
        if cur in region:
            continue
        region.add(cur)
        for nx in (ctrl_succ.get(cur) or ()):
            if nx not in region:
                q.append(nx)

    # Always include original seed too (even if it differs from root)
    region.add(seed_node_id)

    return region

# =========================================================
# Critical fix:
# PDG node_id -> AST stmt mapping fallback:
#   ir2ast_stmt[nid] may be None / misaligned with pdg node ids.
#   We fallback to pdg.node_to_ast.get(nid) to avoid losing coverage.
# =========================================================
def _nodes_to_sorted_ast_stmts_safe(
    node_ids: Set[int],
    ir2ast_stmt: List,
    parent_map,
    pdg,
):
    """
    Safe mapping:
      1) try ir2ast_stmt[nid]
      2) fallback to pdg.node_to_ast[nid]
      3) then run the existing parent-closure logic via nodes_to_sorted_ast_stmts
         by injecting mapped stmts back into a pseudo index.
    """
    stmt_set: Set[Any] = set()

    # Map nid -> stmt
    for nid in node_ids:
        st = None
        if 0 <= nid < len(ir2ast_stmt):
            st = ir2ast_stmt[nid]
        if st is None and hasattr(pdg, "node_to_ast"):
            st = getattr(pdg, "node_to_ast").get(nid)
        if st is not None:
            stmt_set.add(st)

    if not stmt_set:
        return []

    closed: Set[Any] = set(stmt_set)
    work = list(stmt_set)
    while work:
        st = work.pop()
        p = parent_map.get(st)
        while p is not None:
            # Match the same set as pipeline.close_with_control_structures
            cls = type(p).__name__
            if cls in ("IfStmt", "ForStmt", "CaseStmt", "WhileStmt", "RepeatStmt") and p not in closed:
                closed.add(p)
                work.append(p)
            p = parent_map.get(p)

    return sorted(
        closed,
        key=lambda s: (getattr(getattr(s, "loc", None), "line", 0), getattr(getattr(s, "loc", None), "column", 0)),
    )


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
    基于多种切片准则挖掘功能块（主 pipeline）。
    """
    diag = SliceDiag(total_lines=len(code_lines)) if diag_enabled() else None

    parent_map = build_parent_map_from_ir2ast(ir2ast_stmt)

    resolver = PolicyResolver(
        config=None,   # 可接 load_policy_config("policy.json")
        model=None,    # 可接可学习策略模型
    )
    ctx = PolicyModelContext()

    # ---------- diag: criteria stats ----------
    if diag is not None:
        diag.n_criteria = len(criteria)
        diag.n_criteria_by_kind = dict(Counter([(c.kind or "unknown") for c in criteria]))
        diag.n_set_criteria = sum(1 for c in criteria if (c.kind or "").endswith("_set"))

        seed_total = 0
        seed_by_kind = Counter()
        for c in criteria:
            starts = (c.extra or {}).get("start_nodes") or [c.node_id]
            seed_total += len(starts)
            seed_by_kind[c.kind or "unknown"] += len(starts)
        diag.seed_nodes_total = seed_total
        diag.seed_nodes_by_kind = dict(seed_by_kind)

        diag.n_slices = 0
        diag.n_empty_slices = 0
        diag.slice_node_sizes = []
        diag.cluster_node_sizes = []
        diag.extra = getattr(diag, "extra", {}) or {}

    # ---------------------------------------------------------
    # control_region_set criteria dedup by REGION Jaccard (SHORT)
    # ---------------------------------------------------------
    ctrl_pred = getattr(prog_pdg, "ctrl_pred", None)
    if isinstance(ctrl_pred, dict):
        ctrl_succ = getattr(prog_pdg, "ctrl_succ", None)
        if not isinstance(ctrl_succ, dict):
            ctrl_succ = _invert_pred_map(ctrl_pred)

        # 调这个阈值：0.90~0.95 通常比较稳；越小删得越狠
        REGION_JACCARD_TH = 0.70

        def _jaccard_nodes(a: Set[int], b: Set[int]) -> float:
            if not a and not b:
                return 1.0
            inter = len(a & b)
            if inter == 0:
                return 0.0
            uni = len(a | b)
            return inter / uni if uni else 0.0

        before = sum(1 for c in criteria if (c.kind or "") == "control_region_set")

        kept: List[SlicingCriterion] = []
        kept_regions: List[Set[int]] = []

        for c in criteria:
            if (c.kind or "") != "control_region_set":
                kept.append(c)
                continue

            # 1) seed -> root（仍然用你现有的 root 定义）
            root = _pick_control_root(c.node_id, ctrl_pred, ctrl_succ)

            # 2) root -> region（用你现有的 region 定义）
            region = compute_region_nodes(root, prog_pdg, ir2ast_stmt=ir2ast_stmt)

            # 3) 与已保留 region 做 Jaccard 去重
            redundant = False
            for r0 in kept_regions:
                if _jaccard_nodes(region, r0) >= REGION_JACCARD_TH:
                    redundant = True
                    break
            if redundant:
                continue

            # 4) 保留：node_id 归一到 root（稳定）
            kept.append(SlicingCriterion(node_id=root, kind=c.kind, extra=c.extra))
            kept_regions.append(region)

        criteria = kept

        after = sum(1 for c in criteria if (c.kind or "") == "control_region_set")
        print(f"[region-dedup] control_region_set: {before} -> {after} (th={REGION_JACCARD_TH})")

    # =========================================================
    # 1) 每个准则做后向切片（或 region）
    # =========================================================
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        extra = crit.extra or {}
        policy = resolver.resolve(crit.kind or "unknown", extra, ctx=ctx)

        start_nodes = extra.get("start_nodes") or [crit.node_id]

        if policy.slice.slice_strategy == "region":
            # 支持 start_nodes 多 seed：union 多个 region，避免 control_region_set 退化为单点
            nodes = set()
            for nid in (start_nodes or [crit.node_id]):
                nodes |= compute_region_nodes(nid, prog_pdg, ir2ast_stmt=ir2ast_stmt)

            if diag is not None:
                diag.extra.setdefault("region_sizes", []).append(len(nodes))

        else:
            nodes = compute_slice_nodes(
                prog_pdg,
                start_nodes,
                var_sensitive=bool(policy.slice.var_sensitive) if policy.slice.var_sensitive is not None else False,
                seed_vars=policy.slice.seed_vars,
                use_data=bool(policy.slice.use_data) if policy.slice.use_data is not None else True,
                use_control=bool(policy.slice.use_control) if policy.slice.use_control is not None else True,
            )

        if diag is not None:
            diag.n_slices += 1
            diag.slice_node_sizes.append(len(nodes))
            if not nodes:
                diag.n_empty_slices += 1

        if nodes:
            all_slices.append((crit, nodes))

    if not all_slices:
        if diag is not None:
            print(diag.report())
        return []

    # =========================================================
    # L1) cov after slice (lines union, pre-block)
    #     Use SAFE mapping to avoid losing lines due to ir2ast mismatch.
    # =========================================================
    if diag is not None:
        cov_lines = set()
        for _crit, node_ids in all_slices:
            stmts_tmp = _nodes_to_sorted_ast_stmts_safe(node_ids, ir2ast_stmt, parent_map, prog_pdg)
            cov_lines |= set(stmts_to_line_numbers(stmts_tmp, code_lines))
        diag.cov_after_slice_lines = cov_lines

        # Optional: quick mapping loss probe (pdg.node_to_ast loc.line vs safe mapping)
        cov_pdg_loc = set()
        if hasattr(prog_pdg, "node_to_ast"):
            for _crit, node_ids in all_slices:
                for nid in node_ids:
                    ast = prog_pdg.node_to_ast.get(nid)
                    loc = getattr(ast, "loc", None)
                    ln = getattr(loc, "line", None)
                    if isinstance(ln, int) and 1 <= ln <= len(code_lines):
                        cov_pdg_loc.add(ln)
        diag.extra["cov_pdg_loc_lines"] = len(cov_pdg_loc)
        diag.extra["lost_in_mapping_pdg_minus_safe"] = len(cov_pdg_loc - cov_lines)

    # =========================================================
    # 2) 按 kind 分组聚类，避免不同类型互相吞并
    # =========================================================
    grouped: Dict[str, List[Tuple[SlicingCriterion, Set[int]]]] = defaultdict(list)
    for crit, nodes in all_slices:
        kind_key = (crit.kind or "unknown")
        if kind_key.endswith("_set"):
            kind_key = kind_key[:-4]
        grouped[kind_key].append((crit, nodes))

    clusters: List[Dict[str, Any]] = []
    for kind, slices in grouped.items():
        kind_clusters = cluster_slices(slices, overlap_threshold=overlap_threshold)
        for c in kind_clusters:
            c.setdefault("kind", kind)
        clusters.extend(kind_clusters)

    if diag is not None:
        diag.n_groups = len(grouped)
        diag.n_clusters = len(clusters)
        diag.cluster_node_sizes = [len(c.get("nodes", set()) or set()) for c in clusters]

    # =========================================================
    # 3) cluster -> FunctionalBlock
    # =========================================================
    blocks: List[FunctionalBlock] = []
    for cluster in clusters:
        node_ids: Set[int] = cluster.get("nodes", set()) or set()
        crits: List[SlicingCriterion] = cluster.get("criteria", []) or []

        # SAFE stmt mapping
        stmts = _nodes_to_sorted_ast_stmts_safe(node_ids, ir2ast_stmt, parent_map, prog_pdg)

        base_lines = stmts_to_line_numbers(stmts, code_lines)
        fixed = list(base_lines)

        # Note: do NOT use a stale "policy" from previous criterion here.
        # We apply structure patching deterministically for block compilability/readability.
        fixed = list(patch_if_structure(fixed, code_lines, ensure_end_if=True))
        fixed = list(patch_case_structure(fixed, code_lines, ensure_end_case=True, include_branch_headers=True))
        fixed = list(patch_loop_structures(fixed, code_lines, include_header_span=True, include_until_span=True))

        block = FunctionalBlock(
            criteria=crits,
            node_ids=set(node_ids),
            stmts=stmts,
            line_numbers=sorted(set(fixed)),
        )
        blocks.append(block)

    if diag is not None:
        diag.n_blocks_raw = len(blocks)
        diag.cov_after_blocks_raw = set()
        for b in blocks:
            diag.cov_after_blocks_raw |= set(b.line_numbers)

    # =========================================================
    # 4) Stage 切分
    # =========================================================
    blocks = split_blocks_by_stage(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        parent_map=parent_map,
        min_lines=min_lines_stage,
        stage_var_names=("stage", "Stage", "state", "State"),
    )

    if diag is not None:
        diag.n_blocks_stage = len(blocks)
        diag.cov_after_stage = set()
        for b in blocks:
            diag.cov_after_stage |= set(b.line_numbers)

    # =========================================================
    # 5) 尺寸归一
    # =========================================================
    blocks = normalize_block_sizes(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )

    if diag is not None:
        diag.n_blocks_size = len(blocks)
        diag.cov_after_size = set()
        for b in blocks:
            diag.cov_after_size |= set(b.line_numbers)

    # =========================================================
    # 6) 清理空壳结构 + 6.5 折叠半空 IF
    # =========================================================
    blocks = remove_empty_ifs_in_blocks(blocks, code_lines)
    blocks = remove_empty_loops_in_blocks(blocks, code_lines)
    blocks = remove_empty_cases_in_blocks(blocks, code_lines)

    for b in blocks:
        fold_half_empty_ifs_in_block(b, code_lines)

    if diag is not None:
        diag.n_blocks_cleanup = len(blocks)
        diag.cov_after_cleanup = set()
        for b in blocks:
            diag.cov_after_cleanup |= set(b.line_numbers)

    # =========================================================
    # 7) 过滤无意义块
    # =========================================================
    blocks = [b for b in blocks if is_meaningful_block(b, code_lines, min_len=min_lines)]

    if diag is not None:
        diag.n_blocks_meaningful = len(blocks)
        diag.cov_after_meaningful = set()
        for b in blocks:
            diag.cov_after_meaningful |= set(b.line_numbers)

    # 8) 去重 + overlap 去冗余（全量生成后收敛）
    blocks = dedup_blocks_by_code(blocks, code_lines, overlap_jaccard=0.85, prefer_larger=True)

    if diag is not None:
        diag.n_blocks_dedup = len(blocks)
        diag.cov_after_dedup = set()
        for b in blocks:
            diag.cov_after_dedup |= set(b.line_numbers)

        print(diag.report())
        # extra debug summary (optional)
        if isinstance(diag.extra, dict) and diag.extra:
            rs = diag.extra.get("region_sizes", [])
            if rs:
                rs2 = sorted(rs)
                print(f"[extra] region_sizes: n={len(rs2)} min={rs2[0]} p50={rs2[len(rs2)//2]} max={rs2[-1]}")
            print(f"[extra] cov_pdg_loc_lines={diag.extra.get('cov_pdg_loc_lines')}, "
                  f"lost_in_mapping_pdg_minus_safe={diag.extra.get('lost_in_mapping_pdg_minus_safe')}")

    return blocks
