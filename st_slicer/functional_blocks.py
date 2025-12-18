# st_slicer/functional_blocks.py

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional, Iterable, Union

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
    fold_half_empty_ifs_in_block,
)

# =========================================================
# Region slicing: 为兼容你原来的 slice_strategy="region"
# 这里直接放一个本地实现（等价于你 blocks1/slice_ops.py 里的版本）
# =========================================================
def compute_region_nodes(
    crit_node_id: int,
    pdg,
) -> Set[int]:
    """
    区域切片：
    - 找到 crit_node_id 对应的 AST 节点
    - 收集该 AST 子树中所有语句
    - 映射回 PDG node_id 集合
    依赖 pdg.node_to_ast / pdg.ast_to_nodes（若缺失则退化为单点）
    """
    if not hasattr(pdg, "node_to_ast"):
        return {crit_node_id}

    root_ast = pdg.node_to_ast.get(crit_node_id)
    if root_ast is None:
        return {crit_node_id}

    region_stmts: Set[Any] = set()

    def walk(stmt: Any) -> None:
        if stmt is None or stmt in region_stmts:
            return
        region_stmts.add(stmt)

        for attr in ("then_body", "else_body", "body"):
            if hasattr(stmt, attr):
                for s in getattr(stmt, attr) or []:
                    walk(s)

        if hasattr(stmt, "elif_branches"):
            for _, body in getattr(stmt, "elif_branches") or []:
                for s in body or []:
                    walk(s)

        if hasattr(stmt, "entries"):
            for e in getattr(stmt, "entries") or []:
                for s in getattr(e, "body", []) or []:
                    walk(s)

    walk(root_ast)

    ast2nodes = getattr(pdg, "ast_to_nodes", None)
    if not ast2nodes:
        return {crit_node_id}

    nodes: Set[int] = set()
    for s in region_stmts:
        nodes.update(ast2nodes.get(s, []))

    return nodes or {crit_node_id}


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
    已切换到 blocks/ (core/structure/pipeline/render) 的重构版本。
    """

    parent_map = build_parent_map_from_ir2ast(ir2ast_stmt)

    # 1) 每个准则做后向切片（或 region）
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        extra = crit.extra or {}
        strategy = extra.get("slice_strategy", "backward")

        if strategy == "region":
            nodes = compute_region_nodes(crit.node_id, prog_pdg)
        else:
            start_nodes = extra.get("start_nodes") or [crit.node_id]
            nodes = compute_slice_nodes(prog_pdg, start_nodes)

        if nodes:
            all_slices.append((crit, nodes))

    if not all_slices:
        return []

    # 2) 按 kind 分组聚类，避免不同类型互相吞并
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

    # 3) cluster -> FunctionalBlock
    blocks: List[FunctionalBlock] = []
    for cluster in clusters:
        node_ids: Set[int] = cluster["nodes"]
        crits: List[SlicingCriterion] = cluster["criteria"]

        stmts = nodes_to_sorted_ast_stmts(node_ids, ir2ast_stmt, parent_map)

        base_lines = stmts_to_line_numbers(stmts, code_lines)
        fixed = patch_if_structure(base_lines, code_lines, ensure_end_if=True)
        fixed = patch_case_structure(
            fixed,
            code_lines,
            ensure_end_case=True,
            include_branch_headers=True,
        )

        block = FunctionalBlock(
            criteria=crits,
            node_ids=set(node_ids),
            stmts=stmts,
            line_numbers=sorted(fixed),
        )
        blocks.append(block)

    # 4) Stage 切分
    blocks = split_blocks_by_stage(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        parent_map=parent_map,
        min_lines=min_lines_stage,
        stage_var_names=("stage", "Stage", "state", "State"),
    )

    # 5) 尺寸归一
    blocks = normalize_block_sizes(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )

    # 6) 清理空壳结构
    blocks = remove_empty_ifs_in_blocks(blocks, code_lines)
    blocks = remove_empty_loops_in_blocks(blocks, code_lines)
    blocks = remove_empty_cases_in_blocks(blocks, code_lines)

    # 6.5) 折叠“半空 IF 分支”
    for b in blocks:
        fold_half_empty_ifs_in_block(b, code_lines)

    # 7) 过滤无意义块
    blocks = [b for b in blocks if is_meaningful_block(b, code_lines, min_len=min_lines)]

    # 8) 去重
    blocks = dedup_blocks_by_code(blocks, code_lines)

    return blocks
