# st_slicer/functional_blocks.py

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from .criteria import SlicingCriterion
from .blocks.types import FunctionalBlock
from .blocks.slice_ops import (
    compute_slice_nodes,
    cluster_slices,
    nodes_to_sorted_ast_stmts,
    build_parent_map_from_ir2ast,
)
from .blocks.line_map import stmts_to_line_numbers
from .blocks.structure_if_case_loop import patch_if_structure, patch_case_structure
from .blocks.splitters import split_blocks_by_stage, normalize_block_sizes
from .blocks.postprocess import (
    remove_empty_ifs_in_blocks,
    remove_empty_loops_in_blocks,
    remove_empty_cases_in_blocks,
    is_meaningful_block,
    dedup_blocks_by_code,
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
    基于多种切片准则挖掘功能块（拆分后的主 pipeline）。
    """

    parent_map = build_parent_map_from_ir2ast(ir2ast_stmt)

    # 1) 每个准则做后向切片
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        nodes = compute_slice_nodes(prog_pdg, crit.node_id)
        if nodes:
            all_slices.append((crit, nodes))

    if not all_slices:
        return []

    # 2) 按 kind 分组聚类，避免不同类型互相吞并
    grouped: Dict[str, List[Tuple[SlicingCriterion, Set[int]]]] = defaultdict(list)
    for crit, nodes in all_slices:
        grouped[crit.kind or "unknown"].append((crit, nodes))

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
        fixed = patch_case_structure(fixed, code_lines, ensure_end_case=True, include_branch_headers=True)

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

    # 7) 过滤无意义块
    blocks = [b for b in blocks if is_meaningful_block(b, code_lines)]

    # 8) 去重
    blocks = dedup_blocks_by_code(blocks, code_lines)

    return blocks
