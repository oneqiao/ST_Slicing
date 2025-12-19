from __future__ import annotations

from pathlib import Path
from typing import List, Set, Dict, Any, Tuple, Optional
from collections import Counter
import argparse
import json
import copy

from st_slicer.parser.parse_st import parse_st_code
from st_slicer.ast.builder import ASTBuilder
from st_slicer.ir.ir_builder import IRBuilder
from st_slicer.cfg.cfg_builder import CFGBuilder
from st_slicer.dataflow.def_use import DefUseAnalyzer, DefUseResult
from st_slicer.pdg.pdg_builder import PDGBuilder, build_program_dependence_graph, ProgramDependenceGraph
from st_slicer.criteria import mine_slicing_criteria, CriterionConfig
from st_slicer.sema.builder import build_symbol_table

from st_slicer.functional_blocks import extract_functional_blocks
from st_slicer.block_context import build_completed_block

from st_slicer.blocks.pipeline import (
    compute_slice_nodes,
    nodes_to_sorted_ast_stmts,
    build_parent_map_from_ir2ast,
    stmts_to_line_numbers,
)
from st_slicer.blocks.core import SlicingCriterion


# -------------------------
# Coverage utils
# -------------------------
def _is_code_line(code_lines: List[str], i: int) -> bool:
    if not (1 <= i <= len(code_lines)):
        return False
    line = code_lines[i - 1]
    stripped = line.strip()
    if not stripped:
        return False
    # rough comment-only filter
    if stripped.startswith("(*") and stripped.endswith("*)"):
        return False
    if stripped.startswith("//"):
        return False
    return True


def report_coverage(tag: str, blocks_or_lines, code_lines: List[str]) -> Tuple[int, int, float]:
    used: Set[int] = set()

    if isinstance(blocks_or_lines, list) and blocks_or_lines and hasattr(blocks_or_lines[0], "line_numbers"):
        for b in blocks_or_lines:
            used.update(getattr(b, "line_numbers", []) or [])
    else:
        used.update(blocks_or_lines or [])

    total_code_lines = sum(1 for i in range(1, len(code_lines) + 1) if _is_code_line(code_lines, i))
    covered_code_lines = sum(1 for i in used if 1 <= i <= len(code_lines) and _is_code_line(code_lines, i))
    ratio = covered_code_lines / total_code_lines if total_code_lines else 0.0
    print(f"[{tag}] used {covered_code_lines} / {total_code_lines} code lines = {ratio:.1%}")
    return covered_code_lines, total_code_lines, ratio


# -------------------------
# Policy loader (optional)
# policy.json format example:
# {
#   "kind_policy": {
#     "io_output_set": { "slice": { "use_control": true, "use_data": true } },
#     "control_region": { "slice": { "slice_strategy": "region" } }
#   }
# }
# -------------------------
def load_policy_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _base_kind(kind: str) -> str:
    return kind[:-4] if kind.endswith("_set") else kind


def apply_policy_to_criteria(criteria: List[SlicingCriterion], policy: Dict[str, Any]) -> List[SlicingCriterion]:
    """
    将 policy.json 的 kind_policy 合并进 criterion.extra。
    约定：policy 使用 extra["slice"] / extra["post"] 结构（与 PolicyResolver 一致）。
    """
    kind_policy = (policy or {}).get("kind_policy") or {}
    if not kind_policy:
        return criteria

    out: List[SlicingCriterion] = []
    for c in criteria:
        extra = (getattr(c, "extra", None) or {}).copy()

        kp = kind_policy.get(c.kind) or kind_policy.get(_base_kind(c.kind))
        if isinstance(kp, dict) and kp:
            # policy 优先级低于 criterion.extra：先合并 policy 再覆盖 extra
            merged = dict(kp)
            # deep merge for "slice"/"post"
            for k in ("slice", "post"):
                if isinstance(merged.get(k), dict) and isinstance(extra.get(k), dict):
                    tmp = dict(merged[k])
                    tmp.update(extra[k])
                    merged[k] = tmp
            for k, v in extra.items():
                if k not in ("slice", "post"):
                    merged[k] = v
            extra = merged

        out.append(SlicingCriterion(node_id=c.node_id, kind=c.kind, variable=c.variable, extra=extra))
    return out


# -------------------------
# Slice nodes per criterion (same semantics as extract_functional_blocks)
# -------------------------
def compute_nodes_for_criterion(
    pdg: ProgramDependenceGraph,
    crit: SlicingCriterion,
) -> Set[int]:
    """
    这里严格对齐 functional_blocks.extract_functional_blocks：
    - slice_strategy 从 PolicyResolver 里算；但 tests 里我们只看 crit.extra 的覆盖效果，
      因为 extract_functional_blocks 会再次 resolve。
    - start_nodes 从 extra.get("start_nodes")，否则 [crit.node_id]
    - var_sensitive / seed_vars / use_data / use_control 从 extra["slice"] 里取
    """
    extra = getattr(crit, "extra", {}) or {}
    slice_cfg = extra.get("slice") or {}

    strategy = slice_cfg.get("slice_strategy") or extra.get("slice_strategy") or "backward"
    start_nodes = extra.get("start_nodes") or [crit.node_id]

    if strategy == "region":
        # tests 端用 region 时，直接退化为单点（因为 region_nodes 在 functional_blocks 内部实现）
        return set(start_nodes)

    return compute_slice_nodes(
        pdg,
        start_nodes,
        var_sensitive=bool(slice_cfg.get("var_sensitive", False)),
        seed_vars=slice_cfg.get("seed_vars"),
        use_data=bool(slice_cfg.get("use_data", True)),
        use_control=bool(slice_cfg.get("use_control", True)),
    )


def add_coverage_fill_criteria(
    pdg: ProgramDependenceGraph,
    criteria: List[SlicingCriterion],
    *,
    max_fill: int = 0,
) -> List[SlicingCriterion]:
    if max_fill <= 0:
        return criteria

    used_nodes: Set[int] = set()
    for c in criteria:
        used_nodes |= compute_nodes_for_criterion(pdg, c)

    uncovered = sorted([nid for nid in getattr(pdg, "nodes", {}).keys() if nid not in used_nodes])
    if not uncovered:
        return criteria

    step = max(1, len(uncovered) // max_fill) if len(uncovered) > max_fill else 1
    sampled = uncovered[::step][:max_fill]

    out = list(criteria)
    for nid in sampled:
        out.append(
            SlicingCriterion(
                node_id=nid,
                kind="coverage_fill",
                variable="",
                extra={"slice": {"slice_strategy": "backward", "var_sensitive": False, "use_control": True, "use_data": True}},
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", type=str, default="mc_moveabsolute.st")
    ap.add_argument("--max-fill", type=int, default=0)
    ap.add_argument("--min-lines", type=int, default=8)
    ap.add_argument("--max-lines", type=int, default=150)
    ap.add_argument("--min-lines-stage", type=int, default=8)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--policy", type=str, default="policy.json")
    args = ap.parse_args()

    # verify loaded modules (to rule out wrong source)
    import st_slicer
    import st_slicer.functional_blocks as fb
    import st_slicer.blocks.pipeline as pl
    print("st_slicer:", Path(st_slicer.__file__).resolve())
    print("functional_blocks:", Path(fb.__file__).resolve())
    print("pipeline:", Path(pl.__file__).resolve())

    code_path = Path(__file__).resolve().parent / args.st
    code = code_path.read_text(encoding="utf-8")
    code_lines = code.splitlines()

    policy_path = (Path(__file__).resolve().parent / args.policy) if args.policy else None
    policy = load_policy_json(policy_path)

    # parse + AST
    tree = parse_st_code(code)
    builder = ASTBuilder(filename=str(code_path))
    pous = builder.visit(tree)

    print("POU 数量:", len(pous))
    if not pous:
        return

    proj_symtab = build_symbol_table(pous)

    for pou in pous:
        print("\n" + "=" * 80)
        print("POU:", pou.name, "vars:", len(pou.vars), "stmts:", len(pou.body))

        # IR
        irb = IRBuilder(pou_name=pou.name)
        for s in pou.body:
            irb.lower_stmt(s)

        # CFG
        cfg_builder = CFGBuilder(irb.instrs)
        cfg = cfg_builder.build()

        # Def-Use
        du_analyzer = DefUseAnalyzer(cfg, ir2ast_stmt=irb.ir2ast_stmt)
        du_result: DefUseResult = du_analyzer.analyze()

        # PDG raw -> program PDG
        raw_pdg = PDGBuilder(cfg, du_result).build()
        prog_pdg: ProgramDependenceGraph = build_program_dependence_graph(irb.instrs, raw_pdg, du=du_result)

        data_edges = sum(len(v) for v in getattr(prog_pdg, "data_pred", {}).values())
        ctrl_edges = sum(len(v) for v in getattr(prog_pdg, "ctrl_pred", {}).values())
        print("PDG edges => data:", data_edges, "control:", ctrl_edges)
        print("du.def2uses size:", len(du_result.def2uses))

        pou_symtab = proj_symtab.get_pou(pou.name)

        # mine criteria (关键：传 prog_pdg/du_result/pou_symtab/irb.ir2ast_stmt/code_lines)
        config = CriterionConfig(
            enable_motion_quantity=False,
            enable_any_def=False,   # 建议由你改造后的 any_def_sparse 接管
            enable_any_call=False,
            max_any_def=15,
            max_any_call=15,
            max_seeds_per_group=8,
            api_prefix_filter_enabled=False,
            enable_structure_seeds=True,
            enable_call_seeds=True,
            enable_any_def_sparse=True,
        )

        criteria = mine_slicing_criteria(
            prog_pdg,
            du_result,
            pou_symtab,
            config,
            ir2ast_stmt=irb.ir2ast_stmt,
            code_lines=code_lines,
        )

        # policy override -> criteria.extra
        criteria = apply_policy_to_criteria(criteria, policy)

        # optional fill criteria
        criteria = add_coverage_fill_criteria(prog_pdg, criteria, max_fill=args.max_fill)

        kind_counter = Counter(c.kind for c in criteria)
        print("\n=== Criterion kind distribution ===")
        for k, v in kind_counter.items():
            print(f"{k}: {v}")

        # RAW coverage: union of slice line numbers for all criteria
        parent_map = build_parent_map_from_ir2ast(irb.ir2ast_stmt)

        raw_used_lines: Set[int] = set()
        for c in criteria:
            nodes = compute_nodes_for_criterion(prog_pdg, c)
            stmts = nodes_to_sorted_ast_stmts(nodes, irb.ir2ast_stmt, parent_map)
            raw_used_lines |= set(stmts_to_line_numbers(stmts, code_lines))

        report_coverage("RAW Coverage", raw_used_lines, code_lines)

        # functional blocks
        blocks = extract_functional_blocks(
            prog_pdg=prog_pdg,
            criteria=criteria,
            ir2ast_stmt=irb.ir2ast_stmt,
            code_lines=code_lines,
            overlap_threshold=args.overlap,
            min_lines=args.min_lines,
            max_lines=args.max_lines,
            min_lines_stage=args.min_lines_stage,
        )

        report_coverage("BLOCK Coverage", blocks, code_lines)
        print(f"\nTotal functional blocks : {len(blocks)}")

        # write blocks
        out_all = code_path.parent / f"{pou.name}_all_blocks.txt"
        out_all.write_text("", encoding="utf-8")

        for idx, block in enumerate(blocks):
            completed = build_completed_block(
                block=copy.deepcopy(block),
                pou_name=pou.name,
                pou_symtab=pou_symtab,
                code_lines=code_lines,
                block_index=idx,
                normalize_else_only_if=True,
            )
            with out_all.open("a", encoding="utf-8") as f:
                f.write(f"\n===== BLOCK {idx} =====\n\n")
                f.write(completed.code)
                f.write("\n")

        print("\nBlocks written to:", out_all)


if __name__ == "__main__":
    main()
