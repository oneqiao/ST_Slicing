# tests/main.py

from pathlib import Path
from typing import List, Set

from st_slicer.parser.parse_st import parse_st_code
from st_slicer.ast.builder import ASTBuilder
from st_slicer.ir.ir_builder import IRBuilder
from st_slicer.cfg.cfg_builder import CFGBuilder
from st_slicer.dataflow.def_use import DefUseAnalyzer, DefUseResult
from st_slicer.pdg.pdg_builder import PDGBuilder, build_program_dependence_graph, ProgramDependenceGraph
from st_slicer.criteria import mine_slicing_criteria, CriterionConfig, SlicingCriterion
from st_slicer.sema.builder import build_symbol_table
from collections import Counter

from st_slicer.functional_blocks import extract_functional_blocks
from st_slicer.block_context import build_completed_block

from st_slicer.blocks.pipeline import (
    compute_slice_nodes,
    nodes_to_sorted_ast_stmts,
    build_parent_map_from_ir2ast,
    stmts_to_line_numbers,
)

#计算切片代码的覆盖率
def report_coverage(blocks, code_lines):
    used = set()
    for b in blocks:
        used.update(b.line_numbers)

    def is_code_line(i: int) -> bool:
        line = code_lines[i - 1]
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("(*") and stripped.endswith("*)"):
            return False
        if stripped.startswith("//"):
            return False
        return True

    total_code_lines = sum(1 for i in range(1, len(code_lines) + 1) if is_code_line(i))
    covered_code_lines = sum(1 for i in used if 1 <= i <= len(code_lines) and is_code_line(i))

    ratio = covered_code_lines / total_code_lines if total_code_lines else 0.0
    print(f"[Coverage] used {covered_code_lines} / {total_code_lines} code lines = {ratio:.1%}")

def add_coverage_fill_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    criteria: List[SlicingCriterion],
) -> List[SlicingCriterion]:
    """
    覆盖兜底准则：保证每个 PDG 节点至少出现在一个切片中。

    做法：
      1) 用现有 criteria 跑一次 compute_slice_nodes，收集所有已覆盖的 node_id；
      2) 对未覆盖的 node_id，追加 kind="coverage_fill" 的准则。
    """
    used_nodes: Set[int] = set()

    for crit in criteria:
        nodes = compute_slice_nodes(pdg, crit.node_id)
        used_nodes.update(nodes)

    new_criteria = list(criteria)

    for node_id in pdg.nodes.keys():
        if node_id in used_nodes:
            continue

        new_criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind="coverage_fill",
                variable="<fill>",
                extra={},
            )
        )

    return new_criteria

def main():
    # 这里改成你想测试的 ST 文件名
    code_path = Path(__file__).parent / "mc_moveabsolute.st"
    code = code_path.read_text(encoding="utf-8")
    code_lines = code.splitlines()

    # 0) 解析 + AST
    tree = parse_st_code(code)
    builder = ASTBuilder(filename=str(code_path))
    pous = builder.visit(tree)

    print("POU 数量:", len(pous))
    for pou in pous:
        print("POU:", pou.name, "vars:", len(pou.vars), "stmts:", len(pou.body))

        # 1) IR
        irb = IRBuilder(pou_name=pou.name)
        for s in pou.body:
            irb.lower_stmt(s)
        print(f"\n=== IR for POU {pou.name} ===")
        # 如需调试 IR，可以取消下面注释：
        # for idx, ins in enumerate(irb.instrs):
        #     print(idx, vars(ins))

        # 2) CFG
        cfg_builder = CFGBuilder(irb.instrs)
        cfg = cfg_builder.build()

        # 3) Def-Use（一定要把 ir2ast_stmt 传进去）
        du_analyzer = DefUseAnalyzer(cfg, ir2ast_stmt=irb.ir2ast_stmt)
        du_result = du_analyzer.analyze()

        # 4) PDG（后继风格）
        pdg_builder = PDGBuilder(cfg, du_result)
        raw_pdg = pdg_builder.build()

        # print("\n=== PDG Data Dependencies ===")
        # for src, dsts in sorted(raw_pdg.data_deps.items()):
        #     print(f"{src} --data--> {sorted(dsts)}")

        # print("\n=== PDG Control Dependencies ===")
        # for src, dsts in sorted(raw_pdg.control_deps.items()):
        #     print(f"{src} --ctrl--> {sorted(dsts)}")

        # 5) 构建“前驱风格”的 ProgramDependenceGraph（切片用）
        prog_pdg = build_program_dependence_graph(irb.instrs, raw_pdg)
        # 如需查看每个节点的前驱，可以打开下面的调试输出：
        # print("\n=== ProgramDependenceGraph (predecessor view) ===")
        # for nid, node in sorted(prog_pdg.nodes.items()):
        #     preds = prog_pdg.predecessors(nid)
        #     if preds:
        #         print(f"node {nid} <- {sorted(preds)}")

        # 6) 构符号表（project），再取当前 POU 的 symtab
        proj_symtab = build_symbol_table(pous)
        pou_symtab = proj_symtab.get_pou(pou.name)

        print("\n=== Symbols in POU symtab ===")
        for sym in pou_symtab.get_all_symbols():
            print(sym.name, getattr(sym, "type", None), getattr(sym, "role", None))

        # 7) 准则挖掘
        config = CriterionConfig()
        criteria = mine_slicing_criteria(prog_pdg, du_result, pou_symtab, config)

        # 在此基础上增加 coverage_fill 准则
        criteria = add_coverage_fill_criteria(prog_pdg, du_result, criteria)

        print("\n=== Mined slicing criteria ===")
        for c in criteria:
            print(c)

        if not criteria:
            print("\nNo slicing criteria found for this POU.")
            continue

        kind_counter = Counter(c.kind for c in criteria)
        print("\n=== Criterion kind distribution ===")
        for k, v in kind_counter.items():
            print(f"{k}: {v}")

        # === Debug: 对第一个 criterion 做“补全前/补全后”对比 ===
        first_crit = criteria[0]
        print(f"\n[DEBUG] Show control-structure completion effect for criterion: {first_crit}")

        # 0) 构造 parent_map（用你刚刚修好的版本）
        parent_map = build_parent_map_from_ir2ast(irb.ir2ast_stmt)

        # 1) 先算这个准则的切片节点集合
        slice_nodes = compute_slice_nodes(prog_pdg, first_crit.node_id)

        # 2) 不做控制补全：parent_map 传空 dict
        stmts_raw = nodes_to_sorted_ast_stmts(slice_nodes, irb.ir2ast_stmt, parent_map={})
        raw_lines = stmts_to_line_numbers(stmts_raw, code_lines)

        print("\n--- RAW slice without control completion ---")
        print("Lines:", raw_lines)
        for ln in raw_lines:
            if 1 <= ln <= len(code_lines):
                print(f"{ln:4d}: {code_lines[ln-1].rstrip()}")

        # 3) 做控制补全：用真正的 parent_map
        stmts_closed = nodes_to_sorted_ast_stmts(slice_nodes, irb.ir2ast_stmt, parent_map=parent_map)
        closed_lines = stmts_to_line_numbers(stmts_closed, code_lines)

        print("\n--- CLOSED slice with control completion ---")
        print("Lines:", closed_lines)
        for ln in closed_lines:
            if 1 <= ln <= len(code_lines):
                print(f"{ln:4d}: {code_lines[ln-1].rstrip()}")

        #8) 一键“多准则切片 + 聚类 + Stage 切分 + 块大小规范化”
        #    extract_functional_blocks 内部会：
        #      - 对每个准则做 backward slice；
        #      - 用 overlap clustering 合并高度重叠的切片；
        #      - 基于 Stage (stage/Stage) 做一次语义切分；
        #      - 再用 min_lines / max_lines 做尺寸规范化。
        blocks = extract_functional_blocks(
            prog_pdg=prog_pdg,
            criteria=criteria,
            ir2ast_stmt=irb.ir2ast_stmt,
            code_lines=code_lines,
            overlap_threshold=0.5,  # 两个切片重叠比例 >= 0.5 就归为同一功能块
            min_lines=8,           # 每个块至少 20 行（Stage 切分时也会用到）
            max_lines=150,          # 每个块最多 150 行
        )

        #覆盖率
        report_coverage(blocks, code_lines)

        print(f"\nTotal functional blocks : {len(blocks)}")

        out_all = code_path.parent / f"{pou.name}_all_blocks11.txt"
        out_all.write_text("", encoding="utf-8")  # 每次运行清空，避免历史累积干扰判断

        # 9) 对每个功能块做结构补全，生成独立 PROGRAM
        for idx, block in enumerate(blocks):
            completed = build_completed_block(
                block=block,
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

            # 如果你希望写成单独文件，可以取消下面注释：
            # out_name = f"{pou.name}_block_{idx}.st"
            # out_path = code_path.parent / out_name
            # out_path.write_text(completed.code, encoding="utf-8")
            # print(f"\n[Saved completed block to {out_path}]")


if __name__ == "__main__":
    main()
