#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
Smoke test pipeline:
  (ST text/file) -> AST -> IR -> CFG -> Def-Use -> PDG
and compute a structural signature: skeleton_id

Usage:
  # 1) Parse a real ST file (requires your ANTLR generated lexer/parser to be importable)
  python main.py --st path/to/example.st

  # 2) Demo mode (no parser needed): build a handcrafted AST and run IR/CFG/DU/PDG
  python main.py --demo

  # 3) Batch a directory of .st files
  python main.py --st-dir path/to/st_dir --glob "*.st"

  # Optional: write a jsonl summary for each POU
  python main.py --st path/to/example.st --jsonl out.jsonl
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from st_slicer.generated.IEC61131Lexer import IEC61131Lexer
from st_slicer.generated.IEC61131Parser import IEC61131Parser
from st_slicer.ast.builder import ASTBuilder

# Import helpers
def _import_first(candidates: List[str]):
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(f"Cannot import any of: {candidates}\nLast error: {last_err}")


def resolve_project_modules():
    """
    Tries a few common layouts:
      - package layout: st_slicer.ast.nodes, st_slicer.ir.ir_builder, ...
      - flat layout (your uploaded filenames): nodes, ir_builder, cfg_builder, def_use, pdg_builder
    """
    ast_nodes_mod = _import_first([
        "st_slicer.ast.nodes",
        "ast.nodes",
        "nodes",
    ])

    ir_builder_mod = _import_first([
        "st_slicer.ir.ir_builder",
        "ir.ir_builder",
        "ir_builder",
    ])

    ir_nodes_mod = _import_first([
        "st_slicer.ir.ir_nodes",
        "ir.ir_nodes",
        "ir_nodes",
    ])

    cfg_builder_mod = _import_first([
        "st_slicer.cfg.cfg_builder",
        "cfg.cfg_builder",
        "cfg_builder",
    ])

    def_use_mod = _import_first([
        "st_slicer.dataflow.def_use",
        "dataflow.def_use",
        "def_use",
    ])

    pdg_builder_mod = _import_first([
        "st_slicer.pdg.pdg_builder",
        "pdg.pdg_builder",
        "pdg_builder",
    ])

    return {
        "ast_nodes": ast_nodes_mod,
        "ir_builder": ir_builder_mod,
        "ir_nodes": ir_nodes_mod,
        "cfg_builder": cfg_builder_mod,
        "def_use": def_use_mod,
        "pdg_builder": pdg_builder_mod,
    }


# ST -> AST (optional)
def parse_st_to_pous(st_text: str, filename: str) -> List[Any]:
    """
    Parse ST text using your ANTLR-generated lexer/parser if available.

    Expected modules (common):
      - st_slicer.ast.generated.IEC61131Lexer / IEC61131Parser
      - st_slicer.ast.builder.ASTBuilder  (visitor)
    If these are not present, raise and let caller fallback to --demo.
    """
    # antlr runtime
    from antlr4 import InputStream, CommonTokenStream  # type: ignore

    # Try likely generated module paths
    lexer_mod = _import_first([
        "st_slicer.generated.IEC61131Lexer",      # 你的真实路径
        "st_slicer.ast.generated.IEC61131Lexer",  # 兼容旧路径（可留）
        "ast.generated.IEC61131Lexer",
        "generated.IEC61131Lexer",
    ])

    parser_mod = _import_first([
        "st_slicer.generated.IEC61131Parser",      # 你的真实路径
        "st_slicer.ast.generated.IEC61131Parser",  # 兼容旧路径（可留）
        "ast.generated.IEC61131Parser",
        "generated.IEC61131Parser",
    ])
    builder_mod = _import_first([
        "st_slicer.ast.builder",
        "ast.builder",
        "builder",
    ])

    IEC61131Lexer = getattr(lexer_mod, "IEC61131Lexer")
    IEC61131Parser = getattr(parser_mod, "IEC61131Parser")
    ASTBuilder = getattr(builder_mod, "ASTBuilder")

    input_stream = InputStream(st_text)
    lexer = IEC61131Lexer(input_stream)
    tokens = CommonTokenStream(lexer)
    parser = IEC61131Parser(tokens)

    # Start rule: try common ones
    start_rule = None
    for rule_name in ("start", "program", "compilationUnit", "pou", "pous"):
        if hasattr(parser, rule_name):
            start_rule = getattr(parser, rule_name)
            break
    if start_rule is None:
        raise RuntimeError("Cannot find a parser start rule among: start/program/compilationUnit/pou/pous")

    tree = start_rule()
    visitor = ASTBuilder(filename=filename)

    # Your builder exposes visitStart in code; fall back to generic visit if needed.
    if hasattr(visitor, "visitStart"):
        pous = visitor.visitStart(tree)
    else:
        pous = visitor.visit(tree)

    if pous is None:
        return []
    if isinstance(pous, list):
        return pous
    return [pous]

# Pretty printing + skeleton_id
_TEMP_RE = re.compile(r"^t\d+$")


def _is_temp(name: str) -> bool:
    return bool(_TEMP_RE.match(name))


def _is_const(token: str) -> bool:
    # crude but practical: integers, floats, TRUE/FALSE, quoted strings
    if token.upper() in ("TRUE", "FALSE"):
        return True
    if re.fullmatch(r"-?\d+(\.\d+)?", token):
        return True
    if (len(token) >= 2 and ((token[0] == "'" and token[-1] == "'") or (token[0] == '"' and token[-1] == '"'))):
        return True
    return False


def compute_skeleton_id(ir_instrs: List[Any], keep_callee: bool = True) -> str:
    """
    Structural signature for one POU, intended for "structure-bucket" sampling/augmentation.

    Normalizations:
      - labels normalized by encounter order: L0, L1, ...
      - variables normalized by encounter order: v0, v1, ...
      - temps tN kept as 't'
      - constants mapped to 'c'
      - calls include (callee or fK) + arity
    """
    label_map: Dict[str, str] = {}
    var_map: Dict[str, str] = {}
    callee_map: Dict[str, str] = {}

    def norm_label(lb: str) -> str:
        if lb not in label_map:
            label_map[lb] = f"L{len(label_map)}"
        return label_map[lb]

    def norm_var(x: Optional[str]) -> str:
        if x is None:
            return "None"
        if _is_const(x):
            return "c"
        if _is_temp(x):
            return "t"
        # treat everything else as "program var"
        if x not in var_map:
            var_map[x] = f"v{len(var_map)}"
        return var_map[x]

    def norm_callee(c: str) -> str:
        if keep_callee:
            return c
        if c not in callee_map:
            callee_map[c] = f"f{len(callee_map)}"
        return callee_map[c]

    tokens: List[str] = []
    for instr in ir_instrs:
        cname = instr.__class__.__name__

        if cname == "IRLabel":
            tokens.append(f"LABEL({norm_label(instr.name)})")

        elif cname == "IRGoto":
            tokens.append(f"GOTO({norm_label(instr.target_label)})")

        elif cname == "IRBranchCond":
            tokens.append(f"BR({norm_var(instr.cond)},{norm_label(instr.true_label)},{norm_label(instr.false_label)})")

        elif cname == "IRAssign":
            tokens.append(f"ASSIGN({norm_var(instr.target)},{norm_var(instr.src)})")

        elif cname == "IRBinOp":
            tokens.append(f"BIN({instr.op},{norm_var(instr.dest)},{norm_var(instr.left)},{norm_var(instr.right)})")

        elif cname == "IRCall":
            arity = len(getattr(instr, "args", []) or [])
            tokens.append(f"CALL({norm_callee(instr.callee)},{arity},{norm_var(instr.dest)})")

        else:
            # unknown/extended IR instructions
            tokens.append(f"OTHER({cname})")

    s = "|".join(tokens).encode("utf-8")
    return hashlib.md5(s).hexdigest()


def format_ir(ir_instrs: List[Any], max_lines: int = 200) -> str:
    out: List[str] = []
    for i, instr in enumerate(ir_instrs[:max_lines]):
        cname = instr.__class__.__name__
        if cname == "IRAssign":
            out.append(f"{i:04d}: {instr.target} := {instr.src}")
        elif cname == "IRBinOp":
            out.append(f"{i:04d}: {instr.dest} := {instr.left} {instr.op} {instr.right}")
        elif cname == "IRCall":
            args = ", ".join(instr.args)
            out.append(f"{i:04d}: {instr.dest+' := ' if instr.dest else ''}{instr.callee}({args})")
        elif cname == "IRBranchCond":
            out.append(f"{i:04d}: IF {instr.cond} THEN GOTO {instr.true_label} ELSE GOTO {instr.false_label}")
        elif cname == "IRLabel":
            out.append(f"{i:04d}: LABEL {instr.name}")
        elif cname == "IRGoto":
            out.append(f"{i:04d}: GOTO {instr.target_label}")
        else:
            out.append(f"{i:04d}: {cname} {instr.__dict__}")
    if len(ir_instrs) > max_lines:
        out.append(f"... ({len(ir_instrs)-max_lines} more)")
    return "\n".join(out)

def ir_stats(ir_instrs: List[Any]) -> Dict[str, int]:
    def is_type(x, name: str) -> bool:
        return x.__class__.__name__ == name

    return {
        "n_ir_label": sum(1 for ins in ir_instrs if is_type(ins, "IRLabel")),
        "n_ir_goto": sum(1 for ins in ir_instrs if is_type(ins, "IRGoto")),
        "n_ir_branch": sum(1 for ins in ir_instrs if is_type(ins, "IRBranchCond")),
        "n_ir_call": sum(1 for ins in ir_instrs if is_type(ins, "IRCall")),
        "n_ir_assign": sum(1 for ins in ir_instrs if is_type(ins, "IRAssign")),
        "n_ir_binop": sum(1 for ins in ir_instrs if is_type(ins, "IRBinOp")),
    }


def cfg_extra_stats(cfg) -> Dict[str, int]:
    # 边数
    n_edges = sum(len(v) for v in cfg.succ.values()) if getattr(cfg, "succ", None) else 0

    # 回边：用一个简单、稳定的判据（succ 节点编号 <= 当前节点编号）
    # 这与你之前的思路一致，也便于论文复现。
    back_edges = 0
    if getattr(cfg, "succ", None):
        for i, succs in cfg.succ.items():
            for j in succs:
                if j <= i:
                    back_edges += 1

    n_nodes = len(getattr(cfg, "instrs", []))  # 指令级 CFG
    n_exits = len(getattr(cfg, "exits", [])) if getattr(cfg, "exits", None) is not None else 0

    return {
        "n_cfg_nodes": n_nodes,
        "n_cfg_edges": n_edges,
        "n_cfg_backedge": back_edges,
        "n_cfg_exits": n_exits,
    }


def defuse_extra_stats(du) -> Dict[str, int]:
    # du.def_vars / du.use_vars: List[Set[str]]
    def_vars = getattr(du, "def_vars", [])
    use_vars = getattr(du, "use_vars", [])

    n_def_sites = sum(len(s) for s in def_vars)
    n_use_sites = sum(len(s) for s in use_vars)

    uniq_all = set()
    for s in def_vars:
        uniq_all |= set(s)
    for s in use_vars:
        uniq_all |= set(s)

    # 去掉临时变量 t\d+ 和常量（按你现有 _is_const/_is_temp 规则）
    uniq_prog = set()
    for v in uniq_all:
        if v is None:
            continue
        if _is_temp(v):
            continue
        if _is_const(v):
            continue
        uniq_prog.add(v)

    return {
        "n_def_sites": n_def_sites,
        "n_use_sites": n_use_sites,
        "n_unique_vars_all": len(uniq_all),
        "n_unique_vars_prog": len(uniq_prog),
    }


def pdg_extra_stats(pdg) -> Dict[str, int]:
    n_data_edges = sum(len(v) for v in pdg.data_deps.values()) if getattr(pdg, "data_deps", None) else 0
    n_ctrl_edges = sum(len(v) for v in pdg.control_deps.values()) if getattr(pdg, "control_deps", None) else 0
    return {"n_pdg_data_edges": n_data_edges, "n_pdg_ctrl_edges": n_ctrl_edges}


# Demo AST (no parser needed)
def build_demo_pou(ast_nodes_mod):
    """
    Handcrafted AST to test the IR/CFG/DU/PDG pipeline without parsing.

    Program demo:
      A := 1;
      IF (A < 3) THEN
        B := A + 1;
      ELSE
        B := 0;
      END_IF;
      WHILE (B < 5) DO
        B := B + 1;
      END_WHILE;
    """
    SourceLocation = getattr(ast_nodes_mod, "SourceLocation")
    ProgramDecl = getattr(ast_nodes_mod, "ProgramDecl")
    VarDecl = getattr(ast_nodes_mod, "VarDecl")

    Assignment = getattr(ast_nodes_mod, "Assignment")
    IfStmt = getattr(ast_nodes_mod, "IfStmt")
    WhileStmt = getattr(ast_nodes_mod, "WhileStmt")

    VarRef = getattr(ast_nodes_mod, "VarRef")
    Literal = getattr(ast_nodes_mod, "Literal")
    BinOp = getattr(ast_nodes_mod, "BinOp")

    # 注意：参数名是 column，不是 col
    loc = SourceLocation(file="<demo>", line=1, column=0)

    # 注意：VarDecl字段是 name/type/storage/init_expr/loc
    vars_ = [
        VarDecl(name="A", type="INT", storage="VAR", init_expr=None, loc=loc),
        VarDecl(name="B", type="INT", storage="VAR", init_expr=None, loc=loc),
    ]

    # 注意：Literal需要 type 字段（这里用 INT）
    stmt1 = Assignment(
        target=VarRef(name="A", loc=loc),
        value=Literal(value=1, type="INT", loc=loc),
        loc=loc
    )

    cond_if = BinOp(
        op="<",
        left=VarRef(name="A", loc=loc),
        right=Literal(value=3, type="INT", loc=loc),
        loc=loc
    )

    then_stmt = Assignment(
        target=VarRef(name="B", loc=loc),
        value=BinOp(
            op="+",
            left=VarRef(name="A", loc=loc),
            right=Literal(value=1, type="INT", loc=loc),
            loc=loc
        ),
        loc=loc
    )

    else_stmt = Assignment(
        target=VarRef(name="B", loc=loc),
        value=Literal(value=0, type="INT", loc=loc),
        loc=loc
    )

    # IfStmt: cond/then_body/elif_branches/else_body/loc
    stmt2 = IfStmt(
        cond=cond_if,
        then_body=[then_stmt],
        elif_branches=[],
        else_body=[else_stmt],
        loc=loc
    )

    cond_while = BinOp(
        op="<",
        left=VarRef(name="B", loc=loc),
        right=Literal(value=5, type="INT", loc=loc),
        loc=loc
    )

    body_while = Assignment(
        target=VarRef(name="B", loc=loc),
        value=BinOp(
            op="+",
            left=VarRef(name="B", loc=loc),
            right=Literal(value=1, type="INT", loc=loc),
            loc=loc
        ),
        loc=loc
    )

    stmt3 = WhileStmt(cond=cond_while, body=[body_while], loc=loc)

    # ProgramDecl: name/vars/body/loc
    return ProgramDecl(name="DEMO", vars=vars_, body=[stmt1, stmt2, stmt3], loc=loc)

# End-to-end pipeline
def run_pipeline_on_pou(pou: Any, mods: Dict[str, Any], source_file: str = "") -> Dict[str, Any]:
    IRBuilder = getattr(mods["ir_builder"], "IRBuilder")
    CFGBuilder = getattr(mods["cfg_builder"], "CFGBuilder")
    DefUseAnalyzer = getattr(mods["def_use"], "DefUseAnalyzer")
    PDGBuilder = getattr(mods["pdg_builder"], "PDGBuilder")
    build_program_dependence_graph = getattr(mods["pdg_builder"], "build_program_dependence_graph")

    # 1) AST -> IR
    irb = IRBuilder(pou_name=getattr(pou, "name", "<POU>"))
    for s in getattr(pou, "body", []):
        irb.lower_stmt(s)

    ir_instrs = irb.instrs
    skeleton_id = compute_skeleton_id(ir_instrs, keep_callee=True)

    # IR stats
    s_ir = ir_stats(ir_instrs)

    # 2) IR -> CFG
    cfg = CFGBuilder(ir_instrs).build()
    s_cfg = cfg_extra_stats(cfg)

    # 3) CFG -> Def-Use
    du = DefUseAnalyzer(cfg, ir2ast_stmt=getattr(irb, "ir2ast_stmt", None)).analyze()
    s_du = defuse_extra_stats(du)

    # 4) CFG + DU -> PDG
    pdg = PDGBuilder(cfg, du).build()
    _ = build_program_dependence_graph(ir_instrs, pdg, du)  # 你如果不需要 predecessors 查询，可不返回
    s_pdg = pdg_extra_stats(pdg)

    # 结构统计向量（用于 kNN 回退/距离度量）
    z_vec = {
        "n_instr": len(ir_instrs),
        "n_cfg_edges": s_cfg["n_cfg_edges"],
        "n_branch": s_ir["n_ir_branch"],
        "n_backedge": s_cfg["n_cfg_backedge"],
        "n_pdg_data_edges": s_pdg["n_pdg_data_edges"],
        "n_pdg_ctrl_edges": s_pdg["n_pdg_ctrl_edges"],
        "n_def_sites": s_du["n_def_sites"],
        "n_use_sites": s_du["n_use_sites"],
        "n_unique_vars_prog": s_du["n_unique_vars_prog"],
    }

    # 统一输出
    out = {
        # 追溯信息
        "source_file": source_file,
        "pou_name": getattr(pou, "name", "<POU>"),

        # 核心签名
        "skeleton_id": skeleton_id,

        # 统计信息（建议全部保留，后续做消融/筛选/配对都会用到）
        **s_ir,
        **s_cfg,
        **s_du,
        **s_pdg,

        # 用于近邻配对的结构向量
        "z_vec": z_vec,

        # 打印调试用（默认打印时用；写 jsonl 时建议丢弃）
        "ir_preview": format_ir(ir_instrs, max_lines=200),
        "cfg_entry": getattr(cfg, "entry", None),
        "cfg_exits": sorted(list(getattr(cfg, "exits", []))) if getattr(cfg, "exits", None) is not None else [],
    }

    return out



def iter_st_files(st_path: Optional[str], st_dir: Optional[str], glob_pat: str) -> List[str]:
    if st_path:
        return [st_path]
    if st_dir:
        out = []
        for root, _, files in os.walk(st_dir):
            for f in files:
                if fnmatch.fnmatch(f, glob_pat):
                    out.append(os.path.join(root, f))
        return sorted(out)
    return []

# 配置：直接在这里改路径即可
FIXED_DIR = Path(r"F:\study\postgtaduate\AIPython\st_code\Mix_code\failed_only")  # 你的 fixedCode 目录
IN_DIR = FIXED_DIR                      # 输入：递归读取该目录下所有 .st
OUT_JSONL = FIXED_DIR / "dataset.jsonl" # 输出：成功的 POU 写到这里
FAILED_TXT = FIXED_DIR / "failed_parse.txt"  # 输出：解析失败文件清单写到这里
EXT = ".st"                             # 处理后缀
QUIET = False                           # True 就少打印


def main():
    mods = resolve_project_modules()

    if not IN_DIR.exists():
        raise SystemExit(f"IN_DIR not found: {IN_DIR}")

    # 收集失败项：建议写“相对路径 + 错误摘要”，方便你回头定位
    failed_items: List[str] = []

    # 统计
    n_files_total = 0
    n_files_failed = 0
    n_pous_ok = 0
    n_pous_failed = 0

    # 确保输出目录存在
    FIXED_DIR.mkdir(parents=True, exist_ok=True)

    # 流式写 jsonl：边解析边写，避免 results 太大
    with open(OUT_JSONL, "w", encoding="utf-8") as wf:
        # 递归遍历 .st
        for fp in sorted(IN_DIR.rglob(f"*{EXT}")):
            if not fp.is_file():
                continue

            n_files_total += 1
            rel = str(fp.relative_to(IN_DIR))

            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                n_files_failed += 1
                failed_items.append(f"{rel}\tREAD_ERROR: {str(e).replace('\n',' ')[:500]}")
                continue

            # 关键：解析失败不中断
            try:
                pous = parse_st_to_pous(txt, filename=str(fp))
            except Exception as e:
                n_files_failed += 1
                failed_items.append(f"{rel}\tPARSE_ERROR: {str(e).replace('\n',' ')[:500]}")
                continue

            if not pous:
                n_files_failed += 1
                failed_items.append(f"{rel}\tPARSE_EMPTY: no POU returned")
                continue

            # 单个 POU 的 IR/CFG/PDG pipeline 也可能失败：同样不中断
            for pou in pous:
                try:
                    r = run_pipeline_on_pou(pou, mods, source_file=str(fp))

                    # 丢弃调试字段（训练/配对不需要）
                    rr = dict(r)
                    rr.pop("ir_preview", None)
                    rr.pop("cfg_entry", None)
                    rr.pop("cfg_exits", None)

                    wf.write(json.dumps(rr, ensure_ascii=False) + "\n")
                    n_pous_ok += 1

                    if not QUIET:
                        print(f"[OK] {rel} :: POU={rr.get('pou_name')} skeleton_id={rr.get('skeleton_id')}")

                except Exception as e:
                    n_pous_failed += 1
                    failed_items.append(f"{rel}\tPIPELINE_ERROR: {str(e).replace('\n',' ')[:500]}")
                    continue

    # 写 failed_parse.txt（放在 fixedCode 目录）
    if failed_items:
        FAILED_TXT.write_text("\n".join(failed_items) + "\n", encoding="utf-8", errors="ignore")

    if not QUIET:
        print("\nDone.")
        print(f"IN_DIR          : {IN_DIR}")
        print(f"OUT_JSONL       : {OUT_JSONL}")
        print(f"FAILED_TXT      : {FAILED_TXT}")
        print(f"Total .st files : {n_files_total}")
        print(f"Failed files    : {n_files_failed}")
        print(f"POU ok          : {n_pous_ok}")
        print(f"POU failed      : {n_pous_failed}")

if __name__ == "__main__":
    main()
