#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 配置：把路径写死在这里（按你本机目录修改）
# ============================================================
IN_DIR = r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode"  # 输入：你清洗后的 fixedCode
OUT_JSONL = r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode\meta2.jsonl"
FAILED_TXT = r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode\failed_parse2.txt"
GLOB_PAT = "*.st"

# L2 是否保留 callee 名字（强烈建议 True）
KEEP_CALLEE_L2 = True

# ============================================================
# Import helpers
# ============================================================
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
    你的项目 layout 兼容：
      - st_slicer.ast.nodes / st_slicer.ir.ir_builder / st_slicer.cfg.cfg_builder / st_slicer.dataflow.def_use / st_slicer.pdg.pdg_builder
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
        "cfg_builder": cfg_builder_mod,
        "def_use": def_use_mod,
        "pdg_builder": pdg_builder_mod,
    }


# ============================================================
# ST -> AST
# ============================================================
def parse_st_to_pous(st_text: str, filename: str) -> List[Any]:
    from antlr4 import InputStream, CommonTokenStream  # type: ignore

    lexer_mod = _import_first([
        "st_slicer.generated.IEC61131Lexer",
        "st_slicer.ast.generated.IEC61131Lexer",
        "ast.generated.IEC61131Lexer",
        "generated.IEC61131Lexer",
    ])

    parser_mod = _import_first([
        "st_slicer.generated.IEC61131Parser",
        "st_slicer.ast.generated.IEC61131Parser",
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

    start_rule = None
    for rule_name in ("start", "compilationUnit", "pou", "pous", "program"):
        if hasattr(parser, rule_name):
            start_rule = getattr(parser, rule_name)
            break
    if start_rule is None:
        raise RuntimeError("Cannot find parser start rule among: start/compilationUnit/pou/pous/program")

    tree = start_rule()
    visitor = ASTBuilder(filename=filename)

    if hasattr(visitor, "visitStart"):
        pous = visitor.visitStart(tree)
    else:
        pous = visitor.visit(tree)

    if pous is None:
        return []
    if isinstance(pous, list):
        return pous
    return [pous]


# ============================================================
# Advanced normalization for skeleton ids
# ============================================================
_TEMP_RE = re.compile(r"^t\d+$", re.IGNORECASE)

def _is_temp(name: str) -> bool:
    return bool(name) and bool(_TEMP_RE.match(name))

_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")

def _is_const(token: Optional[str]) -> bool:
    if token is None:
        return False
    t = token.strip()
    if not t:
        return False
    if t.upper() in ("TRUE", "FALSE"):
        return True
    if _NUM_RE.match(t):
        return True
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        return True
    return False

def _log2bin(x: int) -> int:
    x = max(0, int(x))
    return int(math.log2(x + 1))

def compute_bucket_z(z_vec: Dict[str, Any], n_ir_call: int) -> str:
    """
    一个“分层桶兜底”的 Z-bucket：
    - 对规模类特征做 log2 分箱（避免过碎）
    - 对结构关键点保持离散（branch、backedge）
    你后续可以很方便地调哪些维度进入 bucket。
    """
    n_instr = int(z_vec.get("n_instr", 0))
    n_edges = int(z_vec.get("n_cfg_edges", 0))
    n_branch = int(z_vec.get("n_branch", 0))
    n_backedge = int(z_vec.get("n_backedge", 0))
    n_pdg_data = int(z_vec.get("n_pdg_data_edges", 0))
    n_pdg_ctrl = int(z_vec.get("n_pdg_ctrl_edges", 0))

    # 分层：规模用 log2bin，结构用原子离散/布尔
    return (
        f"I{_log2bin(n_instr)}"
        f"_E{_log2bin(n_edges)}"
        f"_B{min(n_branch, 7)}"          # branch 过大时截断，避免桶碎
        f"_L{1 if n_backedge > 0 else 0}"
        f"_C{_log2bin(n_ir_call)}"
        f"_PD{_log2bin(n_pdg_data)}"
        f"_PC{_log2bin(n_pdg_ctrl)}"
    )

def _const_bucket(token: str) -> str:
    """
    将常量分箱，减少“具体值”导致的裂桶：
      - BOOL
      - INT: 0/1/-1/SMALL(2..8)/NEG_SMALL(-2..-8)/INT
      - REAL
      - STR
      - OTHER_CONST
    """
    t = token.strip()
    up = t.upper()
    if up in ("TRUE", "FALSE"):
        return "BOOL"
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        return "STR"

    # numeric
    if _NUM_RE.match(t):
        # try int first
        try:
            if "." not in t and "e" not in t.lower():
                v = int(t, 10)
                if v == 0:
                    return "I0"
                if v == 1:
                    return "I1"
                if v == -1:
                    return "IM1"
                if 2 <= v <= 8:
                    return "IS"
                if -8 <= v <= -2:
                    return "INS"
                return "I"
            # float
            _ = float(t)
            return "R"
        except Exception:
            return "NUM"
    return "CONST"


COMMUTATIVE_OPS = {
    "+", "*", "AND", "OR", "XOR", "=",
}

def _op_norm(op: str) -> str:
    return (op or "").upper()


def _callee_family(callee: str) -> str:
    """
    L1 层：把 callee 归一为“功能族”，用于兜底桶。
    你可以按 ST 指令集继续扩展。
    """
    u = (callee or "").upper()

    if "_TO_" in u or u.endswith("_TO") or u.startswith("TO_"):
        return "TYPE_CONV"
    if "SHL" in u or "SHR" in u or u in {"USHLW", "USHRW", "SHL", "SHR"}:
        return "SHIFT"
    if u.startswith(("GT", "GE", "LT", "LE", "EQ", "NE")) or "CMP" in u:
        return "CMP"
    if u in {"ADD", "SUB", "MUL", "DIV", "MOD"}:
        return "ARITH"
    if u in {"AND", "OR", "XOR", "NOT"}:
        return "LOGIC"

    # 常见库函数可按需加白名单
    return "CALL"


class RoleVarEncoder:
    """
    变量“角色化”编码器：
      - 同一 role 内独立编号，降低“出现顺序差异”对 skeleton 的影响
      - temp/const 单独处理
    """
    def __init__(self, const_mode: str):
        """
        const_mode:
          - "bucket": 用 _const_bucket 分箱
          - "collapse": 常量全部折叠为 "C"
        """
        self.const_mode = const_mode
        self.role_maps: Dict[str, Dict[str, str]] = {}
        self.label_map: Dict[str, str] = {}
        self.callee_map: Dict[str, str] = {}

    def norm_label(self, lb: str) -> str:
        if lb not in self.label_map:
            self.label_map[lb] = f"L{len(self.label_map)}"
        return self.label_map[lb]

    def norm_callee(self, c: str, keep: bool, l1: bool) -> str:
        if l1:
            return _callee_family(c)
        if keep:
            return c
        # 若不保留，改成 f0/f1...
        if c not in self.callee_map:
            self.callee_map[c] = f"f{len(self.callee_map)}"
        return self.callee_map[c]

    def norm_atom(self, x: Optional[str], role: str) -> str:
        if x is None:
            return "N"
        s = str(x)

        if _is_const(s):
            if self.const_mode == "collapse":
                return "C"
            return f"C:{_const_bucket(s)}"

        if _is_temp(s):
            return "T"

        # program var: role-based stable ids
        if role not in self.role_maps:
            self.role_maps[role] = {}
        rm = self.role_maps[role]
        if s not in rm:
            rm[s] = f"{role}{len(rm)}"
        return rm[s]

    def shape_atom(self, x: Optional[str]) -> str:
        """
        给 IRCall 的参数用：只保留“形态”，不引入 role 编号，避免轻易裂桶。
        """
        if x is None:
            return "N"
        s = str(x)
        if _is_const(s):
            if self.const_mode == "collapse":
                return "C"
            return f"C{_const_bucket(s)}"
        if _is_temp(s):
            return "T"
        return "V"


def _compute_skeleton_tokens(
    ir_instrs: List[Any],
    *,
    keep_callee: bool,
    l1: bool,
) -> List[str]:
    """
    输出一组 tokens，用于 hash：
      - l1=False: L2（严格）
      - l1=True : L1（松，callee 族归一 + 常量折叠）
    """
    enc = RoleVarEncoder(const_mode=("collapse" if l1 else "bucket"))
    toks: List[str] = []

    for instr in ir_instrs:
        cname = instr.__class__.__name__

        if cname == "IRLabel":
            toks.append(f"LABEL({enc.norm_label(instr.name)})")

        elif cname == "IRGoto":
            toks.append(f"GOTO({enc.norm_label(instr.target_label)})")

        elif cname == "IRBranchCond":
            # cond role
            cond = enc.norm_atom(getattr(instr, "cond", None), role="cond")
            tl = enc.norm_label(instr.true_label)
            fl = enc.norm_label(instr.false_label)
            toks.append(f"BR({cond},{tl},{fl})")

        elif cname == "IRAssign":
            # target/src roles
            tgt = enc.norm_atom(getattr(instr, "target", None), role="lhs")
            src = enc.norm_atom(getattr(instr, "src", None), role="rhs")
            toks.append(f"ASG({tgt},{src})")

        elif cname == "IRBinOp":
            op = _op_norm(getattr(instr, "op", ""))
            dst = enc.norm_atom(getattr(instr, "dest", None), role="dst")
            left = enc.norm_atom(getattr(instr, "left", None), role="lhs")
            right = enc.norm_atom(getattr(instr, "right", None), role="rhs")

            # 交换律规范化（只在 commutative ops）
            if op in COMMUTATIVE_OPS:
                a, b = sorted([left, right])
                left, right = a, b

            toks.append(f"BIN({op},{dst},{left},{right})")

        elif cname == "IRCall":
            callee = enc.norm_callee(getattr(instr, "callee", ""), keep=keep_callee, l1=l1)
            args = list(getattr(instr, "args", []) or [])
            arity = len(args)

            # L2：保留参数“形态序列”，但不引入具体变量名/编号；L1：只保留 arity
            if l1:
                ret = enc.norm_atom(getattr(instr, "dest", None), role="ret")
                toks.append(f"CALL({callee},A{arity},{ret})")
            else:
                ret = enc.norm_atom(getattr(instr, "dest", None), role="ret")
                arg_shapes = ",".join(enc.shape_atom(a) for a in args)
                toks.append(f"CALL({callee},A{arity},{ret},[{arg_shapes}])")

        else:
            toks.append(f"OTHER({cname})")

    return toks


def compute_skeleton_id_v2(ir_instrs: List[Any], *, keep_callee_l2: bool = True) -> Dict[str, str]:
    """
    返回两级 skeleton：
      - skeleton_id_l2：严格
      - skeleton_id_l1：松（用于分层桶兜底）
    """
    t2 = _compute_skeleton_tokens(ir_instrs, keep_callee=keep_callee_l2, l1=False)
    t1 = _compute_skeleton_tokens(ir_instrs, keep_callee=False, l1=True)

    s2 = "|".join(t2).encode("utf-8")
    s1 = "|".join(t1).encode("utf-8")
    return {
        "skeleton_id_l2": hashlib.md5(s2).hexdigest(),
        "skeleton_id_l1": hashlib.md5(s1).hexdigest(),
    }


def ir_to_tokens_v2(ir_instrs: List[Any], *, keep_callee_l2: bool = True) -> Dict[str, List[str]]:
    """
    输出三套序列：
      - ir_tokens_l2：严格归一化（用于主桶）
      - ir_tokens_l1：松归一化（用于兜底桶）
      - ir_tokens_raw：弱归一化（保留更多信息，便于排查/可视化）
    """
    t2 = _compute_skeleton_tokens(ir_instrs, keep_callee=keep_callee_l2, l1=False)
    t1 = _compute_skeleton_tokens(ir_instrs, keep_callee=False, l1=True)

    # raw 还是按你原逻辑，便于定位问题
    raw_seq: List[str] = []
    for ins in ir_instrs:
        cname = ins.__class__.__name__
        if cname == "IRLabel":
            raw_seq.append(f"LABEL({ins.name})")
        elif cname == "IRGoto":
            raw_seq.append(f"GOTO({ins.target_label})")
        elif cname == "IRBranchCond":
            raw_seq.append(f"BR({getattr(ins,'cond',None)},{ins.true_label},{ins.false_label})")
        elif cname == "IRAssign":
            raw_seq.append(f"ASSIGN({getattr(ins,'target',None)},{getattr(ins,'src',None)})")
        elif cname == "IRBinOp":
            raw_seq.append(f"BIN({getattr(ins,'op','')},{getattr(ins,'dest',None)},{getattr(ins,'left',None)},{getattr(ins,'right',None)})")
        elif cname == "IRCall":
            args = list(getattr(ins, "args", []) or [])
            raw_seq.append(f"CALL({getattr(ins,'callee','')},{len(args)},{getattr(ins,'dest',None)},{','.join(args)})")
        else:
            raw_seq.append(f"OTHER({cname})")

    return {
        "ir_tokens_l2": t2,
        "ir_tokens_l1": t1,
        "ir_tokens_raw": raw_seq,
    }


# ============================================================
# Stats (保持你原样)
# ============================================================
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
    n_edges = sum(len(v) for v in cfg.succ.values()) if getattr(cfg, "succ", None) else 0
    back_edges = 0
    if getattr(cfg, "succ", None):
        for i, succs in cfg.succ.items():
            for j in succs:
                if j <= i:
                    back_edges += 1
    n_nodes = len(getattr(cfg, "instrs", []))
    n_exits = len(getattr(cfg, "exits", [])) if getattr(cfg, "exits", None) is not None else 0
    return {
        "n_cfg_nodes": n_nodes,
        "n_cfg_edges": n_edges,
        "n_cfg_backedge": back_edges,
        "n_cfg_exits": n_exits,
    }


def defuse_extra_stats(du) -> Dict[str, int]:
    def_vars = getattr(du, "def_vars", [])
    use_vars = getattr(du, "use_vars", [])

    n_def_sites = sum(len(s) for s in def_vars)
    n_use_sites = sum(len(s) for s in use_vars)

    uniq_all = set()
    for s in def_vars:
        uniq_all |= set(s)
    for s in use_vars:
        uniq_all |= set(s)

    uniq_prog = set()
    for v in uniq_all:
        if v is None:
            continue
        if _is_temp(str(v)):
            continue
        if _is_const(str(v)):
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


# ============================================================
# Pipeline
# ============================================================
def run_pipeline_on_pou(pou: Any, mods: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    IRBuilder = getattr(mods["ir_builder"], "IRBuilder")
    CFGBuilder = getattr(mods["cfg_builder"], "CFGBuilder")
    DefUseAnalyzer = getattr(mods["def_use"], "DefUseAnalyzer")
    PDGBuilder = getattr(mods["pdg_builder"], "PDGBuilder")
    build_program_dependence_graph = getattr(mods["pdg_builder"], "build_program_dependence_graph")

    pou_name = getattr(pou, "name", "<POU>")
    sample_id = hashlib.md5(f"{source_file}::{pou_name}".encode("utf-8")).hexdigest()

    # 1) AST -> IR
    irb = IRBuilder(pou_name=pou_name)
    for s in getattr(pou, "body", []):
        irb.lower_stmt(s)

    ir_instrs = irb.instrs
    if not ir_instrs:
        return {
            "sample_id": sample_id,
            "source_file": source_file,
            "pou_name": pou_name,
            "skeleton_id_l2": None,
            "skeleton_id_l1": None,
            "error": "EMPTY_IR",
        }

    # 2) IR -> CFG
    cfg = CFGBuilder(ir_instrs).build()
    s_cfg = cfg_extra_stats(cfg)

    # 3) CFG -> Def-Use
    du = DefUseAnalyzer(cfg, ir2ast_stmt=getattr(irb, "ir2ast_stmt", None)).analyze()
    s_du = defuse_extra_stats(du)

    # 4) CFG + DU -> PDG
    pdg = PDGBuilder(cfg, du).build()
    _ = build_program_dependence_graph(ir_instrs, pdg, du)
    s_pdg = pdg_extra_stats(pdg)

    # skeleton & tokens（放在 CFG/DU 后面也不冲突，方便你后续扩展“基于DU的角色识别”）
    sk = compute_skeleton_id_v2(ir_instrs, keep_callee_l2=KEEP_CALLEE_L2)
    tok = ir_to_tokens_v2(ir_instrs, keep_callee_l2=KEEP_CALLEE_L2)
    s_ir = ir_stats(ir_instrs)

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
    bucket_z = compute_bucket_z(z_vec, n_ir_call=s_ir["n_ir_call"])

    out = {
        "sample_id": sample_id,
        "source_file": source_file,
        "pou_name": pou_name,
        "bucket_z": bucket_z, 
        **sk,              # skeleton_id_l2 / skeleton_id_l1
        **s_ir,
        **s_cfg,
        **s_du,
        **s_pdg,
        "z_vec": z_vec,
        **tok,             # ir_tokens_l2 / ir_tokens_l1 / ir_tokens_raw
    }
    return out


def iter_st_files(in_dir: str, glob_pat: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if fnmatch.fnmatch(f, glob_pat):
                out.append(os.path.join(root, f))
    return sorted(out)


def main():
    mods = resolve_project_modules()

    files = iter_st_files(IN_DIR, GLOB_PAT)
    if not files:
        raise SystemExit(f"No .st files found in: {IN_DIR}")

    failed_lines: List[str] = []
    ok = 0
    parse_failed = 0
    parse_empty = 0
    empty_ir = 0
    total_pou = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as wf:
        for fp in files:
            try:
                txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
            except Exception as e:
                failed_lines.append(f"{os.path.relpath(fp, IN_DIR)}\tREAD_ERROR: {e}")
                parse_failed += 1
                continue

            try:
                pous = parse_st_to_pous(txt, filename=fp)
            except Exception as e:
                failed_lines.append(f"{os.path.relpath(fp, IN_DIR)}\tPARSE_ERROR: {e}")
                parse_failed += 1
                continue

            if not pous:
                failed_lines.append(f"{os.path.relpath(fp, IN_DIR)}\tPARSE_EMPTY: no POU returned")
                parse_empty += 1
                continue

            for pou in pous:
                total_pou += 1
                try:
                    rec = run_pipeline_on_pou(pou, mods, source_file=fp)
                except Exception as e:
                    failed_lines.append(f"{os.path.relpath(fp, IN_DIR)}\tPIPELINE_ERROR: {e}")
                    parse_failed += 1
                    continue

                if rec.get("error") == "EMPTY_IR":
                    failed_lines.append(f"{os.path.relpath(fp, IN_DIR)}\tEMPTY_IR: no IR instrs")
                    empty_ir += 1
                    continue

                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1

    with open(FAILED_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(failed_lines))

    print("Done.")
    print(f"IN_DIR          : {IN_DIR}")
    print(f"OUT_JSONL       : {OUT_JSONL}")
    print(f"FAILED_TXT      : {FAILED_TXT}")
    print(f"Total .st files : {len(files)}")
    print(f"Total POU found : {total_pou}")
    print(f"POU ok (written): {ok}")
    print(f"PARSE_EMPTY     : {parse_empty}")
    print(f"PARSE_ERROR     : {parse_failed}")
    print(f"EMPTY_IR        : {empty_ir}")


if __name__ == "__main__":
    main()
