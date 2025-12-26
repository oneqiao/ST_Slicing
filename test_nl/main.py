# test_nl.main.py

from pathlib import Path
from antlr4 import CommonTokenStream, InputStream
from antlr4.tree.Trees import Trees
from typing import Iterable

# 1) 按你的项目实际路径修改这两个 import
#    - preprocess_st: 你写的预处理函数
#    - IEC61131Lexer / IEC61131Parser: generated 目录生成的类
from st_nl.parser.preprocess import preprocess_st
from st_nl.generated.IEC61131Lexer import IEC61131Lexer
from st_nl.generated.IEC61131Parser import IEC61131Parser

from st_nl.ast.builder import ASTBuilder  # 按你的实际路径改
from st_nl.ast.nodes import (
    ProgramDecl, FBDecl,
    VarDecl,
    Assignment, IfStmt, ForStmt, WhileStmt, RepeatStmt, CallStmt, CaseStmt,
    VarRef, ArrayAccess, FieldAccess, CallExpr, Literal, BinOp, UnaryOp,
)
from st_nl.ir.normalize import normalize_stmt
from st_nl.ir.nodes import CallIR
from st_nl.ast import nodes as N
from st_nl.nl.ir import stmt_to_callir

from st_nl.nl.generate import emit_pou, NLCfg, DocEntry, NLLevel

import logging
logging.basicConfig(level=logging.DEBUG)

docs = {
  "USHLW": DocEntry(summary="Logical left shift: OUT1 = IN1 << IN2."),
  "FB_Multi": DocEntry(summary="FB with one input and multiple outputs (OUT1, OUT2)."),
  "FB_Test": DocEntry(summary="FB is Function."),
}

def dump_callir(ir):
    print(f"[CallIR] kind={ir.call_kind} callee={ir.callee} @ {ir.loc.file}:{ir.loc.line}")
    if ir.inputs:
        for inp in ir.inputs:
            nm = inp.name if inp.name is not None else "<pos>"
            print(f"   - in : {nm} dir={inp.direction} expr={type(inp.expr).__name__}:{getattr(inp.expr,'name',getattr(inp.expr,'value',''))}")
    if ir.outputs:
        for out in ir.outputs:
            print(f"   - out: {type(out.target).__name__}:{getattr(out.target,'name',out.target)}")
    else:
        print("   - out: (none)")

def dump_expr(e, indent=0):
    pad = " " * indent
    if e is None:
        print(pad + "Expr: <None>")
        return
    t = type(e).__name__
    if t == "VarRef":
        print(pad + f"VarRef(name={e.name})")
    elif t == "Literal":
        print(pad + f"Literal(value={e.value}, type={e.type})")
    elif t == "ArrayAccess":
        print(pad + "ArrayAccess(")
        dump_expr(e.base, indent + 2)
        dump_expr(e.index, indent + 2)
        print(pad + ")")
    elif t == "FieldAccess":
        print(pad + f"FieldAccess(field={e.field})(")
        dump_expr(e.base, indent + 2)
        print(pad + ")")
    elif t == "CallExpr":
        argc = len(e.pos_args) + len(e.named_args)
        print(pad + f"CallExpr(func={e.func}, argc={argc})")
        for a in e.pos_args:
            dump_expr(a, indent + 2)
        for na in e.named_args:
            print(pad + f"  NamedArg({na.name}=")
            dump_expr(na.value, indent + 4)
            print(pad + "  )")
    elif t == "TupleExpr":
        print(pad + "TupleExpr(")
        for item in e.items:
            dump_expr(item, indent + 2)
        print(pad + ")")
    elif t == "UnaryOp":
        print(pad + f"UnaryOp(op={e.op})(")
        dump_expr(e.operand, indent + 2)
        print(pad + ")")
    elif t == "BinOp":
        print(pad + f"BinOp(op={e.op})(")
        dump_expr(e.left, indent + 2)
        dump_expr(e.right, indent + 2)
        print(pad + ")")
    else:
        # fallback：避免因为某节点没覆盖导致崩
        print(pad + f"{t}: {getattr(e, 'name', '') or str(e)}")


def dump_stmt(s, indent=0):
    pad = " " * indent
    if s is None:
        print(pad + "Stmt: <None>")
        return
    t = type(s).__name__

    if t == "Assignment":
        print(pad + "Assignment(")
        print(pad + "  target=")
        dump_expr(s.target, indent + 4)
        print(pad + "  value=")
        dump_expr(s.value, indent + 4)
        print(pad + ")")
    elif t == "CallStmt":
        pos = getattr(s, "pos_args", None)
        named = getattr(s, "named_args", None)

        if pos is None and named is None and hasattr(s, "args"):
            pos = []
            named = []
            for a in s.args:
                if type(a).__name__ == "NamedArg":
                    named.append(a)
                else:
                    pos.append(a)

        pos = pos or []
        named = named or []

        argc = len(pos) + len(named)
        print(pad + f"CallStmt(name={s.fb_name}, argc={argc})")
        for a in pos:
            dump_expr(a, indent + 2)

        for na in named:
            print(pad + f"  NamedArg({na.name}=")
            dump_expr(na.value, indent + 4)
            print(pad + "  )")
    elif t == "IfStmt":
        print(pad + "IfStmt(cond=")
        dump_expr(s.cond, indent + 2)
        print(pad + "then:")
        for x in s.then_body:
            dump_stmt(x, indent + 2)
        if getattr(s, "elif_branches", None):
            print(pad + "elif:")
            for (c, body) in s.elif_branches:
                print(pad + "  elif_cond=")
                dump_expr(c, indent + 4)
                for x in body:
                    dump_stmt(x, indent + 4)
        if getattr(s, "else_body", None):
            print(pad + "else:")
            for x in s.else_body:
                dump_stmt(x, indent + 2)
        print(pad + ")")
    elif t in ("ForStmt", "WhileStmt", "RepeatStmt", "CaseStmt"):
        print(pad + f"{t}(...)")
    else:
        print(pad + f"{t}")

def dump_pou(pou):
    print("=" * 80)
    print(f"POU: {type(pou).__name__} name={pou.name}")
    print(f"Vars: {len(pou.vars)}  BodyStmts: {len(pou.body)}")
    print("- Vars sample (first 5) -")
    for v in pou.vars[:5]:
        print(f"  {v.storage} {v.name} : {v.type}")
    print("- Body AST (first 20 stmts) -")
    for s in pou.body[:20]:
        dump_stmt(s, 2)

def read_st_file(filename: str, encoding: str = "utf-8") -> str:
    """
    从 test_nl/main.py 同目录读取 .st 文件
    """
    base_dir = Path(__file__).resolve().parent  # test_nl 目录
    st_path = base_dir / filename
    if not st_path.exists():
        raise FileNotFoundError(f"ST file not found: {st_path}")
    return st_path.read_text(encoding=encoding)


def parse_st_code_debug(code: str):
    """
    带调试输出的 parse：返回 tree 和 parser
    """
    input_stream = InputStream(code)
    lexer = IEC61131Lexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = IEC61131Parser(token_stream)

    tree = parser.start()  # 入口规则 start
    return tree, parser

def walk_stmts(stmts):
    for s in stmts:
        yield s
        if isinstance(s, N.IfStmt):
            yield from walk_stmts(s.then_body)
            for _, b in (s.elif_branches or []):
                yield from walk_stmts(b)
            yield from walk_stmts(s.else_body or [])
        elif isinstance(s, N.ForStmt):
            yield from walk_stmts(s.body)
        elif isinstance(s, N.WhileStmt):
            yield from walk_stmts(s.body)
        elif isinstance(s, N.RepeatStmt):
            yield from walk_stmts(s.body)
        elif isinstance(s, N.CaseStmt):
            for e in s.entries:
                yield from walk_stmts(e.body)
            yield from walk_stmts(s.else_body or [])

def test_normalize(pou):
    calls = 0
    for s in walk_stmts(pou.body):
        ir = normalize_stmt(s)
        if isinstance(ir, CallIR):
            calls += 1
            outs = ", ".join(o.name or "<expr>" for o in ir.outputs) or "<no outs>"
            ins  = ", ".join((a.name + ":=" if a.name else "") + type(a.expr).__name__ for a in ir.inputs)
            print(f"[CallIR] {outs} <- {ir.callee}({ins}) @ line {ir.loc.line}")
    print(f"[SUMMARY] CallIR count = {calls}")

def main():
    filename = "sample2.st"

    # -----------------------------
    # 1) 读取 + 预处理
    # -----------------------------
    st_code = read_st_file(filename)
    processed = preprocess_st(st_code)

    # -----------------------------
    # 2) 解析 + AST 构建
    # -----------------------------
    tree, parser = parse_st_code_debug(processed)
    builder = ASTBuilder(filename=filename)
    pous = builder.visit(tree)

    # -----------------------------
    # 3) NL 配置 + 指令文档
    # -----------------------------
    cfg = NLCfg(nl_level=NLLevel.FINE, enable_enriched=True)

    # 示例指令文档（你后续会替换为真实文档解析器）
    docs = {
        "USHLW": DocEntry(summary="Logical left shift: OUT = IN << N"),
        # "FB_Multi": DocEntry(...)  # FB 可先不加
    }

    # -----------------------------
    # 4) 对每个 POU 做三件事：
    #    A. 打印 POU 名
    #    B. 验证 CallIR（只统计一次）
    #    C. 生成 NL
    # -----------------------------
    for pou in pous:
        print("=" * 80)
        print(f"POU: {pou.name}")

        # ---- A) CallIR 验证（Normalize 是否正确）----
        call_count = 0
        for stmt in walk_stmts(pou.body):
            cir = stmt_to_callir(stmt)
            if cir is None:
                continue
            call_count += 1
            dump_callir(cir)

        print(f"[SUMMARY] total calls extracted: {call_count}")

        # ---- B) NL 生成（统一风格 + 可控粒度）----
        print("\n--- Generated NL ---")
        lines = emit_pou(pou, cfg, docs)
        print("\n".join(lines))
        print()

if __name__ == "__main__":
    main()