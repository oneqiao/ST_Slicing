# test_nl.main.py

from pathlib import Path
from antlr4 import CommonTokenStream, InputStream
from antlr4.tree.Trees import Trees

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

        # 兼容旧字段：如果历史上你用过 s.args（List[Expr|NamedArg]）
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

        # 先打印位置参数
        for a in pos:
            dump_expr(a, indent + 2)

        # 再打印命名参数
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
        # 先简单打印节点类型，后面需要再细化
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


def main():
    # 你要验证的 .st 文件名（放在 test_nl/ 目录下）
    filename = "sample4.st"

    # 1) 读取文件
    st_code = read_st_file(filename)

    print("=" * 80)
    print(f"[1] RAW ST FILE: {filename}")
    print("=" * 80)
    print(st_code)

    # 2) 预处理
    processed = preprocess_st(st_code)

    print("\n" + "=" * 80)
    print("[2] PREPROCESSED ST CODE")
    print("=" * 80)
    print(processed)

    # 3) IL 剥离检查（可选）
    # 如果你 preprocess 里用 placeholder，就可以检测它；
    # 如果你选择完全删除 IL，下面这段仍然可用（检查 IL 指令是否还存在）
    il_keywords = ["LD ", "ADD ", "ST "]
    if any(k in processed for k in il_keywords):
        print("\n[WARN] Possible IL-like lines still present after preprocessing.")
    else:
        print("\n[OK] IL-like lines removed (or not present).")

    # 4) 解析并打印 parse tree
    print("\n" + "=" * 80)
    print("[3] PARSE TREE (ANTLR)")
    print("=" * 80)
    try:
        tree, parser = parse_st_code_debug(processed)
        tree_str = Trees.toStringTree(tree, None, parser)
        #print(tree_str)
        #print("\n[OK] Parsed successfully with rule: start")
    except Exception as e:
        print("\n[FAIL] Parsing raised exception:")
        print(repr(e))

    # 5) AST 构建
    builder = ASTBuilder(filename=filename)
    pous = builder.visit(tree)   # tree 是 parser.start() 的结果
    print("\n" + "=" * 80)
    print("[4] AST BUILDER OUTPUT")
    print("=" * 80)
    for pou in pous:
        dump_pou(pou)


if __name__ == "__main__":
    main()