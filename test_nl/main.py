from pathlib import Path
from antlr4 import CommonTokenStream, InputStream

from st_nl.parser.preprocess import preprocess_st
from st_nl.generated.IEC61131Lexer import IEC61131Lexer
from st_nl.generated.IEC61131Parser import IEC61131Parser
from st_nl.ast.builder import ASTBuilder
from st_nl.ast import nodes as N
from st_nl.nl.ir import stmt_to_callir
from st_nl.nl.render import render_expr  
from st_nl.nl.generate import emit_pou, NLCfg, NLLevel
from st_nl.rules.semantic_catalog import SemanticCatalog, norm_name
from collections import Counter
from pathlib import Path

def read_st_file(filename: str, encoding: str = "utf-8") -> str:
    base_dir = Path(__file__).resolve().parent
    st_path = base_dir / filename
    if not st_path.exists():
        raise FileNotFoundError(f"ST file not found: {st_path}")
    return st_path.read_text(encoding=encoding)


def parse_st_code_debug(code: str):
    input_stream = InputStream(code)
    lexer = IEC61131Lexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = IEC61131Parser(token_stream)
    tree = parser.start()  # 入口规则
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


def dump_callir(ir):
    print(f"[CallIR] kind={ir.call_kind} callee={ir.callee} @ {ir.loc.file}:{ir.loc.line}")
    if ir.inputs:
        for inp in ir.inputs:
            nm = inp.name if inp.name is not None else "<pos>"
            print(f"   - in : {nm} dir={getattr(inp,'direction','?')} expr={type(inp.expr).__name__}:{getattr(inp.expr,'name',getattr(inp.expr,'value',''))}")
    if ir.outputs:
        for out in ir.outputs:
            print(f"   - out: {type(out.target).__name__}:{getattr(out.target,'name',out.target)}")
    else:
        print("   - out: (none)")


def build_bind_maps(cir, cfg):
    outs = [render_expr(o.target, cfg.render) for o in (cir.outputs or [])]

    pos_ins = [
        render_expr(i.expr, cfg.render)
        for i in (cir.inputs or [])
        if i.name is None
    ]

    named_ins = {}
    named_outs = {}
    for i in (cir.inputs or []):
        if i.name is None:
            continue
        key = norm_name(i.name)
        val = render_expr(i.expr, cfg.render)
        d = getattr(i, "direction", "unknown")
        if key.startswith("IN"):
            named_ins[key] = val
        if key.startswith("OUT") and d in ("out", "inout"):
            named_outs[key] = val

    return outs, pos_ins, named_ins, named_outs

def _find_project_root(start: Path) -> Path:
    """
    从 start 开始向上找工程根目录：以“包含 st_nl 目录”为判定条件。
    找不到就回退到 start 的父目录。
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "st_nl").is_dir():
            return p
    return start.parent

def main():
    filename = "sample3.st"

    # 1) 读取 + 预处理
    st_code = read_st_file(filename)
    processed = preprocess_st(st_code)

    # 2) 解析 + AST
    tree, _parser = parse_st_code_debug(processed)
    builder = ASTBuilder(filename=filename)
    pous = builder.visit(tree)

    # 3) 相对工程根目录加载语义表（不写死绝对路径）
    # test_nl/main.py 的 __file__ -> 向上找到包含 st_nl/ 的目录
    project_root = _find_project_root(Path(__file__).parent)
    yaml_path = project_root / "st_nl" / "rules" / "function.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"semantic catalog yaml not found: {yaml_path}\n"
            f"project_root resolved to: {project_root}\n"
            f"current working dir: {Path.cwd()}"
        )

    catalog = SemanticCatalog.load_yaml(str(yaml_path))

    # 4) lookup 自检（可保留）
    for name in ["UEQB", "UEQW", "UGEB", "USHLW", "FB_MULTI"]:
        e = catalog.lookup(name)
        if e is None:
            print(f"[CATALOG] {name}: NOT FOUND")
        else:
            print(f"[CATALOG] {name}: FOUND kind={e.kind} template={e.semantics_template}")

    # 5) NL 配置
    cfg = NLCfg(nl_level=NLLevel.FINE, enable_enriched=True)

    # 6) 语义命中率统计（全局）
    total_calls = 0
    semantic_hits = 0
    missing_counter = Counter()   # callee -> miss count（包括 entry 不存在或无 template）

    for pou in pous:
        print("=" * 80)
        print(f"POU: {pou.name}")

        # 也可以同时统计每个 POU 的命中率（可选）
        pou_total = 0
        pou_hits = 0

        # ---- A) CallIR 验证 & 命中率统计（不再打印 [BIND]）----
        for stmt in walk_stmts(pou.body):
            cir = stmt_to_callir(stmt)
            if cir is None:
                continue

            total_calls += 1
            pou_total += 1

            dump_callir(cir)

            ent = catalog.lookup(cir.callee)
            if ent is not None and getattr(ent, "semantics_template", None):
                semantic_hits += 1
                pou_hits += 1
            else:
                missing_counter[cir.callee] += 1

        print(f"[SUMMARY] total calls extracted: {pou_total}")
        if pou_total > 0:
            pou_rate = pou_hits / pou_total
            print(f"[SUMMARY] semantic hit: {pou_hits}/{pou_total} = {pou_rate:.2%}")

        # ---- B) NL 输出（由 generate.py 的 maybe_enrich 自动追加 Semantics）----
        print("\n--- Generated NL ---")
        lines = emit_pou(pou, cfg, catalog)
        print("\n".join(lines))
        print()

    # 7) 全局命中率报告 + 未命中 TopK
    print("=" * 80)
    print(f"[GLOBAL] semantic hit: {semantic_hits}/{total_calls} = {(semantic_hits/total_calls if total_calls else 0):.2%}")

    if missing_counter:
        print("[GLOBAL] missing semantics callee Top10:")
        for callee, cnt in missing_counter.most_common(10):
            print(f"  - {callee}: {cnt}")


if __name__ == "__main__":
    main()
