# st_nl/ir/normalize.py
from typing import List, Any
from st_nl.ast import nodes as A
from st_nl.ir.nodes import CallIR, AssignIR, IfIR, ArgIR, OutputIR

def _as_outputs(lhs: A.Expr) -> List[OutputIR]:
    if isinstance(lhs, A.TupleExpr):
        outs = []
        for item in lhs.items:
            outs.append(OutputIR(target=item, name=getattr(item, "name", None)))
        return outs
    return [OutputIR(target=lhs, name=getattr(lhs, "name", None))]

def normalize_call(stmt: A.Stmt) -> CallIR | None:
    # 1) OUT := Func(...)
    if isinstance(stmt, A.Assignment) and isinstance(stmt.value, A.CallExpr):
        inputs: List[ArgIR] = []
        for e in stmt.value.pos_args:
            inputs.append(ArgIR(name=None, expr=e))
        for na in stmt.value.named_args:
            inputs.append(ArgIR(name=na.name, expr=na.value))

        return CallIR(
            callee=stmt.value.func,
            call_kind="function",
            inputs=inputs,
            outputs=_as_outputs(stmt.target),
            loc=stmt.loc,
        )

    # 2) FB(...) 作为语句（无显式输出）
    if isinstance(stmt, A.CallStmt):
        inputs: List[ArgIR] = []
        for e in stmt.pos_args:
            inputs.append(ArgIR(name=None, expr=e))
        for na in stmt.named_args:
            inputs.append(ArgIR(name=na.name, expr=na.value))

        return CallIR(
            callee=stmt.fb_name,
            call_kind="fb",
            inputs=inputs,
            outputs=[],
            loc=stmt.loc,
        )

    return None

def normalize_stmt(stmt: A.Stmt) -> Any:
    # Call 优先
    c = normalize_call(stmt)
    if c is not None:
        return c

    # 普通赋值
    if isinstance(stmt, A.Assignment):
        return AssignIR(target=stmt.target, value=stmt.value, loc=stmt.loc)

    # IF 递归归一
    if isinstance(stmt, A.IfStmt):
        then_ir = [normalize_stmt(s) for s in stmt.then_body]
        elif_ir = [(cond, [normalize_stmt(s) for s in body]) for cond, body in stmt.elif_branches]
        else_ir = [normalize_stmt(s) for s in stmt.else_body]
        return IfIR(
            cond=stmt.cond,
            then_body=then_ir,
            elif_branches=elif_ir,
            else_body=else_ir,
            loc=stmt.loc,
        )

    # 其他 For/While/Case/Repeat 你可以按同样方式补齐
    return stmt  # 兜底：暂时返回原 AST，保证不崩
