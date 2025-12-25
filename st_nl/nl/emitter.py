# st_nl/nl/emitter.py
from dataclasses import dataclass
from typing import List, Optional
from st_nl.ast import nodes as N

@dataclass
class CallSpec:
    callee: str
    pos_args: List[N.Expr]
    named_args: List[N.NamedArg]
    outputs: List[N.Expr]
    loc: N.SourceLocation

def extract_call_spec(stmt: N.Stmt) -> Optional[CallSpec]:
    if isinstance(stmt, N.Assignment) and isinstance(stmt.value, N.CallExpr):
        outs: List[N.Expr]

        # 调试：确认提取的语句
        print(f"[DEBUG] Found Assignment with CallExpr: {stmt}")

        if isinstance(stmt.target, N.TupleExpr):
            outs = stmt.target.items
        else:
            outs = [stmt.target]

        return CallSpec(
            callee=stmt.value.func,
            pos_args=stmt.value.pos_args,
            named_args=stmt.value.named_args,
            outputs=outs,
            loc=stmt.loc,
        )

    if isinstance(stmt, N.CallStmt):
        # 调试：确认提取的函数块调用语句
        print(f"[DEBUG] Found CallStmt: {stmt}")

        return CallSpec(
            callee=stmt.fb_name,
            pos_args=stmt.pos_args,
            named_args=stmt.named_args,
            outputs=[],
            loc=stmt.loc,
        )

    return None
