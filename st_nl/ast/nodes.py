from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple,  Union


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    column: int = 0


# --------------------
# Expressions
# --------------------
class Expr:
    loc: SourceLocation


@dataclass(eq=False)
class VarRef(Expr):
    name: str
    loc: SourceLocation


@dataclass(eq=False)
class ArrayAccess(Expr):
    base: Expr
    index: Expr
    loc: SourceLocation


@dataclass(eq=False)
class FieldAccess(Expr):
    base: Expr
    field: str
    loc: SourceLocation


@dataclass(eq=False)
class Literal(Expr):
    value: Any
    type: str
    loc: SourceLocation


@dataclass(eq=False)
class UnaryOp(Expr):
    op: str
    operand: Expr
    loc: SourceLocation


@dataclass(eq=False)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    loc: SourceLocation


@dataclass(eq=False)
class CallExpr(Expr):
    func: str
    pos_args: List["Expr"] = field(default_factory=list)
    named_args: List["NamedArg"] = field(default_factory=list)
    loc: SourceLocation = None

# --------------------
# Statements
# --------------------
class Stmt:
    loc: SourceLocation


@dataclass(eq=False)
class Assignment(Stmt):
    target: Expr
    value: Expr
    op: str = ":="
    loc: SourceLocation = None


@dataclass(eq=False)
class CallStmt(Stmt):
    fb_name: str
    pos_args: List["Expr"] = field(default_factory=list)
    named_args: List["NamedArg"] = field(default_factory=list)
    loc: SourceLocation = None


@dataclass(eq=False)
class IfStmt(Stmt):
    cond: Expr
    then_body: List[Stmt] = field(default_factory=list)
    elif_branches: List[Tuple[Expr, List[Stmt]]] = field(default_factory=list)
    else_body: List[Stmt] = field(default_factory=list)
    loc: SourceLocation = None


@dataclass(eq=False)
class ForStmt(Stmt):
    var: str
    start: Expr
    end: Expr
    step: Optional[Expr]
    body: List[Stmt] = field(default_factory=list)
    loc: SourceLocation = None


@dataclass(eq=False)
class WhileStmt(Stmt):
    cond: Expr
    body: List[Stmt] = field(default_factory=list)
    loc: SourceLocation = None


@dataclass(eq=False)
class RepeatStmt(Stmt):
    body: List[Stmt] = field(default_factory=list)
    until: Expr = None
    loc: SourceLocation = None


@dataclass(eq=False)
class CaseCond:
    text: str
    loc: SourceLocation


@dataclass(eq=False)
class CaseEntry:
    conds: List[CaseCond]
    body: List[Stmt]
    loc: SourceLocation


@dataclass(eq=False)
class CaseStmt(Stmt):
    cond: Expr
    entries: List[CaseEntry] = field(default_factory=list)
    else_body: List[Stmt] = field(default_factory=list)
    loc: SourceLocation = None


# --------------------
# Declarations
# --------------------
@dataclass(eq=False)
class VarDecl:
    name: str
    type: str
    storage: str
    init_expr: Optional[Expr]
    loc: SourceLocation

@dataclass(eq=False)
class ProgramDecl:
    name: str
    vars: List[VarDecl] = field(default_factory=list)
    body: List[Stmt] = field(default_factory=list)
    loc: SourceLocation = None


@dataclass(eq=False)
class FBDecl:
    name: str
    vars: List[VarDecl] = field(default_factory=list)
    body: List[Stmt] = field(default_factory=list)
    loc: SourceLocation = None

@dataclass(eq=False)
class ContinueStmt(Stmt):
    loc: SourceLocation

@dataclass(eq=False)
class NamedArg:
    name: str
    value: "Expr"
    loc: SourceLocation

