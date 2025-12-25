# st_nl/ast/builder.py
from __future__ import annotations

from typing import List, Optional, Union, Tuple, Any

from antlr4 import ParserRuleContext, TerminalNode

from ..generated.IEC61131ParserVisitor import IEC61131ParserVisitor
from ..generated.IEC61131Parser import IEC61131Parser

from .nodes import (
    SourceLocation,
    ProgramDecl,
    FBDecl,
    VarDecl,
    Stmt,
    Expr,
    Assignment,
    IfStmt,
    ForStmt,
    CallStmt,
    NamedArg,
    WhileStmt,
    RepeatStmt,
    CaseStmt,
    CaseEntry,
    CaseCond,
    VarRef,
    Literal,
    UnaryOp,
    BinOp,
    CallExpr,
    ArrayAccess,
    FieldAccess,
    ContinueStmt,
    TupleExpr,
)

from st_nl.ast import nodes as N

POU = Union[ProgramDecl, FBDecl]


class ASTBuilder(IEC61131ParserVisitor):
    """
    IEC 61131-3 ST parse tree -> simplified AST.

    目标：稳定产出 AST（可用于 AST->NL），对不支持的结构做保底处理。
    """

    def __init__(self, filename: str = "<memory>"):
        self.filename = filename

    # --------------------
    # location / text helpers
    # --------------------
    def _loc(self, ctx: ParserRuleContext) -> SourceLocation:
        tok = ctx.start
        return SourceLocation(
            file=self.filename,
            line=getattr(tok, "line", 0),
            column=getattr(tok, "column", 0),
        )

    def _loc_from_ctx(self, ctx: ParserRuleContext) -> N.SourceLocation:
        tok = ctx.start
        return N.SourceLocation(file=self.filename, line=getattr(tok, "line", 0), column=getattr(tok, "column", 0))

    def _as_text(self, obj: Any) -> Optional[str]:
        """
        统一把“可能是 TerminalNode / Context / Token / CommonToken / 方法getter”等转换为字符串。
        - TerminalNode / Context: getText()
        - Token / CommonToken: .text
        - getter method: obj()
        """
        if obj is None:
            return None

        # ANTLR Python 有时会把 label 生成成方法：id_()
        if callable(obj):
            try:
                obj = obj()
            except TypeError:
                pass

        if obj is None:
            return None

        if hasattr(obj, "getText"):
            try:
                return obj.getText()
            except Exception:
                pass

        txt = getattr(obj, "text", None)
        if txt is not None:
            return txt

        try:
            return str(obj)
        except Exception:
            return None

    # --------------------
    # invocation / callee helpers
    # --------------------
    def _get_invocation_id_sv(self, inv_ctx: IEC61131Parser.InvocationContext) -> IEC61131Parser.Symbolic_variableContext:
        """
        invocation: id=symbolic_variable '(' ... ')'
        Python target 对 label 的生成存在差异：可能是 inv_ctx.id_() / inv_ctx.id / inv_ctx.id_。
        这里统一返回 Symbolic_variableContext。
        """
        for attr in ("id_", "id"):
            getter_or_value = getattr(inv_ctx, attr, None)
            if getter_or_value is None:
                continue
            sv = getter_or_value() if callable(getter_or_value) else getter_or_value
            if sv is not None:
                return sv

        # 兜底：从 children 里找第一个 Symbolic_variableContext
        for ch in getattr(inv_ctx, "children", []) or []:
            if isinstance(ch, IEC61131Parser.Symbolic_variableContext):
                return ch

        raise AttributeError("InvocationContext has no id_/id symbolic_variable")

    def _callee_from_symbolic_variable(self, sv_ctx: IEC61131Parser.Symbolic_variableContext) -> str:
        """
        symbolic_variable : a=variable_names ... (DOT other=symbolic_variable)? ;
        生成调用名字符串：a.b.c
        """
        parts: List[str] = []
        cur = sv_ctx
        while cur is not None:
            a = getattr(cur, "a", None)
            a_text = self._as_text(a)
            if a_text:
                parts.append(a_text)

            nxt = getattr(cur, "other", None)
            cur = nxt() if callable(nxt) else nxt

        return ".".join(parts)

    def _get_invocation_name(self, inv_ctx: IEC61131Parser.InvocationContext) -> str:
        sv = self._get_invocation_id_sv(inv_ctx)
        return self._callee_from_symbolic_variable(sv)

    def _parse_invocation_args_in_order(self, inv_ctx: IEC61131Parser.InvocationContext) -> Tuple[List[Expr], List[NamedArg]]:
        """
        保持参数出现顺序（expression / param_assignment 混排）。
        返回: (pos_args, named_args)
        """
        pos_args: List[Expr] = []
        named_args: List[NamedArg] = []

        for ch in inv_ctx.getChildren():
            if isinstance(ch, TerminalNode):
                continue

            # expression: 直接位置参数
            if isinstance(ch, IEC61131Parser.ExpressionContext):
                e = self.visit(ch)
                if isinstance(e, Expr):
                    pos_args.append(e)
                continue

            # param_assignment: 可能是 NamedArg 或 Expr（位置参数写法）
            if isinstance(ch, IEC61131Parser.Param_assignmentContext):
                v = self.visit(ch)
                if isinstance(v, N.NamedArg):
                    named_args.append(v)
                elif isinstance(v, Expr):
                    pos_args.append(v)
                continue

        return pos_args, named_args

    # --------------------
    # top level
    # --------------------
    def visitStart(self, ctx: IEC61131Parser.StartContext) -> List[POU]:
        pous: List[POU] = []
        for elem in ctx.library_element_declaration():
            r = self.visit(elem)
            if r is None:
                continue
            if isinstance(r, (ProgramDecl, FBDecl)):
                pous.append(r)
            elif isinstance(r, list):
                for x in r:
                    if isinstance(x, (ProgramDecl, FBDecl)):
                        pous.append(x)
        return pous

    def visitLibrary_element_declaration(self, ctx: IEC61131Parser.Library_element_declarationContext):
        if ctx.program_declaration():
            return self.visit(ctx.program_declaration())
        if ctx.function_block_declaration():
            return self.visit(ctx.function_block_declaration())
        return None

    # --------------------
    # PROGRAM / FB
    # --------------------
    def _extract_body(self, body_ctx: IEC61131Parser.BodyContext) -> List[Stmt]:
        # 仅取 statement_list（你已确认不考虑 IL）
        if body_ctx and body_ctx.statement_list():
            return self.visit(body_ctx.statement_list())
        return []

    def visitProgram_declaration(self, ctx: IEC61131Parser.Program_declarationContext) -> ProgramDecl:
        name = ctx.identifier.text if getattr(ctx, "identifier", None) is not None else "PROGRAM"
        vars_ = self.visit(ctx.var_decls()) if ctx.var_decls() else []
        body = self._extract_body(ctx.body()) if ctx.body() else []
        return ProgramDecl(name=name, vars=vars_, body=body, loc=self._loc(ctx))

    def visitFunction_block_declaration(self, ctx: IEC61131Parser.Function_block_declarationContext) -> FBDecl:
        if getattr(ctx, "identifier", None) is not None:
            name = ctx.identifier.text
        elif ctx.IDENTIFIER():
            name = ctx.IDENTIFIER().getText()
        else:
            name = "FUNCTION_BLOCK"

        vars_ = self.visit(ctx.var_decls()) if ctx.var_decls() else []
        body = self._extract_body(ctx.body()) if ctx.body() else []
        return FBDecl(name=name, vars=vars_, body=body, loc=self._loc(ctx))

    # --------------------
    # VAR declarations
    # --------------------
    def visitVar_decls(self, ctx: IEC61131Parser.Var_declsContext) -> List[VarDecl]:
        out: List[VarDecl] = []
        for vd in ctx.var_decl():
            ds = self.visit(vd)
            if ds:
                out.extend(ds)
        return out

    def visitVar_decl(self, ctx: IEC61131Parser.Var_declContext) -> List[VarDecl]:
        storage = "VAR"
        vk = ctx.variable_keyword()
        if vk is not None and vk.getChildCount() > 0:
            storage = vk.getChild(0).getText()

        inner = ctx.var_decl_inner()
        if inner is None:
            return []

        decls: List[VarDecl] = []
        id_lists = list(inner.identifier_list())
        type_decls = list(inner.type_declaration())
        n = min(len(id_lists), len(type_decls))

        for i in range(n):
            id_list_ctx = id_lists[i]
            type_ctx = type_decls[i]
            type_str = type_ctx.getText()
            for name_ctx in id_list_ctx.variable_names():
                decls.append(
                    VarDecl(
                        name=name_ctx.getText(),
                        type=type_str,
                        storage=storage,
                        init_expr=None,
                        loc=self._loc(ctx),
                    )
                )
        return decls

    # --------------------
    # statements
    # --------------------
    def visitStatement_list(self, ctx: IEC61131Parser.Statement_listContext) -> List[Stmt]:
        out: List[Stmt] = []
        for sctx in ctx.statement():
            s = self.visit(sctx)
            if isinstance(s, Stmt):
                out.append(s)
        return out

    def visitStatement(self, ctx: IEC61131Parser.StatementContext) -> Optional[Stmt]:
        if ctx.assignment_statement():
            return self.visit(ctx.assignment_statement())
        if ctx.invocation_statement():
            return self.visit(ctx.invocation_statement())
        if ctx.if_statement():
            return self.visit(ctx.if_statement())
        if ctx.case_statement():
            return self.visit(ctx.case_statement())
        if ctx.for_statement():
            return self.visit(ctx.for_statement())
        if ctx.while_statement():
            return self.visit(ctx.while_statement())
        if ctx.repeat_statement():
            return self.visit(ctx.repeat_statement())
        if ctx.continue_statement():
            return ContinueStmt(loc=self._loc(ctx))
        return None

    def visitAssignment_statement(self, ctx: IEC61131Parser.Assignment_statementContext) -> Assignment:
        value = self.visit(ctx.right)
        op = ctx.op.text if hasattr(ctx, "op") and ctx.op is not None else ":="
        loc = self._loc(ctx)

        left_text = ctx.left.getText()

        if left_text.startswith("(") and left_text.endswith(")"):
            items: List[Expr] = []

            if hasattr(ctx.left, "variable"):
                for vctx in ctx.left.variable():
                    items.append(self.visit(vctx))
            else:
                for ch in ctx.left.getChildren():
                    t = ch.getText()
                    if t in ("(", ")", ","):
                        continue
                    try:
                        items.append(self.visit(ch))
                    except Exception:
                        items.append(VarRef(name=t, loc=loc))

            target = TupleExpr(items=items, loc=loc)
        else:
            target = self.visit(ctx.left)

        return Assignment(target=target, value=value, op=op, loc=loc)

    def visitInvocation_statement(self, ctx: IEC61131Parser.Invocation_statementContext):
        """
        作为语句出现：FB 风格调用 -> CallStmt
        """
        inv = ctx.invocation()
        callee = self._get_invocation_name(inv)
        loc = self._loc_from_ctx(inv)

        pos_args, named_args = self._parse_invocation_args_in_order(inv)
        return N.CallStmt(fb_name=callee, pos_args=pos_args, named_args=named_args, loc=loc)

    # --------------------
    # invocation / param_assignment (expressions)
    # --------------------
    def visitInvocation(self, ctx: IEC61131Parser.InvocationContext):
        """
        作为表达式出现：函数式调用 -> CallExpr
        """
        callee = self._get_invocation_name(ctx)
        loc = self._loc_from_ctx(ctx)

        pos_args, named_args = self._parse_invocation_args_in_order(ctx)
        return N.CallExpr(func=callee, pos_args=pos_args, named_args=named_args, loc=loc)

    def visitParam_assignment(self, ctx: IEC61131Parser.Param_assignmentContext):
        """
        param_assignment:
          1) id=IDENTIFIER ARROW_RIGHT v=variable        -> OUT => var   (direction="out")
          2) (id=IDENTIFIER ASSIGN)? expression          -> IN := expr   (direction="in") 或纯 expr(位置参数)
        返回：
          - NamedArg（命名参数）
          - Expr（位置参数）
        """
        loc = self._loc_from_ctx(ctx)

        # 统一取 id token：可能叫 id_ 或 id（且可能是 CommonToken）
        id_tok = getattr(ctx, "id_", None)
        if id_tok is None:
            id_tok = getattr(ctx, "id", None)
        id_name = self._as_text(id_tok)

        # 1) OUT => var
        if ctx.ARROW_RIGHT() is not None:
            name = id_name if id_name is not None else "/*OUT*/"
            value = self.visit(ctx.v)
            return N.NamedArg(name=name, value=value, loc=loc, direction="out")

        # 2) (id :=)? expression
        expr = self.visit(ctx.expression())

        # 纯 expression：位置参数
        if id_name is None:
            return expr

        # id := expression：命名输入参数
        return N.NamedArg(name=id_name, value=expr, loc=loc, direction="in")

    # --------------------
    # IF / CASE / FOR / WHILE / REPEAT
    # --------------------
    def visitIf_statement(self, ctx: IEC61131Parser.If_statementContext) -> IfStmt:
        if not hasattr(ctx, "cond") or not hasattr(ctx, "thenlist"):
            return IfStmt(cond=VarRef("/*IF_COND*/", self._loc(ctx)), then_body=[], loc=self._loc(ctx))

        main_cond = self.visit(ctx.cond[0])
        main_then = self.visit(ctx.thenlist[0])

        elifs = []
        for i in range(1, len(ctx.cond)):
            c = self.visit(ctx.cond[i])
            t = self.visit(ctx.thenlist[i])
            elifs.append((c, t))

        else_body: List[Stmt] = []
        if getattr(ctx, "elselist", None) is not None:
            else_body = self.visit(ctx.elselist)

        return IfStmt(
            cond=main_cond,
            then_body=main_then,
            elif_branches=elifs,
            else_body=else_body,
            loc=self._loc(ctx),
        )

    def visitCase_statement(self, ctx: IEC61131Parser.Case_statementContext) -> CaseStmt:
        selector = self.visit(ctx.cond) if hasattr(ctx, "cond") else VarRef("/*CASE*/", self._loc(ctx))
        entries: List[CaseEntry] = []

        for ectx in ctx.case_entry():
            conds: List[CaseCond] = []
            for cctx in ectx.case_condition():
                conds.append(CaseCond(text=cctx.getText(), loc=self._loc(cctx)))
            body = self.visit(ectx.statement_list())
            entries.append(CaseEntry(conds=conds, body=body, loc=self._loc(ectx)))

        else_body: List[Stmt] = []
        if getattr(ctx, "elselist", None) is not None:
            else_body = self.visit(ctx.elselist)

        return CaseStmt(cond=selector, entries=entries, else_body=else_body, loc=self._loc(ctx))

    def visitFor_statement(self, ctx: IEC61131Parser.For_statementContext) -> ForStmt:
        var_name = ctx.var.text if hasattr(ctx, "var") else "i"
        start_expr = self.visit(ctx.begin) if hasattr(ctx, "begin") else Literal("0", "CONST", self._loc(ctx))
        end_expr = self.visit(ctx.endPosition) if hasattr(ctx, "endPosition") else Literal("0", "CONST", self._loc(ctx))
        step_expr = self.visit(ctx.by) if getattr(ctx, "by", None) is not None else None
        body = self.visit(ctx.statement_list()) if ctx.statement_list() else []
        return ForStmt(var=var_name, start=start_expr, end=end_expr, step=step_expr, body=body, loc=self._loc(ctx))

    def visitWhile_statement(self, ctx: IEC61131Parser.While_statementContext) -> WhileStmt:
        cond = self.visit(ctx.expression()) if ctx.expression() else VarRef("/*WHILE*/", self._loc(ctx))
        body = self.visit(ctx.statement_list()) if ctx.statement_list() else []
        return WhileStmt(cond=cond, body=body, loc=self._loc(ctx))

    def visitRepeat_statement(self, ctx: IEC61131Parser.Repeat_statementContext) -> RepeatStmt:
        body = self.visit(ctx.statement_list()) if ctx.statement_list() else []
        until = self.visit(ctx.expression()) if ctx.expression() else VarRef("/*UNTIL*/", self._loc(ctx))
        return RepeatStmt(body=body, until=until, loc=self._loc(ctx))

    # --------------------
    # expressions
    # --------------------
    def visitUnaryMinusExpr(self, ctx: IEC61131Parser.UnaryMinusExprContext) -> Expr:
        sub = self.visit(ctx.sub)
        return UnaryOp(op="-", operand=sub, loc=self._loc(ctx))

    def visitUnaryNegateExpr(self, ctx: IEC61131Parser.UnaryNegateExprContext) -> Expr:
        sub = self.visit(ctx.sub)
        return UnaryOp(op="NOT", operand=sub, loc=self._loc(ctx))

    def visitParenExpr(self, ctx: IEC61131Parser.ParenExprContext) -> Expr:
        return self.visit(ctx.sub)

    def _make_binop(self, ctx) -> BinOp:
        left = self.visit(ctx.left)
        right = self.visit(ctx.right)
        op = ctx.op.text if hasattr(ctx, "op") and ctx.op is not None else "?"
        return BinOp(op=op, left=left, right=right, loc=self._loc(ctx))

    def visitBinaryPowerExpr(self, ctx: IEC61131Parser.BinaryPowerExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryModDivExpr(self, ctx: IEC61131Parser.BinaryModDivExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryMultExpr(self, ctx: IEC61131Parser.BinaryMultExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryPlusMinusExpr(self, ctx: IEC61131Parser.BinaryPlusMinusExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryCmpExpr(self, ctx: IEC61131Parser.BinaryCmpExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryEqExpr(self, ctx: IEC61131Parser.BinaryEqExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryAndExpr(self, ctx: IEC61131Parser.BinaryAndExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryOrExpr(self, ctx: IEC61131Parser.BinaryOrExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitBinaryXORExpr(self, ctx: IEC61131Parser.BinaryXORExprContext) -> Expr:
        return self._make_binop(ctx)

    def visitPrimaryExpr(self, ctx: IEC61131Parser.PrimaryExprContext) -> Expr:
        return self.visit(ctx.primary_expression())

    def visitPrimary_expression(self, ctx: IEC61131Parser.Primary_expressionContext) -> Expr:
        if ctx.constant():
            return self.visit(ctx.constant())
        if getattr(ctx, "v", None) is not None:
            return self.visit(ctx.v)
        if ctx.variable():
            return self.visit(ctx.variable())
        if ctx.invocation():
            return self.visit(ctx.invocation())
        return VarRef(name=ctx.getText(), loc=self._loc(ctx))

    def visitConstant(self, ctx: IEC61131Parser.ConstantContext) -> Literal:
        return Literal(value=ctx.getText(), type="CONST", loc=self._loc(ctx))

    # --------------------
    # variables / symbolic chain (for general expressions, not callee strings)
    # --------------------
    def visitVariable(self, ctx: IEC61131Parser.VariableContext) -> Expr:
        """
        variable : direct_variable | symbolic_variable ;
        """
        if ctx.direct_variable():
            return VarRef(name=ctx.direct_variable().getText(), loc=self._loc(ctx))
        if ctx.symbolic_variable():
            return self.visit(ctx.symbolic_variable())
        return VarRef(name=ctx.getText(), loc=self._loc(ctx))

    def visitSymbolic_variable(self, ctx: IEC61131Parser.Symbolic_variableContext) -> Expr:
        """
        构建变量访问表达式：
        - 根 VarRef(a)
        - subscript_list -> ArrayAccess 链
        - DOT other -> FieldAccess/ArrayAccess 链
        """
        base: Expr = VarRef(name=ctx.a.getText(), loc=self._loc(ctx))

        if ctx.subscript_list():
            indices = self.visit(ctx.subscript_list())
            for idx in indices:
                base = ArrayAccess(base=base, index=idx, loc=self._loc(ctx))

        if ctx.other is not None:
            base = self._attach_field_chain(base, ctx.other)

        return base

    def visitSubscript_list(self, ctx: IEC61131Parser.Subscript_listContext) -> List[Expr]:
        exprs: List[Expr] = []
        for ectx in ctx.expression():
            exprs.append(self.visit(ectx))
        return exprs

    def _attach_field_chain(self, base: Expr, other_ctx: IEC61131Parser.Symbolic_variableContext) -> Expr:
        expr: Expr = FieldAccess(base=base, field=other_ctx.a.getText(), loc=self._loc(other_ctx))

        if other_ctx.subscript_list():
            indices = self.visit(other_ctx.subscript_list())
            for idx in indices:
                expr = ArrayAccess(base=expr, index=idx, loc=self._loc(other_ctx))

        if other_ctx.other is not None:
            expr = self._attach_field_chain(expr, other_ctx.other)

        return expr
