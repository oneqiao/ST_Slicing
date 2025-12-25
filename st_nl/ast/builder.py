from __future__ import annotations

from typing import List, Optional, Union

from antlr4 import ParserRuleContext

from ..generated.IEC61131ParserVisitor import IEC61131ParserVisitor
from ..generated.IEC61131Parser import IEC61131Parser

from .nodes import (
    VarRef,
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
    NamedArg,   # 新增
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
)


POU = Union[ProgramDecl, FBDecl]


class ASTBuilder(IEC61131ParserVisitor):
    """
    IEC 61131-3 ST parse tree -> simplified AST.

    目标：先稳定产出 AST（可用于 AST->NL），对不支持的结构做保底处理。
    """

    def __init__(self, filename: str = "<memory>"):
        self.filename = filename

    # --------------------
    # helpers
    # --------------------
    def _loc(self, ctx: ParserRuleContext) -> SourceLocation:
        tok = ctx.start
        return SourceLocation(
            file=self.filename,
            line=getattr(tok, "line", 0),
            column=getattr(tok, "column", 0),
        )

    def _text(self, node) -> str:
        return node.getText() if node is not None else ""

    def _get_ctx_id_text(self, ctx) -> Optional[str]:
        """
        兼容 ANTLR 生成字段命名：ctx.id 或 ctx.id_
        同时兼容：
        - TerminalNode / ParserRuleContext: 有 getText()
        - CommonToken: 只有 .text
        """
        for attr in ("id_", "id"):
            obj = getattr(ctx, attr, None)
            if obj is None:
                continue

            # TerminalNode / Context
            if hasattr(obj, "getText"):
                return obj.getText()

            # CommonToken / Token
            txt = getattr(obj, "text", None)
            if txt is not None:
                return txt

            # 兜底：转字符串（一般不需要）
            try:
                return str(obj)
            except Exception:
                return None

        return None


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

    def visitLibrary_element_declaration(
        self, ctx: IEC61131Parser.Library_element_declarationContext
    ):
        if ctx.program_declaration():
            return self.visit(ctx.program_declaration())
        if ctx.function_block_declaration():
            return self.visit(ctx.function_block_declaration())
        return None

    # --------------------
    # PROGRAM / FB
    # --------------------
    def visitProgram_declaration(
        self, ctx: IEC61131Parser.Program_declarationContext
    ) -> ProgramDecl:
        # 兼容不同命名：有的 grammar 写 identifier=IDENTIFIER
        name = None
        if hasattr(ctx, "identifier") and ctx.identifier is not None:
            name = ctx.identifier.text
        elif ctx.IDENTIFIER():
            name = ctx.IDENTIFIER().getText()
        else:
            name = "PROGRAM"

        vars_ = self.visit(ctx.var_decls()) if ctx.var_decls() else []
        body = self._extract_body(ctx.body()) if ctx.body() else []

        return ProgramDecl(name=name, vars=vars_, body=body, loc=self._loc(ctx))

    def visitFunction_block_declaration(
        self, ctx: IEC61131Parser.Function_block_declarationContext
    ) -> FBDecl:
        name = None
        if hasattr(ctx, "identifier") and ctx.identifier is not None:
            name = ctx.identifier.text
        elif ctx.IDENTIFIER():
            name = ctx.IDENTIFIER().getText()
        else:
            name = "FUNCTION_BLOCK"

        vars_ = self.visit(ctx.var_decls()) if ctx.var_decls() else []
        body = self._extract_body(ctx.body()) if ctx.body() else []

        return FBDecl(name=name, vars=vars_, body=body, loc=self._loc(ctx))

    def _extract_body(self, body_ctx: IEC61131Parser.BodyContext) -> List[Stmt]:
        # 你已确认不考虑 IL；这里仅取 statement_list
        if body_ctx.statement_list():
            return self.visit(body_ctx.statement_list())
        return []

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
        # 这里用 “按对齐列表 zip” 的方式，但加一层长度保护
        id_lists = list(inner.identifier_list())
        type_decls = list(inner.type_declaration())
        n = min(len(id_lists), len(type_decls))

        for i in range(n):
            id_list_ctx = id_lists[i]
            type_ctx = type_decls[i]
            type_str = type_ctx.getText()
            # identifier_list: names+=variable_names (COMMA names+=variable_names)*
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
    def visitStatement_list(
        self, ctx: IEC61131Parser.Statement_listContext
    ) -> List[Stmt]:
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

    # assignment
    def visitAssignment_statement(
        self, ctx: IEC61131Parser.Assignment_statementContext
    ) -> Assignment:
        target = self.visit(ctx.left)
        value = self.visit(ctx.right)
        op = ctx.op.text if hasattr(ctx, "op") and ctx.op is not None else ":="
        return Assignment(target=target, value=value, op=op, loc=self._loc(ctx))

    # invocation as statement
    def visitInvocation_statement(
        self, ctx: IEC61131Parser.Invocation_statementContext
    ) -> CallStmt:
        return self.visit(ctx.invocation())
    
    def _collect_call_args(self, inv_ctx: IEC61131Parser.InvocationContext):
        """
        从 invocation 中提取位置参数与命名参数，保持源码顺序。
        grammar:
        invocation : id=symbolic_variable '(' ((expression|param_assignment) (',' ...)*)? ')' ;
        由于 expression() 与 param_assignment() 在 parse tree 中是分开的列表，
        为了保持原始顺序，这里按 children 扫描，但只处理这两类节点，顺序是可靠的。
        """
        pos_args: List[Expr] = []
        named_args: List[NamedArg] = []

        for child in inv_ctx.getChildren():
            cname = type(child).__name__
            if cname.endswith("Param_assignmentContext"):
                v = self.visit(child)
                if isinstance(v, NamedArg):
                    named_args.append(v)
                elif isinstance(v, Expr):
                    pos_args.append(v)
            elif cname.endswith("ExpressionContext"):
                v = self.visit(child)
                if isinstance(v, Expr):
                    pos_args.append(v)

        return pos_args, named_args


    def visitInvocation(self, ctx: IEC61131Parser.InvocationContext) -> CallStmt:
        fb_name = self._get_invocation_name(ctx)
        pos_args, named_args = self._collect_call_args(ctx)
        return CallStmt(fb_name=fb_name, pos_args=pos_args, named_args=named_args, loc=self._loc(ctx))

    def visitParam_assignment(self, ctx: IEC61131Parser.Param_assignmentContext):
        """
        param_assignment
        : id=IDENTIFIER ARROW_RIGHT v=variable
        | (id=IDENTIFIER ASSIGN)? expression
        ;
        返回：
        - NamedArg(name, value)  （当出现参数名时）
        - Expr                   （当是纯 expression 位置参数时）
        """
        name = self._get_ctx_id_text(ctx)

        # 情况 1：id := variable
        if getattr(ctx, "v", None) is not None and ctx.v is not None:
            val = self.visit(ctx.v)
            if name is None:
                return val
            return NamedArg(name=name, value=val, loc=self._loc(ctx))

        # 情况 2：(id :=)? expression
        if ctx.expression():
            val = self.visit(ctx.expression())
            if name is not None:
                return NamedArg(name=name, value=val, loc=self._loc(ctx))
            return val

        # 防御：不应发生
        return VarRef(name=ctx.getText(), loc=self._loc(ctx))

    # IF
    def visitIf_statement(self, ctx: IEC61131Parser.If_statementContext) -> IfStmt:
        if not hasattr(ctx, "cond") or not hasattr(ctx, "thenlist"):
            # fallback（不崩）
            return IfStmt(
                cond=VarRef("/*IF_COND*/", self._loc(ctx)),
                then_body=[],
                loc=self._loc(ctx),
            )

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

    # CASE
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

    # FOR
    def visitFor_statement(self, ctx: IEC61131Parser.For_statementContext) -> ForStmt:
        var_name = ctx.var.text if hasattr(ctx, "var") else "i"
        start_expr = self.visit(ctx.begin) if hasattr(ctx, "begin") else Literal("0", "CONST", self._loc(ctx))
        end_expr = self.visit(ctx.endPosition) if hasattr(ctx, "endPosition") else Literal("0", "CONST", self._loc(ctx))
        step_expr = self.visit(ctx.by) if getattr(ctx, "by", None) is not None else None
        body = self.visit(ctx.statement_list()) if ctx.statement_list() else []
        return ForStmt(var=var_name, start=start_expr, end=end_expr, step=step_expr, body=body, loc=self._loc(ctx))

    # WHILE
    def visitWhile_statement(self, ctx: IEC61131Parser.While_statementContext) -> WhileStmt:
        cond = self.visit(ctx.expression()) if ctx.expression() else VarRef("/*WHILE*/", self._loc(ctx))
        body = self.visit(ctx.statement_list()) if ctx.statement_list() else []
        return WhileStmt(cond=cond, body=body, loc=self._loc(ctx))

    # REPEAT
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
            inv = ctx.invocation()
            # 这里返回 CallExpr（表达式级调用）
            func_name = self._get_invocation_name(inv)
            pos_args, named_args = self._collect_call_args(inv)
            return CallExpr(func=func_name, pos_args=pos_args, named_args=named_args, loc=self._loc(inv))

        # fallback
        return VarRef(name=ctx.getText(), loc=self._loc(ctx))

    def visitConstant(self, ctx: IEC61131Parser.ConstantContext) -> Literal:
        return Literal(value=ctx.getText(), type="CONST", loc=self._loc(ctx))

    def visitVariable(self, ctx: IEC61131Parser.VariableContext) -> Expr:
        """
        variable : direct_variable | symbolic_variable ;
        """
        if ctx.direct_variable():
            # direct_variable 是 DIRECT_VARIABLE_LITERAL，例如 %IX0.0
            return VarRef(name=ctx.direct_variable().getText(), loc=self._loc(ctx))
        if ctx.symbolic_variable():
            return self.visit(ctx.symbolic_variable())
        # fallback
        return VarRef(name=ctx.getText(), loc=self._loc(ctx))


    def visitSymbolic_variable(self, ctx: IEC61131Parser.Symbolic_variableContext) -> Expr:
        """
        symbolic_variable :
            a=variable_names
            ( (deref += CARET)+ )?
            ( subscript_list (CARET)? )?
            ( DOT other=symbolic_variable )?
        ;
        我们构建：
        - 根名字 VarRef(a)
        - 若有 subscript_list：ArrayAccess 链（支持多维下标）
        - 若有 DOT other：FieldAccess(base, field) + other 的递归结果
        """
        # 1) 根名字
        base: Expr = VarRef(name=ctx.a.getText(), loc=self._loc(ctx))

        # 2) 数组下标（subscript_list 里可以有多个 expression）
        if ctx.subscript_list():
            indices = self.visit(ctx.subscript_list())  # -> List[Expr]
            for idx in indices:
                base = ArrayAccess(base=base, index=idx, loc=self._loc(ctx))

        # 3) 字段访问（递归）
        if ctx.other is not None:
            # other 本身是一个 symbolic_variable，代表 ".xxx[...].yyy"
            # 我们需要把它展开成 FieldAccess/ArrayAccess 链挂到 base 上
            base = self._attach_field_chain(base, ctx.other)

        return base


    def visitSubscript_list(self, ctx: IEC61131Parser.Subscript_listContext) -> List[Expr]:
        """
        subscript_list:
            LBRACKET expression (COMMA expression)* RBRACKET
        ;
        返回 List[Expr]，用于多维数组或多个下标。
        """
        exprs: List[Expr] = []
        for ectx in ctx.expression():
            exprs.append(self.visit(ectx))
        return exprs


    def _attach_field_chain(self, base: Expr, other_ctx: IEC61131Parser.Symbolic_variableContext) -> Expr:
        """
        将 .other 这条 symbolic_variable 链接到 base 上。
        other_ctx 对应 grammar 中的:
            DOT other=symbolic_variable
        例如：
        base = VarRef("InImage")
        other_ctx 表示 symbolic_variable("Width")
        -> FieldAccess(base, "Width")

        更复杂：
        base = ArrayAccess(VarRef("a"), idx)
        other_ctx 表示 symbolic_variable("x", subscript_list=[i], other=...)
        -> FieldAccess(base, "x") 再 ArrayAccess(..., i) 再 FieldAccess(..., ...)
        """
        # other 的根字段名
        expr: Expr = FieldAccess(base=base, field=other_ctx.a.getText(), loc=self._loc(other_ctx))

        # other 的下标（如果字段本身也是数组）
        if other_ctx.subscript_list():
            indices = self.visit(other_ctx.subscript_list())
            for idx in indices:
                expr = ArrayAccess(base=expr, index=idx, loc=self._loc(other_ctx))

        # 递归处理 further .other
        if other_ctx.other is not None:
            expr = self._attach_field_chain(expr, other_ctx.other)

        return expr
    
    def _get_invocation_name(self, inv_ctx: IEC61131Parser.InvocationContext) -> str:
        name = self._get_ctx_id_text(inv_ctx)
        if name:
            return name
        # 最后兜底：从文本切
        text = inv_ctx.getText()
        return text.split("(")[0].strip()





