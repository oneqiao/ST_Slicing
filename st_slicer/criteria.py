# """
# criteria.py

# 通用 PLC / 运动控制 / 状态机 ST 程序切片准则挖掘模块（改进版，普适性增强）

# 核心特性：
# 1) 修复 PDG node_id 与 DefUseResult 列表索引不一致导致的漏检
# 2) 控制结构（IF/CASE/LOOP）默认使用 region slice（避免 backward slice 产生空 IF 壳）
# 3) 调用识别兼容 AST：CallStmt.fb_name / CallExpr.func
# 4) 增加兜底覆盖准则 any_def / any_call（排除已覆盖节点 + 均匀采样），保证覆盖下限
# 5) 点状准则可合并为多起点 seed_set（start_nodes），减少碎片、提高块完整性
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Set, Callable, Tuple

# from .pdg.pdg_builder import ProgramDependenceGraph
# from .dataflow.def_use import DefUseResult
# from .dataflow.var_access import VarAccess
# from .sema.symbols import POUSymbolTable
# from st_slicer.blocks.core import SlicingCriterion


# # =========================
# # 配置
# # =========================

# @dataclass
# class CriterionConfig:
#     # 错误类变量
#     error_name_keywords: Optional[List[str]] = None
#     # 状态类变量（状态机 / 模式）
#     state_name_keywords: Optional[List[str]] = None
#     # API/功能块调用前缀（可选）
#     api_name_prefixes: Optional[List[str]] = None

#     # 轴 / 通道 / 关节等
#     axis_name_keywords: Optional[List[str]] = None
#     # 运动轮廓/物理量变量
#     motion_quantity_keywords: Optional[List[str]] = None
#     # 阈值 / 限值变量
#     limit_keywords: Optional[List[str]] = None
#     # 中断 / 停止 / 退出变量
#     interrupt_keywords: Optional[List[str]] = None
#     # 完成 / 结束 / 允许变量
#     done_keywords: Optional[List[str]] = None

#     # --- 调用策略 ---
#     # api_call 是否启用前缀过滤（默认 False：最大化覆盖；若噪声大建议设 True）
#     api_prefix_filter_enabled: bool = False

#     # any_def / any_call 兜底准则最大数量（0 表示关闭）
#     max_any_def: int = 15
#     max_any_call: int = 30

#     # 控制结构/循环结构准则是否使用 region slice 标记（推荐 True）
#     use_region_for_control: bool = True

#     # region seeds 过滤：仅保留“AST body 非空”的控制结构/循环结构
#     filter_empty_control_region: bool = True

#     # 合并点状准则为 seed_set 时，每组最多保留多少个 seed（过大易把切片扩成整篇）
#     max_seeds_per_group: int = 10

#     def __post_init__(self):
#         if self.error_name_keywords is None:
#             self.error_name_keywords = ["error", "err", "alarm", "fault", "diag", "status"]

#         if self.state_name_keywords is None:
#             self.state_name_keywords = ["stage", "state", "step", "phase", "mode"]

#         if self.api_name_prefixes is None:
#             self.api_name_prefixes = ["MC_", "FB_", "AXIS_", "DRV_"]

#         if self.axis_name_keywords is None:
#             self.axis_name_keywords = ["axis", "axes", "joint", "servo", "motor", "channel"]

#         if self.motion_quantity_keywords is None:
#             self.motion_quantity_keywords = [
#                 "pos", "position", "angle", "dist", "distance",
#                 "vel", "velocity", "speed",
#                 "acc", "accel", "decel", "dec", "jerk",
#                 "time", "t_", "dt",
#                 "len", "length",
#                 "count", "cnt",
#                 "segment", "seg", "section",
#             ]

#         if self.limit_keywords is None:
#             self.limit_keywords = ["min", "max", "limit", "lim", "tol", "threshold"]

#         if self.interrupt_keywords is None:
#             self.interrupt_keywords = ["interrupt", "abort", "stop", "emergency", "hold", "cancel", "reset"]

#         if self.done_keywords is None:
#             self.done_keywords = ["done", "complete", "finished", "ready", "enable", "allow"]


# # 工具函数：匹配/对齐/AST探测
# def _match_any_keyword(name: str, keywords: List[str]) -> bool:
#     if not name:
#         return False
#     low = name.lower()
#     for kw in keywords:
#         if kw and kw.lower() in low:
#             return True
#     return False


# def _build_du_indexer(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
# ) -> Callable[[int], Optional[int]]:
#     """
#     返回函数：node_id -> du_index

#     兼容三种常见情况：
#     A) node_id 连续且可直接做索引
#     B) du 长度 == PDG 节点数：假设 du 顺序与 sorted(pdg.nodes.keys()) 一致
#     C) 无法推断：返回 None
#     """
#     node_ids = list(getattr(pdg, "nodes", {}).keys())
#     if not node_ids:
#         return lambda _nid: None

#     n_du = len(du.def_vars)
#     max_id = max(node_ids)

#     # 情况 A：node_id 可直接作为索引
#     if max_id < n_du and min(node_ids) >= 0:
#         return lambda nid: nid if 0 <= nid < n_du else None

#     # 情况 B：长度一致，按排序对齐（常见）
#     if n_du == len(node_ids):
#         ordered = sorted(node_ids)
#         pos = {nid: i for i, nid in enumerate(ordered)}
#         return lambda nid: pos.get(nid, None)

#     # 情况 C：无法推断
#     return lambda _nid: None


# def _du_defs(du: DefUseResult, idx: Optional[int]) -> Set[str]:
#     if idx is None:
#         return set()
#     if 0 <= idx < len(du.def_vars):
#         return du.def_vars[idx] or set()
#     return set()


# def _du_uses(du: DefUseResult, idx: Optional[int]) -> Set[str]:
#     if idx is None:
#         return set()
#     if 0 <= idx < len(du.use_vars):
#         return du.use_vars[idx] or set()
#     return set()


# def _du_def_accesses(du: DefUseResult, idx: Optional[int]) -> Set[VarAccess]:
#     if idx is None:
#         return set()
#     if 0 <= idx < len(du.def_accesses):
#         return du.def_accesses[idx] or set()
#     return set()


# def _du_use_accesses(du: DefUseResult, idx: Optional[int]) -> Set[VarAccess]:
#     if idx is None:
#         return set()
#     if 0 <= idx < len(du.use_accesses):
#         return du.use_accesses[idx] or set()
#     return set()


# def _get_ast_node_for_pdg_node(pdg: ProgramDependenceGraph, node_id: int) -> Any:
#     mapping = getattr(pdg, "node_to_ast", None)
#     if mapping is None:
#         return None
#     return mapping.get(node_id)


# def _extract_call_callee_name(ast_node: Any) -> Tuple[Optional[str], Optional[str]]:
#     """
#     兼容你的 AST：
#       - CallStmt.fb_name  -> (name, "fb_instance")
#       - CallExpr.func     -> (name, "function")
#     """
#     if ast_node is None:
#         return None, None

#     if hasattr(ast_node, "fb_name"):
#         v = getattr(ast_node, "fb_name", None)
#         if isinstance(v, str) and v:
#             return v, "fb_instance"

#     if hasattr(ast_node, "func"):
#         v = getattr(ast_node, "func", None)
#         if isinstance(v, str) and v:
#             return v, "function"

#     if hasattr(ast_node, "name"):
#         v = getattr(ast_node, "name", None)
#         if isinstance(v, str) and v:
#             return v, "unknown"

#     # 某些 AST 用 callee 节点
#     if hasattr(ast_node, "callee"):
#         callee = getattr(ast_node, "callee", None)
#         if hasattr(callee, "name"):
#             v = getattr(callee, "name", None)
#             if isinstance(v, str) and v:
#                 return v, "callee"

#     return None, None


# def _ast_control_has_body(ast_node: Any) -> bool:
#     """仅依赖 AST 结构判断控制结构/循环结构是否有 body，减少无意义 region seeds。"""
#     if ast_node is None:
#         return False
#     cls = type(ast_node).__name__

#     if cls == "IfStmt":
#         then_n = len(getattr(ast_node, "then_body", []) or [])
#         elif_n = sum(len(body or []) for _, body in (getattr(ast_node, "elif_branches", []) or []))
#         else_n = len(getattr(ast_node, "else_body", []) or [])
#         return (then_n + elif_n + else_n) > 0

#     if cls == "CaseStmt":
#         entries = getattr(ast_node, "entries", []) or []
#         else_n = len(getattr(ast_node, "else_body", []) or [])
#         entry_n = sum(len(getattr(e, "body", []) or []) for e in entries)
#         return (entry_n + else_n) > 0

#     if cls in ("ForStmt", "WhileStmt", "RepeatStmt"):
#         return len(getattr(ast_node, "body", []) or []) > 0

#     return True


# def _uniform_sample(seq: List[int], k: int) -> List[int]:
#     """均匀采样：避免兜底准则集中在程序头部。"""
#     if k <= 0 or not seq:
#         return []
#     if len(seq) <= k:
#         return seq
#     step = max(1, len(seq) // k)
#     return seq[::step][:k]


# def _crit_key(c: SlicingCriterion) -> Tuple[Any, ...]:
#     extra = getattr(c, "extra", {}) or {}
#     access = extra.get("access")
#     access_str = access.pretty() if (access is not None and hasattr(access, "pretty")) else None
#     return (c.node_id, c.kind, c.variable, access_str)


# # =========================
# # 通用：I/O 输出准则
# # =========================

# def discover_io_output_criteria(
#     symtab: POUSymbolTable,
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []

#     output_bases: Set[str] = set()
#     for name, sym in getattr(symtab, "vars", {}).items():
#         storage = (getattr(sym, "storage", "") or "").upper()
#         if "OUTPUT" in storage or "IN_OUT" in storage:
#             output_bases.add(name)

#     if not output_bases:
#         return criteria

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)

#         for v in _du_defs(du, idx):
#             if v in output_bases:
#                 criteria.append(SlicingCriterion(
#                     node_id=node_id,
#                     kind="io_output",
#                     variable=v,
#                     extra={"base": v, "access": None},
#                 ))

#         for va in _du_def_accesses(du, idx):
#             if getattr(va, "base", None) in output_bases:
#                 pretty = va.pretty() if hasattr(va, "pretty") else va.base
#                 criteria.append(SlicingCriterion(
#                     node_id=node_id,
#                     kind="io_output",
#                     variable=pretty,
#                     extra={"base": va.base, "access": va},
#                 ))

#     return criteria


# # =========================
# # 通用：状态变量
# # =========================

# def discover_state_variables(
#     symtab: POUSymbolTable,
#     du: DefUseResult,
#     pdg: ProgramDependenceGraph,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> Set[str]:
#     candidates: Set[str] = set()
#     for name in getattr(symtab, "vars", {}).keys():
#         if _match_any_keyword(name, config.state_name_keywords or []):
#             candidates.add(name)
#     if not candidates:
#         return set()

#     def_counts = {v: 0 for v in candidates}
#     use_counts = {v: 0 for v in candidates}

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         for v in _du_defs(du, idx):
#             if v in def_counts:
#                 def_counts[v] += 1
#         for v in _du_uses(du, idx):
#             if v in use_counts:
#                 use_counts[v] += 1

#     # 更稳健：优先 use>=2 且 def>=1；再退化 use>=2；最后 candidates
#     strong = {v for v in candidates if use_counts[v] >= 2 and def_counts[v] >= 1}
#     if strong:
#         return strong

#     mid = {v for v in candidates if use_counts[v] >= 2}
#     return mid or candidates


# def discover_state_transition_criteria(
#     state_vars: Set[str],
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []
#     if not state_vars:
#         return criteria

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         defs = _du_defs(du, idx)
#         def_acc = _du_def_accesses(du, idx)

#         hit = None
#         for v in defs:
#             if v in state_vars:
#                 hit = v
#                 break
#         if hit is None:
#             for va in def_acc:
#                 if getattr(va, "base", None) in state_vars:
#                     hit = va.base
#                     break
#         if hit is None:
#             continue

#         criteria.append(SlicingCriterion(
#             node_id=node_id,
#             kind="state_transition",
#             variable=hit,
#             extra={"state_vars": sorted(state_vars)},
#         ))

#     return criteria


# # =========================
# # 通用：错误逻辑
# # =========================

# def discover_error_criteria(
#     symtab: POUSymbolTable,
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []

#     error_bases = {name for name in getattr(symtab, "vars", {}).keys()
#                    if _match_any_keyword(name, config.error_name_keywords or [])}
#     if not error_bases:
#         return criteria

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         for v in _du_defs(du, idx):
#             if v in error_bases:
#                 criteria.append(SlicingCriterion(
#                     node_id=node_id,
#                     kind="error_logic",
#                     variable=v,
#                     extra={"base": v, "access": None},
#                 ))

#         for va in _du_def_accesses(du, idx):
#             if getattr(va, "base", None) in error_bases:
#                 pretty = va.pretty() if hasattr(va, "pretty") else va.base
#                 criteria.append(SlicingCriterion(
#                     node_id=node_id,
#                     kind="error_logic",
#                     variable=pretty,
#                     extra={"base": va.base, "access": va},
#                 ))

#     return criteria


# # =========================
# # 状态机：阶段边界 / 运行事件
# # =========================

# def discover_stage_boundary_criteria(
#     symtab: POUSymbolTable,
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []

#     state_vars = discover_state_variables(symtab, du, pdg, du_index, config)
#     if not state_vars:
#         return criteria

#     output_bases: Set[str] = set()
#     for name, sym in getattr(symtab, "vars", {}).items():
#         storage = (getattr(sym, "storage", "") or "").upper()
#         if "OUTPUT" in storage or "IN_OUT" in storage:
#             output_bases.add(name)
#     if not output_bases:
#         return criteria

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         defs = _du_defs(du, idx)
#         def_acc = _du_def_accesses(du, idx)

#         has_state_def = any(v in state_vars for v in defs) or any(getattr(va, "base", None) in state_vars for va in def_acc)
#         has_output_def = any(v in output_bases for v in defs) or any(getattr(va, "base", None) in output_bases for va in def_acc)
#         if not (has_state_def and has_output_def):
#             continue

#         state_name = next((v for v in defs if v in state_vars), None)
#         if state_name is None:
#             for va in def_acc:
#                 if getattr(va, "base", None) in state_vars:
#                     state_name = va.base
#                     break

#         criteria.append(SlicingCriterion(
#             node_id=node_id,
#             kind="stage_boundary",
#             variable=state_name or "<state>",
#             extra={"state_vars": sorted(state_vars), "output_bases": sorted(output_bases)},
#         ))

#     return criteria


# def discover_runtime_event_criteria(
#     symtab: POUSymbolTable,
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []

#     state_vars = discover_state_variables(symtab, du, pdg, du_index, config)
#     if not state_vars:
#         return criteria

#     error_bases = {name for name in getattr(symtab, "vars", {}).keys()
#                    if _match_any_keyword(name, config.error_name_keywords or [])}
#     done_like = {name for name in getattr(symtab, "vars", {}).keys()
#                  if _match_any_keyword(name, config.done_keywords or [])}

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         defs = _du_defs(du, idx)
#         uses = _du_uses(du, idx)
#         def_acc = _du_def_accesses(du, idx)
#         use_acc = _du_use_accesses(du, idx)

#         has_state_def = any(v in state_vars for v in defs) or any(getattr(va, "base", None) in state_vars for va in def_acc)
#         if not has_state_def:
#             continue

#         has_error_def = any(v in error_bases for v in defs) or any(getattr(va, "base", None) in error_bases for va in def_acc)
#         has_done_def = any(v in done_like for v in defs) or any(getattr(va, "base", None) in done_like for va in def_acc)
#         has_interrupt_use = any(_match_any_keyword(v, config.interrupt_keywords or []) for v in uses) or \
#                             any(_match_any_keyword(getattr(va, "base", ""), config.interrupt_keywords or []) for va in use_acc)

#         if has_interrupt_use:
#             kind = "runtime_interrupt"
#         elif has_error_def:
#             kind = "runtime_error"
#         elif has_done_def:
#             kind = "runtime_done"
#         else:
#             continue

#         state_name = next((v for v in defs if v in state_vars), None)
#         if state_name is None:
#             for va in def_acc:
#                 if getattr(va, "base", None) in state_vars:
#                     state_name = va.base
#                     break

#         criteria.append(SlicingCriterion(
#             node_id=node_id,
#             kind=kind,
#             variable=state_name or "<state>",
#             extra={
#                 "state_vars": sorted(state_vars),
#                 "error_bases": sorted(error_bases),
#                 "done_outputs": sorted(done_like),
#             },
#         ))

#     return criteria


# # =========================
# # 控制结构：Region slice 准则
# # =========================

# def discover_control_region_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []
#     if not hasattr(pdg, "node_to_ast"):
#         return criteria

#     for node_id in pdg.nodes.keys():
#         ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
#         if ast_node is None:
#             continue
#         cls = type(ast_node).__name__

#         if cls in ("IfStmt", "CaseStmt"):
#             if config.filter_empty_control_region and not _ast_control_has_body(ast_node):
#                 continue

#             criteria.append(SlicingCriterion(
#                 node_id=node_id,
#                 kind="control_region",
#                 variable=f"<{cls}>",
#                 extra={
#                     "ast_type": cls,
#                     "slice_strategy": "region" if config.use_region_for_control else "backward",
#                 },
#             ))

#     return criteria


# def discover_loop_region_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []
#     if not hasattr(pdg, "node_to_ast"):
#         return criteria

#     for node_id in pdg.nodes.keys():
#         ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
#         if ast_node is None:
#             continue
#         cls = type(ast_node).__name__

#         if cls in ("ForStmt", "WhileStmt", "RepeatStmt"):
#             if config.filter_empty_control_region and not _ast_control_has_body(ast_node):
#                 continue

#             criteria.append(SlicingCriterion(
#                 node_id=node_id,
#                 kind="loop_region",
#                 variable=f"<{cls}>",
#                 extra={
#                     "ast_type": cls,
#                     "slice_strategy": "region" if config.use_region_for_control else "backward",
#                 },
#             ))

#     return criteria


# # =========================
# # 运动物理量 / 特殊场景
# # =========================

# def discover_motion_quantity_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []
#     mq = config.motion_quantity_keywords or []
#     if not mq:
#         return criteria

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         defs = _du_defs(du, idx)
#         def_acc = _du_def_accesses(du, idx)

#         for v in defs:
#             if _match_any_keyword(v, mq):
#                 criteria.append(SlicingCriterion(
#                     node_id=node_id,
#                     kind="motion_quantity",
#                     variable=v,
#                     extra={"base": v, "access": None},
#                 ))

#         for va in def_acc:
#             base = getattr(va, "base", "")
#             if _match_any_keyword(base, mq):
#                 pretty = va.pretty() if hasattr(va, "pretty") else base
#                 criteria.append(SlicingCriterion(
#                     node_id=node_id,
#                     kind="motion_quantity",
#                     variable=pretty,
#                     extra={"base": base, "access": va},
#                 ))

#     return criteria


# def discover_motion_special_case_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []
#     mq = config.motion_quantity_keywords or []
#     lim = config.limit_keywords or []
#     if not mq or not lim:
#         return criteria

#     for node_id in pdg.nodes.keys():
#         idx = du_index(node_id)
#         defs = _du_defs(du, idx)
#         uses = _du_uses(du, idx)
#         def_acc = _du_def_accesses(du, idx)
#         use_acc = _du_use_accesses(du, idx)

#         has_mq_def = any(_match_any_keyword(v, mq) for v in defs) or any(_match_any_keyword(getattr(va, "base", ""), mq) for va in def_acc)
#         if not has_mq_def:
#             continue

#         has_lim_use = any(_match_any_keyword(v, lim) for v in uses) or any(_match_any_keyword(getattr(va, "base", ""), lim) for va in use_acc)
#         if not has_lim_use:
#             continue

#         mq_name = next((v for v in defs if _match_any_keyword(v, mq)), None)
#         if mq_name is None:
#             for va in def_acc:
#                 base = getattr(va, "base", None)
#                 if base and _match_any_keyword(base, mq):
#                     mq_name = base
#                     break

#         criteria.append(SlicingCriterion(
#             node_id=node_id,
#             kind="motion_special_case",
#             variable=mq_name or "<motion>",
#             extra={"limit_keywords": lim},
#         ))

#     return criteria


# # =========================
# # 调用准则：api_call
# # =========================

# def discover_api_call_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     criteria: List[SlicingCriterion] = []
#     if not hasattr(pdg, "node_to_ast"):
#         return criteria

#     prefixes = config.api_name_prefixes or []

#     for node_id in pdg.nodes.keys():
#         ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
#         if ast_node is None:
#             continue

#         cls = type(ast_node).__name__
#         callee, call_kind = _extract_call_callee_name(ast_node)

#         # 如果不是显式调用节点，跳过
#         if not callee:
#             if "Call" not in cls:
#                 continue
#             callee = "<call>"
#             call_kind = "unknown"

#         # function 类调用噪声通常很大：建议默认启用 prefix_filter 或只保留 fb_instance
#         if call_kind == "function" and config.api_prefix_filter_enabled and prefixes:
#             up = callee.upper()
#             if not any(up.startswith(p.upper()) for p in prefixes):
#                 continue

#         # 如果启用前缀过滤，则对所有调用统一过滤
#         if config.api_prefix_filter_enabled and prefixes:
#             up = callee.upper()
#             if not any(up.startswith(p.upper()) for p in prefixes):
#                 continue

#         criteria.append(SlicingCriterion(
#             node_id=node_id,
#             kind="api_call",
#             variable=callee,
#             extra={"ast_type": cls, "call_kind": call_kind},
#         ))

#     return criteria


# # =========================
# # 覆盖兜底：any_def / any_call（排除已覆盖节点 + 均匀采样）
# # =========================

# def discover_any_def_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     max_n: int,
#     exclude_node_ids: Optional[Set[int]] = None,
# ) -> List[SlicingCriterion]:
#     if max_n <= 0:
#         return []
#     exclude_node_ids = exclude_node_ids or set()

#     candidates: List[int] = []
#     for node_id in pdg.nodes.keys():
#         if node_id in exclude_node_ids:
#             continue
#         idx = du_index(node_id)
#         defs = _du_defs(du, idx)
#         def_acc = _du_def_accesses(du, idx)
#         if defs or def_acc:
#             candidates.append(node_id)

#     sampled = _uniform_sample(sorted(candidates), max_n)
#     return [
#         SlicingCriterion(node_id=nid, kind="any_def", variable="<def>", extra={})
#         for nid in sampled
#     ]


# def discover_any_call_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     du_index: Callable[[int], Optional[int]],
#     max_n: int,
#     exclude_node_ids: Optional[Set[int]] = None,
# ) -> List[SlicingCriterion]:
#     if max_n <= 0 or not hasattr(pdg, "node_to_ast"):
#         return []
#     exclude_node_ids = exclude_node_ids or set()

#     candidates: List[int] = []
#     for node_id in pdg.nodes.keys():
#         if node_id in exclude_node_ids:
#             continue
#         ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
#         if ast_node is None:
#             continue
#         cls = type(ast_node).__name__
#         callee, _kind = _extract_call_callee_name(ast_node)
#         if callee or ("Call" in cls):
#             candidates.append(node_id)

#     sampled = _uniform_sample(sorted(candidates), max_n)
#     out: List[SlicingCriterion] = []
#     for nid in sampled:
#         ast_node = _get_ast_node_for_pdg_node(pdg, nid)
#         callee, _kind = _extract_call_callee_name(ast_node)
#         out.append(SlicingCriterion(
#             node_id=nid,
#             kind="any_call",
#             variable=callee or "<call>",
#             extra={"ast_type": type(ast_node).__name__ if ast_node is not None else "<ast>"},
#         ))
#     return out


# # =========================
# # 点状准则合并为 seed_set（start_nodes）
# # =========================

# def merge_point_criteria_to_seed_sets(
#     criteria: List[SlicingCriterion],
#     kinds_to_merge: Set[str],
#     max_seeds_per_group: int = 15,
# ) -> List[SlicingCriterion]:
#     """
#     把“点状准则”合并成“集合起点准则”：
#       - 同 kind + 同 base 变量 的多个 node_id 合并成一个 criterion
#       - 通过 extra["start_nodes"] 提供多起点
#     """
#     groups: Dict[Tuple[str, str], List[SlicingCriterion]] = {}
#     for c in criteria:
#         if c.kind not in kinds_to_merge:
#             continue
#         extra = getattr(c, "extra", {}) or {}
#         base = extra.get("base") or c.variable
#         key = (c.kind, str(base))
#         groups.setdefault(key, []).append(c)

#     out: List[SlicingCriterion] = []
#     consumed: Set[Tuple[Any, ...]] = set()

#     for (kind, base), items in groups.items():
#         seed_ids = sorted({it.node_id for it in items})
#         if not seed_ids:
#             continue

#         if max_seeds_per_group > 0 and len(seed_ids) > max_seeds_per_group:
#             step = max(1, len(seed_ids) // max_seeds_per_group)
#             seed_ids = seed_ids[::step][:max_seeds_per_group]

#         rep = items[0]
#         rep_extra = (getattr(rep, "extra", {}) or {}).copy()
#         rep_extra["start_nodes"] = seed_ids
#         rep_extra["slice_strategy"] = "backward_multi"
#         rep_extra["merged_from"] = len(items)

#         out.append(SlicingCriterion(
#             node_id=rep.node_id,
#             kind=rep.kind + "_set",
#             variable=str(base),
#             extra=rep_extra,
#         ))

#         for it in items:
#             consumed.add(_crit_key(it))

#     for c in criteria:
#         if _crit_key(c) in consumed:
#             continue
#         out.append(c)

#     return out


# # =========================
# # 总入口：mine_slicing_criteria
# # =========================

# def mine_slicing_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     symtab: POUSymbolTable,
#     config: Optional[CriterionConfig] = None,
# ) -> List[SlicingCriterion]:
#     if config is None:
#         config = CriterionConfig()

#     du_index = _build_du_indexer(pdg, du)

#     criteria: List[SlicingCriterion] = []

#     # 1) 通用 I/O / 状态 / 错误
#     criteria.extend(discover_io_output_criteria(symtab, pdg, du, du_index))

#     state_vars = discover_state_variables(symtab, du, pdg, du_index, config)
#     criteria.extend(discover_state_transition_criteria(state_vars, pdg, du, du_index))

#     criteria.extend(discover_error_criteria(symtab, pdg, du, du_index, config))

#     # 2) 状态机事件
#     criteria.extend(discover_stage_boundary_criteria(symtab, pdg, du, du_index, config))
#     criteria.extend(discover_runtime_event_criteria(symtab, pdg, du, du_index, config))

#     # 3) 控制结构 / 循环结构：region slice 锚点
#     criteria.extend(discover_control_region_criteria(pdg, du, du_index, config))
#     criteria.extend(discover_loop_region_criteria(pdg, du, du_index, config))

#     # 4) 运动 / 物理量
#     criteria.extend(discover_motion_quantity_criteria(pdg, du, du_index, config))
#     criteria.extend(discover_motion_special_case_criteria(pdg, du, du_index, config))

#     # 5) 调用点
#     criteria.extend(discover_api_call_criteria(pdg, du, du_index, config))

#     # 6.5) 点状准则 -> seed_set（减少碎片）
#     criteria = merge_point_criteria_to_seed_sets(
#         criteria,
#         kinds_to_merge={
#             "error_logic",
#             "state_transition",
#             "io_output",
#             "motion_quantity",
#             "api_call",
#         },
#         max_seeds_per_group=config.max_seeds_per_group,
#     )

#     # 6) 兜底覆盖：排除已覆盖节点再采样（避免噪声主导）
#     strong_kinds = {
#         "io_output", "io_output_set",
#         "state_transition", "state_transition_set",
#         "error_logic", "error_logic_set",
#         "stage_boundary",
#         "runtime_interrupt", "runtime_error", "runtime_done",
#         "control_region", "loop_region",
#         "motion_quantity", "motion_quantity_set",
#         "motion_special_case",
#         "api_call", "api_call_set",
#     }
#     covered = {c.node_id for c in criteria if c.kind in strong_kinds}

#     criteria.extend(discover_any_def_criteria(pdg, du, du_index, config.max_any_def, exclude_node_ids=covered))
#     criteria.extend(discover_any_call_criteria(pdg, du, du_index, config.max_any_call, exclude_node_ids=covered))

#     # 7) 去重（按 node_id, kind, variable, access.pretty）
#     seen: Set[Tuple[Any, ...]] = set()
#     deduped: List[SlicingCriterion] = []
#     for c in criteria:
#         key = _crit_key(c)
#         if key in seen:
#             continue
#         seen.add(key)
#         deduped.append(c)

#     return deduped

"""
criteria.py (lean version)

面向通用 ST/PLC 的切片准则挖掘（精简版）：
- 保留：错误/状态机/运行事件/控制结构region/api调用
- 可选：io_output
- 默认关闭：motion_quantity、any_def、any_call（减少碎片与同质块）
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from collections import defaultdict

from .pdg.pdg_builder import ProgramDependenceGraph
from .dataflow.def_use import DefUseResult
from .dataflow.var_access import VarAccess
from .sema.symbols import POUSymbolTable
from st_slicer.blocks.core import SlicingCriterion


@dataclass
class CriterionConfig:
    # 名称关键词
    error_name_keywords: Optional[List[str]] = None
    state_name_keywords: Optional[List[str]] = None
    api_name_prefixes: Optional[List[str]] = None

    axis_name_keywords: Optional[List[str]] = None
    motion_quantity_keywords: Optional[List[str]] = None
    limit_keywords: Optional[List[str]] = None
    interrupt_keywords: Optional[List[str]] = None
    done_keywords: Optional[List[str]] = None

    # --- 精简开关 ---
    enable_io_output: bool = True
    enable_motion_special_case: bool = True
    enable_motion_quantity: bool = False          # 默认关闭（碎片主要来源）
    enable_any_def: bool = False                  # 默认关闭（用 coverage_fill 更好）
    enable_any_call: bool = False                 # 默认关闭

    # --- 覆盖率增强开关（新增） ---
    enable_structure_seeds: bool = True   # IF/CASE/FOR/WHILE/REPEAT 结构头种子
    enable_call_seeds: bool = True        # CallStmt/函数块调用种子
    enable_any_def_sparse: bool = True    # 稀疏 any_def（比 enable_any_def 更干净）
    enable_control_region: bool = False
    enable_loop_region: bool = False


    # api_call 前缀过滤（大工程建议开启）
    api_prefix_filter_enabled: bool = False

    # any_def / any_call 上限（只有 enable_* = True 时才生效）
    max_any_def: int = 15
    max_any_call: int = 15

    # 控制结构/循环结构使用 region slice
    use_region_for_control: bool = True

    # 合并点准则时的 seed 限制（减少碎片）
    max_seeds_per_group: int = 8

    def __post_init__(self):
        if self.error_name_keywords is None:
            self.error_name_keywords = ["error", "err", "alarm", "fault", "diag", "status"]
        if self.state_name_keywords is None:
            self.state_name_keywords = ["stage", "state", "step", "phase", "mode"]
        if self.api_name_prefixes is None:
            self.api_name_prefixes = ["MC_", "FB_", "AXIS_", "DRV_"]

        if self.axis_name_keywords is None:
            self.axis_name_keywords = ["axis", "axes", "joint", "servo", "motor", "channel"]

        if self.motion_quantity_keywords is None:
            self.motion_quantity_keywords = [
                "pos", "position", "angle", "dist", "distance",
                "vel", "velocity", "speed",
                "acc", "accel", "decel", "dec", "jerk",
                "time", "t_", "dt",
                "len", "length",
                "count", "cnt",
                "segment", "seg", "section",
            ]

        if self.limit_keywords is None:
            self.limit_keywords = ["min", "max", "limit", "lim", "tol", "threshold"]

        if self.interrupt_keywords is None:
            self.interrupt_keywords = ["interrupt", "abort", "stop", "emergency", "hold", "cancel", "reset"]

        if self.done_keywords is None:
            self.done_keywords = ["done", "complete", "finished", "ready", "enable", "allow"]


def _match_any_keyword(name: str, keywords: List[str]) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(kw and kw.lower() in low for kw in keywords)


def _build_du_indexer(pdg: ProgramDependenceGraph, du: DefUseResult) -> Callable[[int], Optional[int]]:
    node_ids = list(getattr(pdg, "nodes", {}).keys())
    if not node_ids:
        return lambda _nid: None

    n_du = len(du.def_vars)
    max_id = max(node_ids)

    # A) node_id 可直接索引
    if max_id < n_du and min(node_ids) >= 0:
        return lambda nid: nid if 0 <= nid < n_du else None

    # B) 长度一致，按排序对齐
    if n_du == len(node_ids):
        ordered = sorted(node_ids)
        pos = {nid: i for i, nid in enumerate(ordered)}
        return lambda nid: pos.get(nid, None)

    return lambda _nid: None


def _du_defs(du: DefUseResult, idx: Optional[int]) -> Set[str]:
    if idx is None:
        return set()
    if 0 <= idx < len(du.def_vars):
        return du.def_vars[idx] or set()
    return set()


def _du_uses(du: DefUseResult, idx: Optional[int]) -> Set[str]:
    if idx is None:
        return set()
    if 0 <= idx < len(du.use_vars):
        return du.use_vars[idx] or set()
    return set()


def _du_def_accesses(du: DefUseResult, idx: Optional[int]) -> Set[VarAccess]:
    if idx is None:
        return set()
    if 0 <= idx < len(du.def_accesses):
        return du.def_accesses[idx] or set()
    return set()


def _du_use_accesses(du: DefUseResult, idx: Optional[int]) -> Set[VarAccess]:
    if idx is None:
        return set()
    if 0 <= idx < len(du.use_accesses):
        return du.use_accesses[idx] or set()
    return set()


def _get_ast_node_for_pdg_node(pdg: ProgramDependenceGraph, node_id: int) -> Any:
    mapping = getattr(pdg, "node_to_ast", None)
    if mapping is None:
        return None
    return mapping.get(node_id)


def _extract_call_callee_name(ast_node: Any) -> Optional[str]:
    if ast_node is None:
        return None

    for attr in ("fb_name", "func", "name"):
        if hasattr(ast_node, attr):
            v = getattr(ast_node, attr, None)
            if isinstance(v, str) and v:
                return v

    if hasattr(ast_node, "callee"):
        callee = getattr(ast_node, "callee", None)
        if hasattr(callee, "name"):
            v = getattr(callee, "name", None)
            if isinstance(v, str) and v:
                return v
    return None

def _build_line_to_node_ids(ir2ast_stmt: List[Any]) -> Dict[int, List[int]]:
    """
    用 ir2ast_stmt 构建 line -> [node_id] 的反向索引（可靠性远高于 pdg.node_to_ast.loc）
    """
    m: Dict[int, List[int]] = defaultdict(list)
    for nid, st in enumerate(ir2ast_stmt or []):
        if st is None:
            continue
        loc = getattr(st, "loc", None)
        ln = getattr(loc, "line", None)
        if isinstance(ln, int) and ln > 0:
            m[ln].append(nid)
    return dict(m)


def discover_structural_seed_criteria(
    ir2ast_stmt: List[Any],
    code_lines: List[str],
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    结构头作为 seed（覆盖率增益最大且低噪声）：
      IF/CASE -> control_region (region slice)
      FOR/WHILE/REPEAT -> loop_region (region slice)
    """
    if not ir2ast_stmt or not code_lines:
        return []

    line2nids = _build_line_to_node_ids(ir2ast_stmt)
    out: List[SlicingCriterion] = []

    for ln, raw in enumerate(code_lines, start=1):
        t = raw.strip()
        if not t:
            continue
        u = t.upper()

        kind = None
        if u.startswith("IF "):
            kind = "control_region"
        elif u.startswith("CASE "):
            kind = "control_region"
        elif u.startswith("FOR "):
            kind = "loop_region"
        elif u.startswith("WHILE "):
            kind = "loop_region"
        elif u.startswith("REPEAT"):
            kind = "loop_region"

        if kind is None:
            continue

        nids = line2nids.get(ln) or []
        if not nids:
            continue

        # region slice：避免 backward 扩张过大，同时提升“结构完整性”
        out.append(SlicingCriterion(
            node_id=nids[0],
            kind=kind,
            variable=f"<line:{ln}>",
            extra={"slice": {"slice_strategy": "region" if config.use_region_for_control else "backward"}},
        ))

    return out


def discover_call_seed_criteria(
    ir2ast_stmt: List[Any],
) -> List[SlicingCriterion]:
    """
    AST 级调用语句种子：CallStmt / CallExpr
    不依赖 pdg.node_to_ast。
    """
    if not ir2ast_stmt:
        return []

    out: List[SlicingCriterion] = []
    for nid, st in enumerate(ir2ast_stmt):
        if st is None:
            continue
        cls = type(st).__name__
        if cls == "CallStmt":
            fb = getattr(st, "fb_name", None) or "<call>"
            out.append(SlicingCriterion(node_id=nid, kind="api_call", variable=str(fb), extra={"base": str(fb)}))
    return out


def discover_any_def_sparse_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    du_index,
    symtab: POUSymbolTable,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    稀疏 any_def：不是“随机抓节点”，而是按“变量”抽样，控制 max_any_def 上限。
    目标：补普通业务逻辑覆盖，但不制造太多碎片。
    """
    if config.max_any_def <= 0:
        return []

    # 排除掉已经被强语义准则覆盖的变量（减少同质块）
    excluded: Set[str] = set()
    for name, sym in getattr(symtab, "vars", {}).items():
        storage = (getattr(sym, "storage", "") or "").upper()
        if "OUTPUT" in storage or "IN_OUT" in storage:
            excluded.add(name)
    excluded |= {name for name in getattr(symtab, "vars", {}).keys() if _match_any_keyword(name, config.error_name_keywords or [])}
    excluded |= {name for name in getattr(symtab, "vars", {}).keys() if _match_any_keyword(name, config.state_name_keywords or [])}

    # var -> [node_id...]（只收 defs）
    var2nodes: Dict[str, List[int]] = defaultdict(list)
    for node_id in getattr(pdg, "nodes", {}).keys():
        idx = du_index(node_id)
        for v in _du_defs(du, idx):
            if v and v not in excluded:
                var2nodes[v].append(node_id)

    if not var2nodes:
        return []

    # 选 def 最频繁的一批变量（更可能覆盖“核心逻辑”）
    items = sorted(var2nodes.items(), key=lambda kv: len(set(kv[1])), reverse=True)

    out: List[SlicingCriterion] = []
    budget = config.max_any_def

    for v, nids in items:
        if budget <= 0:
            break
        uniq = sorted(set(nids))
        if not uniq:
            continue

        # 对每个变量只抽样少量 seed（最多 2），让覆盖扩张“平滑”而不是碎片化
        picks = [uniq[0]]
        if len(uniq) >= 3:
            picks.append(uniq[len(uniq)//2])

        for nid in picks[:budget]:
            out.append(SlicingCriterion(node_id=nid, kind="any_def_sparse", variable=v, extra={"base": v}))
            budget -= 1
            if budget <= 0:
                break

    return out

def discover_io_output_criteria(symtab: POUSymbolTable, pdg: ProgramDependenceGraph, du: DefUseResult, du_index) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []

    output_bases: Set[str] = set()
    for name, sym in getattr(symtab, "vars", {}).items():
        storage = (getattr(sym, "storage", "") or "").upper()
        if "OUTPUT" in storage or "IN_OUT" in storage:
            output_bases.add(name)

    if not output_bases:
        return criteria

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)

        for v in _du_defs(du, idx):
            if v in output_bases:
                criteria.append(SlicingCriterion(node_id=node_id, kind="io_output", variable=v, extra={"base": v, "access": None}))

        for va in _du_def_accesses(du, idx):
            if getattr(va, "base", None) in output_bases:
                pretty = va.pretty() if hasattr(va, "pretty") else va.base
                criteria.append(SlicingCriterion(node_id=node_id, kind="io_output", variable=pretty, extra={"base": va.base, "access": va}))

    return criteria


def discover_state_variables(symtab: POUSymbolTable, du: DefUseResult, pdg: ProgramDependenceGraph, du_index, config: CriterionConfig) -> Set[str]:
    candidates = {name for name in getattr(symtab, "vars", {}).keys() if _match_any_keyword(name, config.state_name_keywords or [])}
    if not candidates:
        return set()

    def_counts = {v: 0 for v in candidates}
    use_counts = {v: 0 for v in candidates}

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)
        for v in _du_defs(du, idx):
            if v in def_counts:
                def_counts[v] += 1
        for v in _du_uses(du, idx):
            if v in use_counts:
                use_counts[v] += 1

    result = {v for v in candidates if def_counts[v] >= 2 and use_counts[v] >= 2}
    return result or candidates


def discover_state_transition_criteria(state_vars: Set[str], pdg: ProgramDependenceGraph, du: DefUseResult, du_index) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    if not state_vars:
        return criteria

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)
        defs = _du_defs(du, idx)
        def_acc = _du_def_accesses(du, idx)

        hit = next((v for v in defs if v in state_vars), None)
        if hit is None:
            for va in def_acc:
                if getattr(va, "base", None) in state_vars:
                    hit = va.base
                    break
        if hit is None:
            continue

        criteria.append(SlicingCriterion(node_id=node_id, kind="state_transition", variable=hit, extra={"state_vars": sorted(state_vars)}))
    return criteria


def discover_error_criteria(symtab: POUSymbolTable, pdg: ProgramDependenceGraph, du: DefUseResult, du_index, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    error_bases = {name for name in getattr(symtab, "vars", {}).keys() if _match_any_keyword(name, config.error_name_keywords or [])}
    if not error_bases:
        return criteria

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)

        for v in _du_defs(du, idx):
            if v in error_bases:
                criteria.append(SlicingCriterion(node_id=node_id, kind="error_logic", variable=v, extra={"base": v, "access": None}))

        for va in _du_def_accesses(du, idx):
            if getattr(va, "base", None) in error_bases:
                pretty = va.pretty() if hasattr(va, "pretty") else va.base
                criteria.append(SlicingCriterion(node_id=node_id, kind="error_logic", variable=pretty, extra={"base": va.base, "access": va}))

    return criteria


def discover_stage_boundary_criteria(symtab: POUSymbolTable, pdg: ProgramDependenceGraph, du: DefUseResult, du_index, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    state_vars = discover_state_variables(symtab, du, pdg, du_index, config)
    if not state_vars:
        return criteria

    output_bases: Set[str] = set()
    for name, sym in getattr(symtab, "vars", {}).items():
        storage = (getattr(sym, "storage", "") or "").upper()
        if "OUTPUT" in storage or "IN_OUT" in storage:
            output_bases.add(name)
    if not output_bases:
        return criteria

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)
        defs = _du_defs(du, idx)
        def_acc = _du_def_accesses(du, idx)

        has_state_def = any(v in state_vars for v in defs) or any(getattr(va, "base", None) in state_vars for va in def_acc)
        has_output_def = any(v in output_bases for v in defs) or any(getattr(va, "base", None) in output_bases for va in def_acc)
        if not (has_state_def and has_output_def):
            continue

        state_name = next((v for v in defs if v in state_vars), None)
        if state_name is None:
            for va in def_acc:
                if getattr(va, "base", None) in state_vars:
                    state_name = va.base
                    break

        criteria.append(SlicingCriterion(
            node_id=node_id,
            kind="stage_boundary",
            variable=state_name or "<state>",
            extra={"state_vars": sorted(state_vars), "output_bases": sorted(output_bases)},
        ))
    return criteria


def discover_runtime_event_criteria(symtab: POUSymbolTable, pdg: ProgramDependenceGraph, du: DefUseResult, du_index, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    state_vars = discover_state_variables(symtab, du, pdg, du_index, config)
    if not state_vars:
        return criteria

    error_bases = {name for name in getattr(symtab, "vars", {}).keys() if _match_any_keyword(name, config.error_name_keywords or [])}
    done_like = {name for name in getattr(symtab, "vars", {}).keys() if _match_any_keyword(name, config.done_keywords or [])}

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)
        defs = _du_defs(du, idx)
        uses = _du_uses(du, idx)
        def_acc = _du_def_accesses(du, idx)
        use_acc = _du_use_accesses(du, idx)

        has_state_def = any(v in state_vars for v in defs) or any(getattr(va, "base", None) in state_vars for va in def_acc)
        if not has_state_def:
            continue

        has_error_def = any(v in error_bases for v in defs) or any(getattr(va, "base", None) in error_bases for va in def_acc)
        has_done_def = any(v in done_like for v in defs) or any(getattr(va, "base", None) in done_like for va in def_acc)
        has_interrupt_use = any(_match_any_keyword(v, config.interrupt_keywords or []) for v in uses) or \
                            any(_match_any_keyword(getattr(va, "base", ""), config.interrupt_keywords or []) for va in use_acc)

        if has_interrupt_use:
            kind = "runtime_interrupt"
        elif has_error_def:
            kind = "runtime_error"
        elif has_done_def:
            kind = "runtime_done"
        else:
            continue

        state_name = next((v for v in defs if v in state_vars), None)
        if state_name is None:
            for va in def_acc:
                if getattr(va, "base", None) in state_vars:
                    state_name = va.base
                    break

        criteria.append(SlicingCriterion(
            node_id=node_id,
            kind=kind,
            variable=state_name or "<state>",
            extra={"state_vars": sorted(state_vars)},
        ))
    return criteria


def discover_control_region_criteria(pdg: ProgramDependenceGraph, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    if not hasattr(pdg, "node_to_ast"):
        return criteria

    for node_id in pdg.nodes.keys():
        ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
        if ast_node is None:
            continue
        cls = type(ast_node).__name__
        if cls in ("IfStmt", "CaseStmt"):
            criteria.append(SlicingCriterion(
            node_id=node_id,
            kind="control_region",
            variable=f"<{cls}>",
            extra={
                "base": cls,  # 关键：让 merge 按 IfStmt/CaseStmt 分组，而不是按 node_id 分裂
                "ast_type": cls,
                "slice": {"slice_strategy": "region"} if config.use_region_for_control else {"slice_strategy": "backward"},
            },
        ))
    return criteria


def discover_loop_region_criteria(pdg: ProgramDependenceGraph, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    if not hasattr(pdg, "node_to_ast"):
        return criteria

    for node_id in pdg.nodes.keys():
        ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
        if ast_node is None:
            continue
        cls = type(ast_node).__name__
        criteria.append(SlicingCriterion(
            node_id=node_id,
            kind="control_region",
            variable=f"<{cls}>",
            extra={
                "base": cls,  # 关键：让 merge 按 IfStmt/CaseStmt 分组，而不是按 node_id 分裂
                "ast_type": cls,
                "slice": {"slice_strategy": "region"} if config.use_region_for_control else {"slice_strategy": "backward"},
            },
        ))
    return criteria


def discover_motion_special_case_criteria(pdg: ProgramDependenceGraph, du: DefUseResult, du_index, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    mq = config.motion_quantity_keywords or []
    lim = config.limit_keywords or []
    if not mq or not lim:
        return criteria

    for node_id in pdg.nodes.keys():
        idx = du_index(node_id)
        defs = _du_defs(du, idx)
        uses = _du_uses(du, idx)
        def_acc = _du_def_accesses(du, idx)
        use_acc = _du_use_accesses(du, idx)

        has_mq_def = any(_match_any_keyword(v, mq) for v in defs) or any(_match_any_keyword(getattr(va, "base", ""), mq) for va in def_acc)
        if not has_mq_def:
            continue

        has_lim_use = any(_match_any_keyword(v, lim) for v in uses) or any(_match_any_keyword(getattr(va, "base", ""), lim) for va in use_acc)
        if not has_lim_use:
            continue

        mq_name = next((v for v in defs if _match_any_keyword(v, mq)), None)
        if mq_name is None:
            for va in def_acc:
                base = getattr(va, "base", None)
                if base and _match_any_keyword(base, mq):
                    mq_name = base
                    break

        criteria.append(SlicingCriterion(node_id=node_id, kind="motion_special_case", variable=mq_name or "<motion>", extra={"limit_keywords": lim}))
    return criteria


def discover_api_call_criteria(pdg: ProgramDependenceGraph, config: CriterionConfig) -> List[SlicingCriterion]:
    criteria: List[SlicingCriterion] = []
    if not hasattr(pdg, "node_to_ast"):
        return criteria

    prefixes = config.api_name_prefixes or []

    for node_id in pdg.nodes.keys():
        ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
        if ast_node is None:
            continue

        cls = type(ast_node).__name__
        callee = _extract_call_callee_name(ast_node)

        if not callee:
            if "Call" not in cls:
                continue
            callee = "<call>"

        if config.api_prefix_filter_enabled and prefixes:
            up = callee.upper()
            if not any(up.startswith(p.upper()) for p in prefixes):
                continue

        criteria.append(SlicingCriterion(node_id=node_id, kind="api_call", variable=callee, extra={"ast_type": cls}))
    return criteria


def merge_point_criteria_to_seed_sets(criteria: List[SlicingCriterion], kinds_to_merge: Set[str], max_seeds_per_group: int) -> List[SlicingCriterion]:
    groups: Dict[Tuple[str, str], List[SlicingCriterion]] = {}
    for c in criteria:
        if c.kind not in kinds_to_merge:
            continue
        extra = getattr(c, "extra", {}) or {}
        base = extra.get("base") or c.variable
        groups.setdefault((c.kind, str(base)), []).append(c)

    out: List[SlicingCriterion] = []
    consumed: Set[int] = set()

    for (kind, base), items in groups.items():
        seed_ids = sorted({it.node_id for it in items})
        if not seed_ids:
            continue

        if max_seeds_per_group > 0 and len(seed_ids) > max_seeds_per_group:
            step = max(1, len(seed_ids) // max_seeds_per_group)
            seed_ids = seed_ids[::step][:max_seeds_per_group]

        rep = items[0]
        rep_extra = (getattr(rep, "extra", {}) or {}).copy()
        rep_extra["start_nodes"] = seed_ids
        rep_extra["slice_strategy"] = "backward_multi"
        rep_extra["merged_from"] = len(items)

        out.append(SlicingCriterion(
            node_id=rep.node_id,
            kind=rep.kind + "_set",
            variable=str(base),
            extra=rep_extra,
        ))

        for it in items:
            consumed.add(id(it))

    for c in criteria:
        if id(c) in consumed:
            continue
        out.append(c)

    return out


def _normalize_control_region_criteria(criteria, pdg):
    # Build ctrl_succ once for root picking
    ctrl_pred = getattr(pdg, "ctrl_pred", {}) or {}
    ctrl_succ = {}
    for n, ps in ctrl_pred.items():
        for p in ps:
            ctrl_succ.setdefault(p, set()).add(n)

    def pick_root(seed: int) -> int:
        if seed in ctrl_succ and ctrl_succ[seed]:
            return seed
        # walk upward to find a predicate root
        visited = set()
        frontier = [seed]
        for _ in range(32):
            if not frontier:
                break
            nxt = []
            for x in frontier:
                if x in visited:
                    continue
                visited.add(x)
                for p in (ctrl_pred.get(x) or ()):
                    if p in ctrl_succ and ctrl_succ[p]:
                        return p
                    nxt.append(p)
            frontier = nxt
        return seed

    seen = set()
    out = []
    for c in criteria:
        if c.kind == "control_region_set":
            root = pick_root(c.node_id)
            # 关键：让后续 pipeline 走 region 策略
            extra = dict(getattr(c, "extra", {}) or {})
            extra["slice_strategy"] = "region"
            # 去重：同一个 root 只要一个 criterion
            key = (c.kind, root)
            if key in seen:
                continue
            seen.add(key)
            try:
                c.node_id = root
                c.extra = extra
                out.append(c)
            except Exception:
                from st_slicer.blocks.core import SlicingCriterion
                out.append(SlicingCriterion(node_id=root, kind=c.kind, variable=c.variable, extra=extra))
        else:
            out.append(c)
    return out

def mine_slicing_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    symtab: POUSymbolTable,
    config: Optional[CriterionConfig] = None,
    *,
    ir2ast_stmt: Optional[List[Any]] = None,
    code_lines: Optional[List[str]] = None,
) -> List[SlicingCriterion]:

    if config is None:
        config = CriterionConfig()

    du_index = _build_du_indexer(pdg, du)
    criteria: List[SlicingCriterion] = []

    # 1) IO 输出（可选）
    if config.enable_io_output:
        criteria.extend(discover_io_output_criteria(symtab, pdg, du, du_index))

    # 2) 状态机（强烈建议保留）
    state_vars = discover_state_variables(symtab, du, pdg, du_index, config)
    criteria.extend(discover_state_transition_criteria(state_vars, pdg, du, du_index))

    # 3) 错误逻辑（强烈建议保留）
    criteria.extend(discover_error_criteria(symtab, pdg, du, du_index, config))

    # 4) 状态机事件（建议保留）
    criteria.extend(discover_stage_boundary_criteria(symtab, pdg, du, du_index, config))
    criteria.extend(discover_runtime_event_criteria(symtab, pdg, du, du_index, config))

    # 5) 控制结构 region（强烈建议保留）
    criteria.extend(discover_control_region_criteria(pdg, config))
    criteria.extend(discover_loop_region_criteria(pdg, config))

    # 6) 运动限值特殊片段（建议保留）
    if config.enable_motion_special_case:
        criteria.extend(discover_motion_special_case_criteria(pdg, du, du_index, config))

    # 7) API 调用（建议保留）
    criteria.extend(discover_api_call_criteria(pdg, config))

    # 7.1) 结构头 seeds（覆盖率增益最大，且不依赖 pdg.node_to_ast）
    if config.enable_structure_seeds and ir2ast_stmt is not None and code_lines is not None:
        criteria.extend(discover_structural_seed_criteria(ir2ast_stmt, code_lines, config))

    # 7.2) 调用 seeds（不依赖 pdg.node_to_ast）
    if config.enable_call_seeds and ir2ast_stmt is not None:
        criteria.extend(discover_call_seed_criteria(ir2ast_stmt))

    # 7.3) 稀疏 any_def（补普通逻辑覆盖，低噪声）
    if config.enable_any_def_sparse:
        criteria.extend(discover_any_def_sparse_criteria(pdg, du, du_index, symtab, config))

    if config.enable_control_region:
        criteria.extend(discover_control_region_criteria(pdg, config))
    if config.enable_loop_region:
        criteria.extend(discover_loop_region_criteria(pdg, config))

    # 8) 合并点准则为 seed_set（减少碎片）
    criteria = merge_point_criteria_to_seed_sets(
        criteria,
        kinds_to_merge={"error_logic", "state_transition", "io_output", "api_call", "control_region", "loop_region", "any_def_sparse"},
        max_seeds_per_group=config.max_seeds_per_group,
    )

    criteria = _normalize_control_region_criteria(criteria, pdg)

    # 9) 去重（node_id, kind, variable, access.pretty）
    seen: Set[Tuple[Any, ...]] = set()
    deduped: List[SlicingCriterion] = []
    for c in criteria:
        extra = getattr(c, "extra", {}) or {}
        access = extra.get("access")
        access_str = access.pretty() if (access is not None and hasattr(access, "pretty")) else None
        key = (c.node_id, c.kind, c.variable, access_str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    try:
        from collections import Counter
        cnt = Counter([c.kind for c in deduped])
        set_cnt = sum(1 for c in deduped if (c.kind or "").endswith("_set"))
        seeds_total = 0
        seeds_by_kind = Counter()
        for c in deduped:
            extra = c.extra or {}
            starts = extra.get("start_nodes") or [c.node_id]
            seeds_total += len(starts)
            seeds_by_kind[c.kind] += len(starts)
        print("=== criteria stats ===")
        print(f"criteria: {len(deduped)}, set_kinds: {set_cnt}, seed_nodes_total: {seeds_total}")
        print("kinds:", dict(cnt))
        print("seeds_by_kind:", dict(seeds_by_kind))
    except Exception:
        pass

    return deduped
