# st_slicer/slicer.py

"""ST slicer core algorithms.

Key feature:
  - Classic PDG backward slice
  - Optional variable-sensitive backward slice (to improve semantic focus)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set, Optional, Tuple, Any
from .pdg.pdg_builder import ProgramDependenceGraph

# def backward_slice(
#     pdg: ProgramDependenceGraph,
#     start_nodes: Iterable[int],
#     use_data: bool = True,
#     use_control: bool = True,
# ) -> Set[int]:
#     """
#     基于 PDG 的静态后向切片，考虑 data + control 两类前驱。
#     """
#     worklist = list(start_nodes)
#     slice_nodes: Set[int] = set()

#     while worklist:
#         n = worklist.pop()
#         if n in slice_nodes:
#             continue
#         slice_nodes.add(n)

#         for pred, edge_type in pdg.predecessors(n):
#             if (edge_type == "data" and not use_data) or \
#                (edge_type == "control" and not use_control):
#                 continue
#             # 目前 data / control 全都要；如果之后想做变量敏感可以在这里收紧
#             if pred not in slice_nodes:
#                 worklist.append(pred)

#     return slice_nodes

# st_slicer/slicer.py

def backward_slice(
    pdg: ProgramDependenceGraph,
    start_nodes: Iterable[int],
    use_data: bool = True,
    use_control: bool = True,
    *,
    var_sensitive: bool = False,
    seed_vars: Optional[Iterable[str]] = None,
    max_var_expansions: int = 10_000,
) -> Set[int]:
    """
    基于 PDG 的静态后向切片，考虑 data + control 两类前驱。

    当 var_sensitive=True 时，会基于“相关变量集合”过滤 data 依赖回溯，
    以减少“节点级全回溯”造成的语义发散。

    兼容性：
    - 若 PDG.predecessors(n) 返回 (pred, edge_type, edge_var)，优先用 edge_var 过滤；
    - 否则尝试从 PDG 上获取 node defs/uses（node_defs/node_uses/defs/uses/node_info/nodes[*].defs 等）；
    - 再不行则自动退化为经典后向切片。
    """

    def _normalize_pred_item(item: Any) -> Tuple[int, str, Optional[str]]:
        """Normalize predecessors() output to (pred_id, edge_type, edge_var?)."""
        if isinstance(item, (tuple, list)):
            if len(item) >= 3:
                return int(item[0]), str(item[1]), (None if item[2] is None else str(item[2]))
            if len(item) == 2:
                return int(item[0]), str(item[1]), None
        raise TypeError(f"Unsupported predecessor item: {item!r}")

    def _get_node_set_attr(node_id: int, key: str) -> Set[str]:
        """Best-effort to retrieve per-node def/use sets from PDG."""
        candidates = []
        if key in ("defs", "uses"):
            candidates.extend([f"node_{key}", key, f"node_{key}s", f"{key}s"])

        for attr in candidates:
            obj = getattr(pdg, attr, None)
            if obj is None:
                continue
            try:
                val = obj.get(node_id) if hasattr(obj, "get") else obj[node_id]
            except Exception:
                continue
            if val is None:
                continue
            try:
                return {str(x) for x in val}
            except Exception:
                continue

        info = getattr(pdg, "node_info", None)
        if info is not None:
            try:
                val = info.get(node_id, {}).get(key)
                if val is not None:
                    return {str(x) for x in val}
            except Exception:
                pass

        nodes = getattr(pdg, "nodes", None)
        if nodes is not None:
            try:
                nobj = nodes.get(node_id) if hasattr(nodes, "get") else nodes[node_id]
                val = getattr(nobj, key, None)
                if val is not None:
                    return {str(x) for x in val}
            except Exception:
                pass

        return set()

    start_nodes = list(start_nodes)
    worklist = list(start_nodes)
    slice_nodes: Set[int] = set()

    # Variable-sensitive bookkeeping
    relevant_vars: Set[str] = set(str(v) for v in seed_vars) if seed_vars else set()
    if var_sensitive and not relevant_vars:
        for s in start_nodes:
            relevant_vars |= _get_node_set_attr(s, "uses")

    # If we still cannot obtain any variable information, degrade to classic slicing
    if var_sensitive and not relevant_vars:
        var_sensitive = False

    while worklist:
        n = worklist.pop()
        if n in slice_nodes:
            continue
        slice_nodes.add(n)

        for item in pdg.predecessors(n,include_var=True):
            try:
                pred, edge_type, edge_var = _normalize_pred_item(item)
            except Exception:
                continue

            if (edge_type == "data" and not use_data) or (edge_type == "control" and not use_control):
                continue

            if not var_sensitive:
                if pred not in slice_nodes:
                    worklist.append(pred)
                continue

            if edge_type == "data":
                # Prefer edge-labeled variable if available
                if edge_var is not None:
                    if edge_var not in relevant_vars:
                        continue
                    if pred not in slice_nodes:
                        worklist.append(pred)
                    relevant_vars |= _get_node_set_attr(pred, "uses")
                    continue

                # Fallback: defs ∩ relevant_vars
                pred_defs = _get_node_set_attr(pred, "defs")
                if pred_defs and pred_defs.isdisjoint(relevant_vars):
                    continue

                if pred not in slice_nodes:
                    worklist.append(pred)
                relevant_vars |= _get_node_set_attr(pred, "uses")
                continue

            if edge_type == "control":
                if pred not in slice_nodes:
                    worklist.append(pred)
                relevant_vars |= _get_node_set_attr(pred, "uses")
                if len(relevant_vars) > max_var_expansions:
                    var_sensitive = False  # safety valve
                continue

    return slice_nodes

