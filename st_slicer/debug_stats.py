# st_slicer/debug_stats.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import statistics

def _pct(a: int, b: int) -> float:
    return (100.0 * a / b) if b else 0.0

def _union_line_coverage(blocks: Iterable[Any]) -> Set[int]:
    cov: Set[int] = set()
    for b in blocks:
        cov |= set(getattr(b, "line_numbers", []) or [])
    return cov

def _quantiles(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    xs2 = sorted(xs)
    def q(p: float) -> float:
        i = int(p * (len(xs2) - 1))
        return float(xs2[i])
    return {
        "n": len(xs2),
        "min": float(xs2[0]),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "max": float(xs2[-1]),
        "mean": float(statistics.mean(xs2)),
    }

@dataclass
class SliceDiag:
    total_lines: int = 0

    # criteria
    n_criteria: int = 0
    n_criteria_by_kind: Dict[str, int] = field(default_factory=dict)
    n_set_criteria: int = 0
    seed_nodes_total: int = 0
    seed_nodes_by_kind: Dict[str, int] = field(default_factory=dict)

    # slicing
    n_slices: int = 0
    n_empty_slices: int = 0
    slice_node_sizes: List[int] = field(default_factory=list)

    # grouping/clustering
    n_groups: int = 0
    n_clusters: int = 0
    cluster_node_sizes: List[int] = field(default_factory=list)

    # line coverage at checkpoints
    cov_after_slice_lines: Set[int] = field(default_factory=set)   # 只用 node->stmt->lines 的并集（未成块也统计）
    cov_after_blocks_raw: Set[int] = field(default_factory=set)    # cluster->block 初始
    cov_after_stage: Set[int] = field(default_factory=set)
    cov_after_size: Set[int] = field(default_factory=set)
    cov_after_cleanup: Set[int] = field(default_factory=set)
    cov_after_meaningful: Set[int] = field(default_factory=set)
    cov_after_dedup: Set[int] = field(default_factory=set)

    # block counts at checkpoints
    n_blocks_raw: int = 0
    n_blocks_stage: int = 0
    n_blocks_size: int = 0
    n_blocks_cleanup: int = 0
    n_blocks_meaningful: int = 0
    n_blocks_dedup: int = 0

    def report(self) -> str:
        TL = self.total_lines
        lines = []
        lines.append("========== Slice Diagnostics ==========")
        lines.append(f"[A] criteria: {self.n_criteria} (set-kind: {self.n_set_criteria})")
        if self.n_criteria_by_kind:
            top = sorted(self.n_criteria_by_kind.items(), key=lambda x: -x[1])[:12]
            lines.append("    kinds(top): " + ", ".join([f"{k}:{v}" for k, v in top]))
        lines.append(f"    seeds(total node ids): {self.seed_nodes_total}")
        if self.seed_nodes_by_kind:
            top = sorted(self.seed_nodes_by_kind.items(), key=lambda x: -x[1])[:12]
            lines.append("    seeds by kind(top): " + ", ".join([f"{k}:{v}" for k, v in top]))

        lines.append(f"[B] slicing: slices={self.n_slices}, empty={self.n_empty_slices}")
        qs = _quantiles(self.slice_node_sizes)
        lines.append(f"    slice node sizes: {qs}")

        lines.append(f"[C] clustering: groups={self.n_groups}, clusters={self.n_clusters}")
        qs2 = _quantiles(self.cluster_node_sizes)
        lines.append(f"    cluster node sizes: {qs2}")

        # 逐段 coverage 变化
        def cov_line(tag: str, cov: Set[int]) -> None:
            lines.append(f"{tag}: {len(cov)}/{TL} = {_pct(len(cov), TL):.1f}%")

        cov_line("[L1] cov after slice (lines union, pre-block)", self.cov_after_slice_lines)
        cov_line("[L2] cov after block build", self.cov_after_blocks_raw)
        cov_line("[L3] cov after stage split", self.cov_after_stage)
        cov_line("[L4] cov after size norm", self.cov_after_size)
        cov_line("[L5] cov after empty-structure cleanup", self.cov_after_cleanup)
        cov_line("[L6] cov after meaningful filter", self.cov_after_meaningful)
        cov_line("[L7] cov after dedup", self.cov_after_dedup)

        # block 数量
        lines.append(f"[B1] blocks: raw={self.n_blocks_raw}, stage={self.n_blocks_stage}, size={self.n_blocks_size}, "
                     f"cleanup={self.n_blocks_cleanup}, meaningful={self.n_blocks_meaningful}, dedup={self.n_blocks_dedup}")

        return "\n".join(lines)

def diag_enabled() -> bool:
    # 你可以用环境变量控制，这里简单做成常量/开关都行
    return True
