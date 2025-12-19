# st_slicer/policy.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import json


# =========================
# Schema
# =========================

@dataclass
class SlicePolicy:
    slice_strategy: Optional[str] = None         # "backward" / "backward_multi" / "region"
    var_sensitive: Optional[bool] = None
    seed_vars: Optional[List[str]] = None
    use_data: Optional[bool] = None
    use_control: Optional[bool] = None


@dataclass
class PostPolicy:
    # 这里预留：后处理策略，比如结构补全、空壳清理、规范化等
    patch_if: Optional[bool] = None
    patch_case: Optional[bool] = None
    patch_loop: Optional[bool] = None


@dataclass
class ResolvedPolicy:
    slice: SlicePolicy = field(default_factory=SlicePolicy)
    post: PostPolicy = field(default_factory=PostPolicy)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    """从 policy.json 读出来的结构。"""
    kind_policy: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PolicyModelContext:
    """可学习策略的上下文（以后扩展用）。"""
    pou_name: Optional[str] = None
    file: Optional[str] = None


class PolicyModel:
    """可学习策略接口（你暂时可以不实现）。"""
    def predict(self, kind: str, criterion_extra: Dict[str, Any], ctx: PolicyModelContext) -> Optional[Dict[str, Any]]:
        return None


# =========================
# Hardcoded registry (baseline)
# =========================
# 你可以只放你最需要的 kind；没有命中的 kind 会返回空 dict
DEFAULT_KIND_POLICY: Dict[str, Dict[str, Any]] = {
    # 示例：对控制结构/循环结构优先 region slice
    "control_region": {"slice": {"slice_strategy": "region", "use_control": True, "use_data": True}},
    "loop_region":    {"slice": {"slice_strategy": "region", "use_control": True, "use_data": True}},

    # 示例：对 *_set 默认 backward_multi
    "io_output_set":         {"slice": {"slice_strategy": "backward_multi"}},
    "state_transition_set":  {"slice": {"slice_strategy": "backward_multi"}},
    "error_logic_set":       {"slice": {"slice_strategy": "backward_multi"}},
    "api_call_set":          {"slice": {"slice_strategy": "backward_multi"}},
}


# =========================
# Utils
# =========================

def base_kind(kind: str) -> str:
    """把 io_output_set -> io_output，方便 fallback。"""
    if kind and kind.endswith("_set"):
        return kind[:-4]
    return kind


def deep_merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：b 覆盖 a。"""
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def get_hardcoded_policy_for_kind(kind: str) -> Dict[str, Any]:
    if not kind:
        return {}
    return DEFAULT_KIND_POLICY.get(kind) or DEFAULT_KIND_POLICY.get(base_kind(kind)) or {}


def _dict_to_slice_policy(d: Dict[str, Any]) -> SlicePolicy:
    d = d or {}
    return SlicePolicy(
        slice_strategy=d.get("slice_strategy"),
        var_sensitive=d.get("var_sensitive"),
        seed_vars=d.get("seed_vars"),
        use_data=d.get("use_data"),
        use_control=d.get("use_control"),
    )


def _dict_to_post_policy(d: Dict[str, Any]) -> PostPolicy:
    d = d or {}
    return PostPolicy(
        patch_if=d.get("patch_if"),
        patch_case=d.get("patch_case"),
        patch_loop=d.get("patch_loop"),
    )


# =========================
# Config loader (optional)
# =========================

def load_policy_config(path: Optional[Union[str, Path]]) -> Optional[PolicyConfig]:
    """
    配置是可选的：
    - path=None 或不存在：返回 None
    - 存在：读取 JSON，返回 PolicyConfig
    JSON 格式示例：
    {
      "kind_policy": {
        "io_output_set": { "slice": { "use_control": false } }
      }
    }
    """
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        return None

    data = json.loads(p.read_text(encoding="utf-8"))
    kind_policy = data.get("kind_policy") or {}
    if not isinstance(kind_policy, dict):
        kind_policy = {}
    return PolicyConfig(kind_policy=kind_policy)


# =========================
# Resolver
# =========================

@dataclass
class PolicyResolver:
    """
    优先级链：
      1) hardcoded registry (baseline)
      2) config override (可选)
      3) model override (可选)
      4) criterion.extra override（最强覆盖）
    """
    config: Optional[PolicyConfig] = None
    model: Optional[PolicyModel] = None

    def resolve(
        self,
        kind: str,
        criterion_extra: Optional[Dict[str, Any]] = None,
        ctx: Optional[PolicyModelContext] = None,
    ) -> ResolvedPolicy:
        criterion_extra = criterion_extra or {}

        # 1) hardcoded
        merged: Dict[str, Any] = get_hardcoded_policy_for_kind(kind) or {}

        # 2) config override
        if self.config is not None:
            kp = self.config.kind_policy.get(kind) or self.config.kind_policy.get(base_kind(kind))
            if isinstance(kp, dict) and kp:
                merged = deep_merge_dict(merged, kp)

        # 3) model override
        if self.model is not None:
            pred = self.model.predict(kind, criterion_extra, ctx or PolicyModelContext())
            if isinstance(pred, dict) and pred:
                merged = deep_merge_dict(merged, pred)

        # 4) criterion.extra override（最高）
        merged = deep_merge_dict(merged, criterion_extra)

        slice_dict = merged.get("slice") or {}
        post_dict = merged.get("post") or {}

        return ResolvedPolicy(
            slice=_dict_to_slice_policy(slice_dict),
            post=_dict_to_post_policy(post_dict),
            extra={k: v for k, v in merged.items() if k not in ("slice", "post")},
        )

    def resolve_slice_overrides(
        self,
        kind: str,
        criterion_extra: Optional[Dict[str, Any]] = None,
        ctx: Optional[PolicyModelContext] = None,
    ) -> Dict[str, Any]:
        """
        给你一个“直接能塞回 crit.extra 用”的扁平 dict（只输出 slice 相关字段）。
        """
        rp = self.resolve(kind, criterion_extra=criterion_extra, ctx=ctx)
        d = asdict(rp.slice)
        return {k: v for k, v in d.items() if v is not None}
