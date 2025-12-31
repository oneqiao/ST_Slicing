from __future__ import annotations

import json
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set

# =========================
# 配置
# =========================
META_JSONL = r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode\meta2.jsonl"
OUT_PAIRS  = r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode\pairs2_ModeA3.jsonl"

MODE = "A3"  # "A1" | "A2" | "A3"
K_PAIRS_PER_SAMPLE = 3
KNN_TOPK = 20
BETA_ALPHA = 2.0
SEED = 20251230
TOP_N_LARGEST_BUCKETS = 20

# 若 MODE=A3 是否要求 meta 必须包含 bucket_z
REQUIRE_BUCKET_Z_FOR_A3 = True

Z_KEYS = [
    "n_instr", "n_cfg_edges", "n_branch", "n_backedge",
    "n_pdg_data_edges", "n_pdg_ctrl_edges",
    "n_def_sites", "n_use_sites", "n_unique_vars_prog",
]


def load_meta(meta_path: str) -> List[dict]:
    rows: List[dict] = []
    missing_bucket_z = 0

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if not obj.get("sample_id"):
                continue
            if "z_vec" not in obj:
                continue

            # skeleton_id_l2 / l1 回退逻辑保留
            if not obj.get("skeleton_id_l2"):
                obj["skeleton_id_l2"] = obj.get("skeleton_id")
            if not obj.get("skeleton_id_l1"):
                obj["skeleton_id_l1"] = obj.get("skeleton_id")

            if not obj.get("skeleton_id_l2"):
                continue

            if not obj.get("bucket_z"):
                missing_bucket_z += 1

            rows.append(obj)

    # 严格模式：A3 期望使用 bucket_z，但 meta 缺失就直接停止
    if MODE == "A3" and REQUIRE_BUCKET_Z_FOR_A3 and missing_bucket_z > 0:
        raise SystemExit(
            f"[FATAL] META missing bucket_z for {missing_bucket_z}/{len(rows)} samples. "
            f"Do NOT fallback to a single 'NO_BUCKET_Z' bucket. "
            f"Please regenerate meta with bucket_z (e.g., meta2.jsonl) or set REQUIRE_BUCKET_Z_FOR_A3=False."
        )

    return rows


def compute_z_stats(rows: List[dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean = {k: 0.0 for k in Z_KEYS}
    var = {k: 0.0 for k in Z_KEYS}
    n = len(rows)
    if n == 0:
        return mean, {k: 1.0 for k in Z_KEYS}

    for r in rows:
        z = r["z_vec"]
        for k in Z_KEYS:
            mean[k] += float(z.get(k, 0.0))
    for k in Z_KEYS:
        mean[k] /= n

    for r in rows:
        z = r["z_vec"]
        for k in Z_KEYS:
            d = float(z.get(k, 0.0)) - mean[k]
            var[k] += d * d
    for k in Z_KEYS:
        var[k] = var[k] / max(1, (n - 1))

    std = {k: math.sqrt(var[k]) if var[k] > 1e-12 else 1.0 for k in Z_KEYS}
    return mean, std


def z_distance(a: dict, b: dict, mean: Dict[str, float], std: Dict[str, float]) -> float:
    za = a["z_vec"]
    zb = b["z_vec"]
    s = 0.0
    for k in Z_KEYS:
        va = (float(za.get(k, 0.0)) - mean[k]) / std[k]
        vb = (float(zb.get(k, 0.0)) - mean[k]) / std[k]
        d = va - vb
        s += d * d
    return math.sqrt(s)


def build_knn_index(rows: List[dict], mean: Dict[str, float], std: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    for i, a in enumerate(rows):
        a_id = a["sample_id"]
        dists: List[Tuple[str, float]] = []
        for j, b in enumerate(rows):
            if i == j:
                continue
            dist = z_distance(a, b, mean, std)
            dists.append((b["sample_id"], dist))
        dists.sort(key=lambda x: x[1])
        out[a_id] = dists
    return out


def print_bucket_stats(title: str, buckets: Dict[str, List[str]], total_samples: int) -> None:
    sizes = [len(v) for v in buckets.values()]
    if not sizes:
        print(f"\n[{title}] No buckets.")
        return
    cnt = Counter(sizes)
    n_buckets = len(sizes)
    singleton_buckets = cnt.get(1, 0)
    singleton_samples = singleton_buckets  # size=1 的桶，每桶 1 个样本
    max_size = max(sizes)
    min_size = min(sizes)
    avg_size = sum(sizes) / n_buckets

    print("\n" + "=" * 80)
    print(f"[{title}] bucket size distribution")
    print(f"Total samples                : {total_samples}")
    print(f"Total buckets                : {n_buckets}")
    print(f"Bucket size (min/avg/max)    : {min_size} / {avg_size:.3f} / {max_size}")
    print(f"Singleton samples (size=1)   : {singleton_samples} ({singleton_samples/total_samples:.2%})")

    largest = sorted(((sid, len(v)) for sid, v in buckets.items()),
                     key=lambda x: x[1], reverse=True)[:max(1, TOP_N_LARGEST_BUCKETS)]
    print(f"\n[Top {len(largest)} Largest Buckets] (bucket_id, size)")
    for sid, sz in largest:
        print(f"  {sid}\t{sz}")
    print("=" * 80 + "\n")


def main():
    rng = random.Random(SEED)
    rows = load_meta(META_JSONL)
    if not rows:
        raise SystemExit(f"meta.jsonl is empty or invalid: {META_JSONL}")

    by_id: Dict[str, dict] = {r["sample_id"]: r for r in rows}

    # three layers of buckets
    buckets_l2: Dict[str, List[str]] = defaultdict(list)
    buckets_l1: Dict[str, List[str]] = defaultdict(list)
    buckets_z:  Dict[str, List[str]] = defaultdict(list)

    for r in rows:
        buckets_l2[r["skeleton_id_l2"]].append(r["sample_id"])
        buckets_l1[r["skeleton_id_l1"]].append(r["sample_id"])
        # bucket_z 可能不存在（若 REQUIRE_BUCKET_Z_FOR_A3=False），则跳过该层
        if r.get("bucket_z") is not None:
            buckets_z[r["bucket_z"]].append(r["sample_id"])

    print_bucket_stats("L2(skeleton_id_l2)", buckets_l2, total_samples=len(rows))
    print_bucket_stats("L1(skeleton_id_l1)", buckets_l1, total_samples=len(rows))
    if buckets_z:
        print_bucket_stats("Z(bucket_z)", buckets_z, total_samples=len(rows))
    else:
        print("[Z(bucket_z)] skipped because bucket_z is missing.\n")

    mean, std = compute_z_stats(rows)
    knn = build_knn_index(rows, mean, std) if MODE in ("A2", "A3") else None

    def _sample_from_bucket(a_id: str, bucket_map: Dict[str, List[str]], key_field: str,
                            used: Set[str]) -> Tuple[Optional[str], str]:
        bid = by_id[a_id].get(key_field)
        if bid is None:
            return None, "NO_BUCKET_KEY"
        cand = bucket_map.get(bid, [])
        if len(cand) <= 1:
            return None, "NO_BUCKET_MATE"

        # 去重采样：优先抽没用过的 b
        cand2 = [x for x in cand if x != a_id and x not in used]
        if cand2:
            b_id = rng.choice(cand2)
            return b_id, f"SAME_{key_field}"

        # 候选都用过了：允许重复，但仍避免 self
        cand3 = [x for x in cand if x != a_id]
        if not cand3:
            return None, "NO_BUCKET_MATE"
        b_id = rng.choice(cand3)
        return b_id, f"SAME_{key_field}_REUSE"

    def _sample_from_knn(a_id: str, used: Set[str]) -> Tuple[Optional[str], str]:
        assert knn is not None
        neigh = knn.get(a_id, [])
        if not neigh:
            return None, "NO_KNN"
        top = neigh[:max(1, KNN_TOPK)]

        top2 = [nid for (nid, _) in top if nid not in used and nid != a_id]
        if top2:
            return rng.choice(top2), "KNN_FALLBACK"

        top3 = [nid for (nid, _) in top if nid != a_id]
        if not top3:
            return None, "NO_KNN"
        return rng.choice(top3), "KNN_FALLBACK_REUSE"

    n_pairs = 0
    reason_cnt = Counter()

    with open(OUT_PAIRS, "w", encoding="utf-8") as wf:
        for a in rows:
            a_id = a["sample_id"]
            used_b: Set[str] = set()

            for _ in range(K_PAIRS_PER_SAMPLE):
                b_id: Optional[str] = None
                reason = ""

                if MODE == "A1":
                    b_id, reason = _sample_from_bucket(a_id, buckets_l2, "skeleton_id_l2", used_b)

                elif MODE == "A2":
                    b_id, reason = _sample_from_knn(a_id, used_b)

                elif MODE == "A3":
                    # 分层桶兜底：L2 -> L1 -> Z -> KNN
                    b_id, reason = _sample_from_bucket(a_id, buckets_l2, "skeleton_id_l2", used_b)
                    if not b_id:
                        b_id, reason = _sample_from_bucket(a_id, buckets_l1, "skeleton_id_l1", used_b)
                    if not b_id and buckets_z:
                        b_id, reason = _sample_from_bucket(a_id, buckets_z, "bucket_z", used_b)
                    if not b_id:
                        b_id, reason = _sample_from_knn(a_id, used_b)

                else:
                    raise SystemExit(f"Unknown MODE: {MODE}")

                if not b_id:
                    continue

                used_b.add(b_id)
                lam = rng.betavariate(BETA_ALPHA, BETA_ALPHA)

                rec = {
                    "a_id": a_id,
                    "b_id": b_id,
                    "lambda": lam,
                    "mode": MODE,
                    "reason": reason,
                    "a_skeleton_id_l2": by_id[a_id].get("skeleton_id_l2"),
                    "b_skeleton_id_l2": by_id[b_id].get("skeleton_id_l2"),
                    "a_bucket_z": by_id[a_id].get("bucket_z"),
                    "b_bucket_z": by_id[b_id].get("bucket_z"),
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
                reason_cnt[reason] += 1

    print("Done.")
    print(f"META_JSONL           : {META_JSONL}")
    print(f"OUT_PAIRS            : {OUT_PAIRS}")
    print(f"MODE                 : {MODE}")
    print(f"K_PAIRS_PER_SAMPLE   : {K_PAIRS_PER_SAMPLE}")
    print(f"KNN_TOPK             : {KNN_TOPK}")
    print(f"BETA_ALPHA           : {BETA_ALPHA}")
    print(f"Total meta samples   : {len(rows)}")
    print(f"Total pairs written  : {n_pairs}")
    print("\n[Pair Reasons]")
    for k, v in reason_cnt.most_common():
        print(f"  {k:>24} : {v} ({v/max(1,n_pairs):.2%})")


if __name__ == "__main__":
    main()
