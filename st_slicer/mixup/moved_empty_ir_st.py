#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

# =========================
# 你只需要改这里的路径
# =========================
FIXED_DIR = Path(r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode")
JSONL_IN  = FIXED_DIR / "dataset.jsonl"              # 你的现有 jsonl
JSONL_OUT = FIXED_DIR / "dataset_no_empty_ir.jsonl"  # 输出：删掉空 IR 后的新 jsonl
MOVE_DIR  = FIXED_DIR.parent / "moved_empty_ir_st"   # 输出：被移走的 .st 文件放这里
LOG_TXT   = FIXED_DIR / "moved_empty_ir.log.txt"     # 日志

# 空 IR 的判定：优先看 z_vec.n_instr，其次看 n_cfg_nodes（指令级 CFG）
def is_empty_ir(obj: Dict[str, Any]) -> bool:
    z = obj.get("z_vec") or {}
    n_instr = z.get("n_instr", None)
    if isinstance(n_instr, int):
        return n_instr == 0
    # 兜底：有些导出可能没有 z_vec.n_instr
    n_cfg_nodes = obj.get("n_cfg_nodes", None)
    if isinstance(n_cfg_nodes, int):
        return n_cfg_nodes == 0
    # 再兜底：空串 md5（你之前看到的 d41d8...）
    sk = obj.get("skeleton_id", "")
    return sk == "d41d8cd98f00b204e9800998ecf8427e"

def resolve_st_path(source_file: str) -> Path:
    """
    source_file 可能是：
    - 绝对路径（最常见）
    - 相对路径（相对 fixedCode）
    - 只有文件名
    这里统一解析为一个实际存在的 Path（优先 fixedCode 内）。
    """
    p = Path(source_file) if source_file else Path("")
    if p.is_absolute() and p.exists():
        return p

    # 尝试相对 FIXED_DIR
    cand = FIXED_DIR / p
    if cand.exists():
        return cand

    # 只有文件名的情况：在 fixedCode 内递归找
    if p.name:
        hits = list(FIXED_DIR.rglob(p.name))
        if hits:
            return hits[0]

    return cand  # 最后返回一个“可能的位置”，用于日志

def safe_move(src: Path, dst: Path) -> Tuple[bool, str]:
    """
    尝试移动文件：跨盘失败则 copy2 + unlink。
    返回 (是否成功, 失败原因/成功信息)
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move 在跨盘时可能退化为 copy+delete，但有时会失败；这里做兜底
        shutil.move(str(src), str(dst))
        return True, "moved"
    except Exception as e1:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            src.unlink()
            return True, f"copied_then_deleted (move_failed={e1})"
        except Exception as e2:
            return False, f"move_failed={e1}; copy_delete_failed={e2}"

def main():
    if not FIXED_DIR.exists():
        raise SystemExit(f"FIXED_DIR not found: {FIXED_DIR}")
    if not JSONL_IN.exists():
        raise SystemExit(f"JSONL_IN not found: {JSONL_IN}")

    MOVE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Pass 1: 统计每个 source_file 是否存在非空 IR
    # -------------------------
    file_stat = {}  # key: normalized st path str, val: {"has_empty":bool,"has_nonempty":bool,"examples":[...]}
    total = 0
    empty_lines = 0

    with JSONL_IN.open("r", encoding="utf-8", errors="ignore") as rf:
        for line_no, line in enumerate(rf, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                # 坏行直接跳过（也可选择写日志）
                continue

            src_file = str(obj.get("source_file", "")).strip()
            st_path = resolve_st_path(src_file)
            key = str(st_path)

            st = file_stat.setdefault(key, {"has_empty": False, "has_nonempty": False})
            if is_empty_ir(obj):
                st["has_empty"] = True
                empty_lines += 1
            else:
                st["has_nonempty"] = True

    # 需要移动的文件：存在空 IR 且不存在任何非空 IR
    files_to_move = [Path(k) for k, v in file_stat.items() if v["has_empty"] and (not v["has_nonempty"])]

    # -------------------------
    # Pass 2: 写出新 JSONL（删除所有空 IR 行）
    # -------------------------
    kept = 0
    removed = 0
    with JSONL_IN.open("r", encoding="utf-8", errors="ignore") as rf, \
         JSONL_OUT.open("w", encoding="utf-8") as wf:
        for line in rf:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                obj = json.loads(line_stripped)
            except Exception:
                continue

            if is_empty_ir(obj):
                removed += 1
                continue

            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    # -------------------------
    # Pass 3: 移动 ST 文件（并从 fixedCode 中“删除”）
    # -------------------------
    moved_ok = 0
    moved_fail = 0
    log_lines = []
    for src in files_to_move:
        # 只移动 fixedCode 内的文件（避免误动别的路径）
        try:
            src_rel = src.relative_to(FIXED_DIR)
            dst = MOVE_DIR / src_rel
        except Exception:
            # 不在 FIXED_DIR 下面的，记录但不动
            log_lines.append(f"[SKIP_NOT_UNDER_FIXED] {src}")
            continue

        if not src.exists():
            log_lines.append(f"[MISSING] {src}")
            moved_fail += 1
            continue

        ok, msg = safe_move(src, dst)
        if ok:
            moved_ok += 1
            log_lines.append(f"[MOVED] {src_rel} -> {dst} | {msg}")
        else:
            moved_fail += 1
            log_lines.append(f"[MOVE_FAIL] {src_rel} -> {dst} | {msg}")

    # 写日志
    summary = [
        "==== SUMMARY ====",
        f"FIXED_DIR      : {FIXED_DIR}",
        f"JSONL_IN       : {JSONL_IN}",
        f"JSONL_OUT      : {JSONL_OUT}",
        f"MOVE_DIR       : {MOVE_DIR}",
        f"Total jsonl rows read (non-empty lines): {total}",
        f"Empty-IR rows detected               : {empty_lines}",
        f"Rows removed from jsonl              : {removed}",
        f"Rows kept in new jsonl               : {kept}",
        f"ST files to move (file-level)        : {len(files_to_move)}",
        f"ST moved ok                          : {moved_ok}",
        f"ST moved fail/skipped                : {moved_fail}",
        "",
        "==== DETAILS ====",
    ]
    LOG_TXT.write_text("\n".join(summary + log_lines), encoding="utf-8", errors="ignore")

    print("Done.")
    print(f"New JSONL written: {JSONL_OUT}")
    print(f"Moved ST folder  : {MOVE_DIR}")
    print(f"Log written      : {LOG_TXT}")
    print(f"Removed rows     : {removed}")
    print(f"Moved ST ok      : {moved_ok}, fail/skip: {moved_fail}")

if __name__ == "__main__":
    main()
