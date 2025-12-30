# 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# =========================
# 直接在这里改路径
# =========================
FAILED_TXT = Path(r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode\failed_parse.txt")
FIXED_DIR  = Path(r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode")

# 删除日志（可选）
DELETED_LOG = FIXED_DIR / "deleted_by_failed_parse.txt"
NOTFOUND_LOG = FIXED_DIR / "notfound_by_failed_parse.txt"


def parse_failed_names(failed_txt: Path) -> set[str]:
    """
    从 failed_parse.txt 每行提取文件名。
    兼容格式：
      xxx.st\tPARSE_...
      xxx.st  PARSE_...
      xxx.st
    """
    names: set[str] = set()
    if not failed_txt.exists():
        raise FileNotFoundError(f"failed_parse.txt not found: {failed_txt}")

    for line in failed_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        # 取第一个字段作为文件名（以 tab 或空格分隔）
        fname = line.split("\t", 1)[0].split(" ", 1)[0].strip()
        if fname.lower().endswith(".st"):
            names.add(fname)
    return names


def delete_matching_files(fixed_dir: Path, failed_names: set[str]) -> tuple[list[str], list[str]]:
    """
    在 fixed_dir 下递归删除所有文件名在 failed_names 中的 .st 文件。
    返回 (deleted_paths, not_found_names)。
    """
    # 建立 fixed_dir 下 “文件名 -> 绝对路径列表” 的索引
    index: dict[str, list[Path]] = {}
    for p in fixed_dir.rglob("*.st"):
        index.setdefault(p.name, []).append(p)

    deleted: list[str] = []
    not_found: list[str] = []

    for name in sorted(failed_names):
        paths = index.get(name, [])
        if not paths:
            not_found.append(name)
            continue

        for fp in paths:
            try:
                fp.unlink()
                deleted.append(str(fp))
            except Exception as e:
                # 删除失败也记录一下
                deleted.append(f"{fp}\tDELETE_FAILED: {e}")

    return deleted, not_found


def main():
    if not FIXED_DIR.exists():
        raise FileNotFoundError(f"fixedCode dir not found: {FIXED_DIR}")

    failed_names = parse_failed_names(FAILED_TXT)
    deleted, not_found = delete_matching_files(FIXED_DIR, failed_names)

    # 写日志
    DELETED_LOG.write_text("\n".join(deleted), encoding="utf-8", errors="ignore")
    NOTFOUND_LOG.write_text("\n".join(not_found), encoding="utf-8", errors="ignore")

    print("Done.")
    print(f"failed_parse.txt      : {FAILED_TXT}")
    print(f"fixedCode dir         : {FIXED_DIR}")
    print(f"Failed names loaded   : {len(failed_names)}")
    print(f"Files deleted entries : {len(deleted)}")
    print(f"Names not found       : {len(not_found)}")
    print(f"Deleted log           : {DELETED_LOG}")
    print(f"Not-found log         : {NOTFOUND_LOG}")


if __name__ == "__main__":
    main()
