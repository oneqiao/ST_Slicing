#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from pathlib import Path

# 1) 关键字统一大写（最小集合，后续你可扩充）
KW = [
    "PROGRAM","END_PROGRAM","FUNCTION_BLOCK","END_FUNCTION_BLOCK","FUNCTION","END_FUNCTION",
    "VAR","VAR_INPUT","VAR_OUTPUT","VAR_IN_OUT","END_VAR",
    "IF","THEN","ELSIF","ELSE","END_IF",
    "FOR","TO","DO","END_FOR",
    "WHILE","END_WHILE",
    "REPEAT","UNTIL","END_REPEAT",
    "CASE","OF","END_CASE",
    "NOT","AND","OR","XOR",
    "RETURN","EXIT",
    "TRUE","FALSE",
    # 常用基本类型（你贴的 REAL/BOOL/INT/USINT/UINT/DINT 等）
    "BOOL","INT","UINT","DINT","UDINT","SINT","USINT","REAL","LREAL","STRING","WSTRING"
]
KW_RE = re.compile(r"\b(" + "|".join(KW) + r")\b", re.IGNORECASE)

def uppercase_keywords(text: str) -> str:
    # 简化版：直接替换（不处理字符串/注释隔离）
    # 如果你数据里字符串常量很多，再做“跳过字符串/注释”的增强版
    return KW_RE.sub(lambda m: m.group(1).upper(), text)

# 2) 删除空输出参数：把 "Name => ," 或 "Name => )" 这种删掉
EMPTY_OUT_1 = re.compile(r"\b([A-Za-z_]\w*)\s*=>\s*,")
EMPTY_OUT_2 = re.compile(r",\s*\b([A-Za-z_]\w*)\s*=>\s*\)")

# 额外：清理多余逗号（drop empty out param 后常见）
_COMMA_FIX_1 = re.compile(r"\(\s*,")     # "(," -> "("
_COMMA_FIX_2 = re.compile(r",\s*\)")     # ",)" -> ")"
_MULTI_COMMA = re.compile(r",\s*,+")    # ", ," -> ","

def drop_empty_out_params(text: str) -> str:
    # 反复清理，直到不再变化（处理连续空参数的情况）
    while True:
        old = text
        text = EMPTY_OUT_1.sub("", text)
        text = EMPTY_OUT_2.sub(")", text)
        # 处理 "Name => )" 这种没有前置逗号的情况
        text = re.sub(r"\b([A-Za-z_]\w*)\s*=>\s*\)", ")", text)

        # 清理调用参数列表中可能出现的 "(," ",)" ",,"
        text = _COMMA_FIX_1.sub("(", text)
        text = _COMMA_FIX_2.sub(")", text)
        text = _MULTI_COMMA.sub(",", text)

        if text == old:
            break
    return text

def clean_st_text(text: str) -> str:
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) HTML 实体
    text = text.replace("&lt;", "<").replace("&gt;", ">")

    # 2) 删 // 行注释
    text = strip_slashslash_comments(text)

    # 3) 修连续分号
    text = normalize_semicolons(text)

    # 4) 关键字大写（在去注释之后做更安全）
    text = uppercase_keywords(text)

    # 5) 删除空输出参数
    text = drop_empty_out_params(text)

    return text

# =========================
# 配置：直接在这里改路径即可
# =========================
IN_DIR = Path(r"F:\study\postgtaduate\AIPython\st_code\Mix_code\yuanCode")     # 输入目录
OUT_DIR = Path(r"F:\study\postgtaduate\AIPython\st_code\Mix_code\fixedCode") # 输出目录
EXT = ".st"  # 只处理这种后缀（不区分大小写）

# 记录无 POU header 的文件清单
SKIPPED_TXT = OUT_DIR / "skipped.txt"


# =========================
# 正则：匹配 POU 头
# =========================
POU_HEADER_RE = re.compile(
    r"^[ \t]*(PROGRAM|FUNCTION_BLOCK|FUNCTION)\b.*$",
    re.IGNORECASE | re.MULTILINE
)

def find_first_pou_header(text: str) -> re.Match | None:
    """返回第一个 POU header 的 match；找不到则返回 None"""
    return POU_HEADER_RE.search(text)


def move_header_to_first_line(text: str, header_match: re.Match) -> str:
    """
    把匹配到的 POU header 行移动到第一行：
    - 将 header 之前的所有内容挪到 header 后面
    """
    header_start = header_match.start()
    header_end = header_match.end()

    prefix = text[:header_start]
    header_line = text[header_start:header_end]
    suffix = text[header_end:]

    if prefix.strip().lstrip("\ufeff") == "":
        return header_line.strip("\r\n") + "\n" + suffix.lstrip("\r\n")

    prefix_clean = prefix.lstrip("\ufeff").strip("\r\n")
    suffix_clean = suffix.lstrip("\r\n")

    out = header_line.strip("\r\n") + "\n"
    if prefix_clean:
        out += prefix_clean + "\n"
    out += suffix_clean
    return out

def strip_slashslash_comments(text: str) -> str:
    """
    删除 // 行注释（尽量避免误删字符串内的 //）
    简化假设：你的 ST 代码里字符串较少；若字符串很多，可再增强。
    """
    out_lines = []
    for line in text.splitlines():
        # 粗略处理：如果这一行包含单引号/双引号，保守起见不做截断（避免误删字符串）
        if "'" in line or '"' in line:
            out_lines.append(line)
            continue
        pos = line.find("//")
        if pos >= 0:
            out_lines.append(line[:pos].rstrip())
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")

_SEMI_RE = re.compile(r";{2,}")

def normalize_semicolons(text: str) -> str:
    """把连续多个分号折叠为一个分号。"""
    return _SEMI_RE.sub(";", text)

def process_st_text(text: str) -> tuple[str, bool]:
    text = clean_st_text(text)

    m = find_first_pou_header(text)
    if not m:
        return text, False

    text = move_header_to_first_line(text, m)
    return text, True

def safe_delete(path: Path) -> bool:
    """删除文件；成功返回 True，不存在返回 False。"""
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def cleanup_empty_dirs(base_dir: Path) -> None:
    """从最深层开始清理空目录（不会删 base_dir 本身）。"""
    # 自底向上遍历
    for root, dirs, files in os.walk(base_dir, topdown=False):
        root_path = Path(root)
        # 跳过 base_dir 本身
        if root_path == base_dir:
            continue
        # 如果目录为空，删除
        if not any(root_path.iterdir()):
            try:
                root_path.rmdir()
            except Exception:
                pass


def main():
    if not IN_DIR.exists():
        raise SystemExit(f"Input directory not found: {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    total = 0
    written = 0
    deleted = 0

    for root, _, files in os.walk(IN_DIR):
        root_path = Path(root)
        for fn in files:
            if not fn.lower().endswith(EXT.lower()):
                continue

            total += 1
            in_path = root_path / fn
            rel = in_path.relative_to(IN_DIR)
            out_path = OUT_DIR / rel

            raw = in_path.read_text(encoding="utf-8", errors="ignore")
            fixed, has_header = process_st_text(raw)

            if not has_header:
                # 记录并删除 OUT_DIR 中对应文件（如果存在）
                skipped.append(str(rel))
                if safe_delete(out_path):
                    deleted += 1
                continue  # 不写入 OUT_DIR

            # 有 header：正常写入
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(fixed, encoding="utf-8", errors="ignore")
            written += 1

    # 写 skipped.txt
    SKIPPED_TXT.write_text("\n".join(skipped), encoding="utf-8", errors="ignore")

    # 清理空目录（因为我们删/不写某些文件后，可能留下空目录）
    cleanup_empty_dirs(OUT_DIR)

    print("Done.")
    print(f"Input dir : {IN_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Total .st files found     : {total}")
    print(f"Files written to OUT_DIR  : {written}")
    print(f"Files deleted from OUT_DIR: {deleted}")
    print(f"Skipped list written to   : {SKIPPED_TXT}")


if __name__ == "__main__":
    main()
