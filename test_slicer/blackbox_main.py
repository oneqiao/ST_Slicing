# tests/blackbox_main.py
from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict
import re

# 你已有的 IF/CASE 补全（必有）
from st_slicer.blocks.structure import patch_if_structure, patch_case_structure

# 循环补全：可能有也可能没有（黑盒测试要兼容两种情况）
try:
    from st_slicer.blocks.structure import patch_for_structure
except Exception:
    patch_for_structure = None

try:
    from st_slicer.blocks.structure import patch_while_structure
except Exception:
    patch_while_structure = None

try:
    from st_slicer.blocks.structure import patch_repeat_structure
except Exception:
    patch_repeat_structure = None


# -------------------------
# Helpers: simple ST scanning
# -------------------------

RE_FOR = re.compile(r"^\s*FOR\b", re.IGNORECASE)
RE_END_FOR = re.compile(r"^\s*END_FOR\b", re.IGNORECASE)

RE_WHILE = re.compile(r"^\s*WHILE\b", re.IGNORECASE)
RE_END_WHILE = re.compile(r"^\s*END_WHILE\b", re.IGNORECASE)

RE_REPEAT = re.compile(r"^\s*REPEAT\b", re.IGNORECASE)
RE_UNTIL = re.compile(r"^\s*UNTIL\b", re.IGNORECASE)
RE_END_REPEAT = re.compile(r"^\s*END_REPEAT\b", re.IGNORECASE)

RE_ASSIGN = re.compile(r":=")

def clean(line: str) -> str:
    """黑盒足够的清理：去两端空白。"""
    return line.strip()

def find_matching_end(lines: List[str], start_ln: int, head_re: re.Pattern, end_re: re.Pattern) -> Optional[int]:
    """
    在同类嵌套结构下，从 start_ln 往下找匹配的 END_*，返回行号（1-based）。
    """
    depth = 0
    n = len(lines)
    for ln in range(start_ln, n + 1):
        t = clean(lines[ln - 1])
        if not t:
            continue
        if head_re.search(t):
            depth += 1
            continue
        if end_re.search(t):
            depth -= 1
            if depth == 0:
                return ln
    return None

def find_body_seed_line(lines: List[str], head_ln: int, end_ln: int, prefer_assign: bool = True) -> Optional[int]:
    """
    在结构体内部找一行作为 base_lines 的 seed：
      - 优先找 ':='
      - 否则找第一行非空、非注释
    """
    for ln in range(head_ln + 1, end_ln):
        t = clean(lines[ln - 1])
        if not t:
            continue
        if prefer_assign and RE_ASSIGN.search(t):
            return ln
    for ln in range(head_ln + 1, end_ln):
        t = clean(lines[ln - 1])
        if not t:
            continue
        return ln
    return None

def collect_loop_seeds(code_lines: List[str]) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    返回：
      {
        "FOR":    [(head_ln, seed_ln, end_ln), ...],
        "WHILE":  [...],
        "REPEAT": [...],
      }
    """
    seeds: Dict[str, List[Tuple[int, int, int]]] = {"FOR": [], "WHILE": [], "REPEAT": []}
    n = len(code_lines)

    # FOR
    for ln in range(1, n + 1):
        t = clean(code_lines[ln - 1])
        if RE_FOR.search(t):
            end_ln = find_matching_end(code_lines, ln, RE_FOR, RE_END_FOR)
            if not end_ln:
                continue
            seed_ln = find_body_seed_line(code_lines, ln, end_ln)
            if seed_ln:
                seeds["FOR"].append((ln, seed_ln, end_ln))

    # WHILE
    for ln in range(1, n + 1):
        t = clean(code_lines[ln - 1])
        if RE_WHILE.search(t):
            end_ln = find_matching_end(code_lines, ln, RE_WHILE, RE_END_WHILE)
            if not end_ln:
                continue
            seed_ln = find_body_seed_line(code_lines, ln, end_ln)
            if seed_ln:
                seeds["WHILE"].append((ln, seed_ln, end_ln))

    # REPEAT（UNTIL 在尾部，END_REPEAT 结束）
    for ln in range(1, n + 1):
        t = clean(code_lines[ln - 1])
        if RE_REPEAT.search(t):
            end_ln = find_matching_end(code_lines, ln, RE_REPEAT, RE_END_REPEAT)
            if not end_ln:
                continue
            # seed 取 REPEAT..END_REPEAT 内的赋值行
            seed_ln = find_body_seed_line(code_lines, ln, end_ln)
            if seed_ln:
                seeds["REPEAT"].append((ln, seed_ln, end_ln))

    return seeds

def assert_closed(text: str) -> None:
    """
    最小闭合断言：出现 head 必须出现 end（黑盒能证明补全是否生效）
    """
    up = text.upper()
    if "FOR" in up and "END_FOR" not in up:
        raise AssertionError("FOR appears but END_FOR missing")
    if "WHILE" in up and "END_WHILE" not in up:
        raise AssertionError("WHILE appears but END_WHILE missing")
    if "REPEAT" in up and "END_REPEAT" not in up:
        raise AssertionError("REPEAT appears but END_REPEAT missing")
    # REPEAT 推荐也检查 UNTIL
    if "REPEAT" in up and "UNTIL" not in up:
        raise AssertionError("REPEAT appears but UNTIL missing")


# -------------------------
# Black-box test main
# -------------------------

def run_one_case(code_lines: List[str], seed_ln: int) -> None:
    base_lines: Set[int] = {seed_ln}

    fixed = patch_if_structure(base_lines, code_lines, ensure_end_if=True)
    fixed = patch_case_structure(fixed, code_lines, ensure_end_case=True, include_branch_headers=True)

    # 循环补全（如果你已经实现）
    if patch_for_structure is not None:
        fixed = patch_for_structure(fixed, code_lines, include_header_span=True)  # 若你函数签名不同，改这里
    else:
        print("[WARN] patch_for_structure not found, skip FOR completion")

    if patch_while_structure is not None:
        fixed = patch_while_structure(fixed, code_lines, include_header_span=True)
    else:
        print("[WARN] patch_while_structure not found, skip WHILE completion")

    if patch_repeat_structure is not None:
        fixed = patch_repeat_structure(fixed, code_lines, include_until_span=True)
    else:
        print("[WARN] patch_repeat_structure not found, skip REPEAT completion")

    base_sorted = sorted(base_lines)
    fixed_sorted = sorted(fixed)
    added = [ln for ln in fixed_sorted if ln not in base_lines]

    print("\n[BLACKBOX] base_lines:", base_sorted)
    print("[BLACKBOX] fixed_lines:", fixed_sorted)
    print("[BLACKBOX] added_lines:", added)

    # 打印 fixed 内容
    print("\n----- FIXED SNIPPET -----")
    for ln in fixed_sorted:
        if 1 <= ln <= len(code_lines):
            print(f"{ln:4d}: {code_lines[ln-1].rstrip()}")
    print("-------------------------")

    # 闭合断言
    fixed_text = "\n".join(code_lines[ln - 1] for ln in fixed_sorted if 1 <= ln <= len(code_lines))
    assert_closed(fixed_text)
    print("[BLACKBOX] PASS: closure assertions satisfied")

def main():
    # 改成你要测试的 ST 文件
    code_path = Path(__file__).parent / "loop_min.st"
    code = code_path.read_text(encoding="utf-8")
    code_lines = code.splitlines()

    seeds = collect_loop_seeds(code_lines)
    print("Found loop seeds:")
    for k, items in seeds.items():
        print(f"  {k}: {len(items)}")

    # 优先测到一个 FOR/WHILE/REPEAT；如果文件没有循环，就提示
    picked: Optional[Tuple[str, int, int, int]] = None
    for kind in ("FOR", "WHILE", "REPEAT"):
        if seeds[kind]:
            head_ln, seed_ln, end_ln = seeds[kind][0]
            picked = (kind, head_ln, seed_ln, end_ln)
            break

    if not picked:
        print("[BLACKBOX] No FOR/WHILE/REPEAT found in this file. "
              "Please test with a file containing loops (or add a small loop snippet).")
        return

    kind, head_ln, seed_ln, end_ln = picked
    print(f"\n[BLACKBOX] Testing kind={kind} head={head_ln} seed(body)={seed_ln} end={end_ln}")
    run_one_case(code_lines, seed_ln)

if __name__ == "__main__":
    main()

