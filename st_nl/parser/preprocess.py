import re

def preprocess_st(code: str) -> str:
    """
    Preprocess IEC 61131-3 Structured Text (ST) source code.

    Notes:
    - IL is NOT considered/handled here.
    - Removes ST comments: (* ... *), { ... }, // ...
    """

    # 1) Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    # 2) HTML entity unescape
    code = code.replace("&lt;", "<").replace("&gt;", ">")

    # 3) Remove comments (order matters: block first, then line)
    # 3.1 Remove (* ... *) comments (multiline)
    code = re.sub(r"\(\*.*?\*\)", " ", code, flags=re.S)

    # 3.2 Remove { ... } comments (multiline)
    code = re.sub(r"\{.*?\}", " ", code, flags=re.S)

    # 3.3 Remove // ... comments (single line)
    code = re.sub(r"//[^\n]*", " ", code)

    # 4) Fix common malformed ST patterns
    # Empty output parameter: => ,
    code = re.sub(r"=>\s*,", "=> __dummy__,", code)

    # Empty output parameter before ')'
    code = re.sub(r"=>\s*\)", "=> __dummy__ )", code)

    # 5) Cleanup whitespace
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n", "\n", code)

    return code
