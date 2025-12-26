#彻底禁止“随手写字符串”，后续可做模板替换 / 多语言 / paraphrase，训练数据风格稳定
#st_nl/nl/templates.py
def tpl_call_function(name: str, outs: str, ins: str) -> str:
    return f"Call function {name}: {outs} <- {name}({ins})"

def tpl_call_fb(name: str, outs: str, ins: str) -> str:
    return f"Call FB {name}: {outs} <- {name}({ins})"

def tpl_assign(lhs: str, rhs: str) -> str:
    return f"Assign {lhs} = {rhs}"

def tpl_if(cond: str) -> str:
    return f"IF {cond} THEN"

def tpl_elsif(cond: str) -> str:
    return f"ELSIF {cond} THEN"

def tpl_else() -> str:
    return "ELSE"

def tpl_end_if() -> str:
    return "END_IF"

def tpl_case(sel: str) -> str:
    return f"CASE {sel} OF"

def tpl_when(conds: str) -> str:
    return f"WHEN {conds}:"

def tpl_end_case() -> str:
    return "END_CASE"

def tpl_for(var: str, start: str, end: str, step: str) -> str:
    return f"FOR {var} = {start} TO {end} BY {step} DO"

def tpl_end_for() -> str:
    return "END_FOR"

def tpl_while(cond: str) -> str:
    return f"WHILE {cond} DO"

def tpl_end_while() -> str:
    return "END_WHILE"

def tpl_repeat() -> str:
    return "REPEAT"

def tpl_until(cond: str) -> str:
    return f"UNTIL {cond}"

def tpl_end_repeat() -> str:
    return "END_REPEAT"

def tpl_then() -> str:
    return "THEN"

def tpl_then_actions(actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"THEN actions: {actions}" if actions else "THEN actions:"

def tpl_elsif_actions(actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"ELSIF actions: {actions}" if actions else "ELSIF actions:"

def tpl_else_actions(actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"ELSE actions: {actions}" if actions else "ELSE actions:"

def tpl_when_actions(conds: str, actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"WHEN {conds}: actions: {actions}" if actions else f"WHEN {conds}: actions:"

def tpl_loop_actions(actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"LOOP actions: {actions}" if actions else "LOOP actions:"

def tpl_if_actions(cond: str, actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"IF {cond} THEN actions: {actions}" if actions else f"IF {cond} THEN actions:"

def tpl_elsif_actions_cond(cond: str, actions: str = "") -> str:
    actions = (actions or "").strip()
    return f"ELSIF {cond} THEN actions: {actions}" if actions else f"ELSIF {cond} THEN actions:"

def tpl_return() -> str:
    return "Return"

def tpl_exit() -> str:
    # IEC 61131-3 的 EXIT 通常是“退出当前循环”
    return "Exit loop"