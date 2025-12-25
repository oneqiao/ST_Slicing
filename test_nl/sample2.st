PROGRAM P
VAR
  OUT1 : UINT;
  OUT2 : UINT;
  IN1  : UINT;
  N    : USINT;
  statusword : INT;
END_VAR

OUT1 := USHLW(IN1, N);                 (* 位置参数 function call *)
(OUT1, OUT2) := MOVB_5(IN1, IN1, N);   (* 多输出 tuple *)
FB_Test(EN := TRUE, IN1 := IN1);       (* CallStmt / function block style *)
OUT1 := INT_TO_UINT(IN1 := statusword);(* 命名参数 function call *)
END_PROGRAM
