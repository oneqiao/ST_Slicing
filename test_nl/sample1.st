PROGRAM P
VAR
  IN3  : UINT;
  N    : USINT;
  OUT3 : UINT;

  IN1  : INT;
  OUT1 : INT;
  OUT2 : INT;

  A    : INT;
  B    : INT;
  C    : INT;

  i    : INT;
END_VAR

(* FB call: named IN + multiple OUTs (=>) *)
FB_Multi(IN1 := IN1, OUT1 => OUT1, OUT2 => OUT2);

(* function call: positional args, assignment LHS output *)
OUT3 := USHLW(IN3, N);

(* IF: nested calls in branches + assignment *)
IF A > 0 THEN
  OUT3 := USHLW(OUT3, N);
  FB_Test(IN1 := OUT1);
ELSIF A = 0 THEN
  OUT3 := USHLW(IN3, N);
ELSE
  OUT1 := OUT2;
END_IF;

(* CASE: multiple labels + ELSE branch *)
CASE B OF
  0:
    FB_Test(IN1 := OUT1);
  1,2:
    OUT3 := USHLW(IN3, N);
ELSE
    OUT2 := OUT1;
END_CASE;

(* FOR: loop body with a call *)
FOR i := 0 TO 2 BY 1 DO
  FB_Test(IN1 := OUT2);
END_FOR;

END_PROGRAM
