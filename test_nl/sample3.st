PROGRAM P_LONG
VAR
  A   : USINT;
  B   : USINT;
  W1  : UINT;
  W2  : UINT;
  C   : INT;
  i   : INT;
  j   : INT;

  R1  : BOOL;
  R2  : BOOL;
  R3  : BOOL;

  OUT1 : INT;
  OUT2 : INT;
END_VAR

(* outer loop *)
FOR i := 0 TO 2 BY 1 DO

  (* if/elsif/else with calls *)
  IF i = 0 THEN
    UEQB(IN1 := A, IN2 := B, OUT1 => R1);

    (* nested CASE *)
    CASE A OF
      0:
        UEQW(IN1 := W1, IN2 := W2, OUT1 => R2);
      1,2:
        UGEB(IN1 := A, IN2 := B, OUT1 => R3);
        (* unknown in catalog, should have no Semantics line *)
        OUT1 := OUT2;
    END_CASE;

  ELSIF i = 1 THEN
    (* nested WHILE *)
    WHILE C > 0 DO
      UEQB(IN1 := A, IN2 := B, OUT1 => R1);
      C := C - 1;
    END_WHILE;

  ELSE
    (* nested REPEAT *)
    j := 0;
    REPEAT
      UGEB(IN1 := A, IN2 := B, OUT1 => R3);
      j := j + 1;
    UNTIL j > 2
    END_REPEAT;

    (* unknown function, should have no Semantics line *)
    OUT2 := USHLW(OUT2, 1);
  END_IF;

END_FOR;

END_PROGRAM
