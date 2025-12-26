PROGRAM P_NEST
VAR
  A    : INT;
  B    : INT;
  C    : INT;
  i    : INT;
  j    : INT;

  IN3  : UINT;
  N    : USINT;
  OUT1 : UINT;
  OUT2 : UINT;
  OUT3 : UINT;
END_VAR

FOR i := 0 TO 2 BY 1 DO

  IF A > 0 THEN

    CASE B OF
      0:
        OUT3 := USHLW(IN3, N);

      1, 2:
        IF B = 1 THEN
          FB_Test(IN1 := OUT1);
        ELSE
          FB_Multi(IN1 := A, OUT1 => OUT1, OUT2 => OUT2);
        END_IF;

    ELSE
      WHILE C > 0 DO
        OUT1 := OUT2;
        C := C - 1;
      END_WHILE;
    END_CASE;

  ELSIF A = 0 THEN

    REPEAT
      OUT3 := USHLW(OUT3, N);
      j := j + 1;
    UNTIL j > 3
    END_REPEAT;

  ELSE
    OUT2 := OUT1;
  END_IF;

END_FOR;

END_PROGRAM

