PROGRAM P
VAR
  a : INT; // counter
  b : INT; (* output *)
END_VAR

//IL
LD a
ADD 1
ST a

IF a > 0 THEN { positive branch }
  a := a - 1; // decrement
END_IF;

END_PROGRAM
