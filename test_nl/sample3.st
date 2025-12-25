PROGRAM P
VAR
  i : INT;
  contorIndex : INT;

  Rect1 : ARRAY [0..7] OF INT;
  RectPoint : ARRAY [0..100] OF INT;

  a : ARRAY [0..10] OF MyStruct;
  InImage : MyImage;

END_VAR

Rect1[i] := RectPoint[contorIndex * 8 + i + 1];

IF InImage.Width > 0 THEN
  a[1].x := 3;
END_IF;

END_PROGRAM
