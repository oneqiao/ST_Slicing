PROGRAM P_ComplexControlTest
VAR
    i       : INT;
    j       : INT;
    k       : INT;
    x       : INT;
    y       : INT;
    mode    : INT;
    sum     : INT;
    flag    : BOOL;
    done    : BOOL;
END_VAR

// init
sum := 0;
x := 0;
y := 0;
mode := 0;
flag := FALSE;
done := FALSE;

// outer FOR
FOR i := 1 TO 5 DO

    // IF inside FOR
    IF (i MOD 2) = 0 THEN
        x := x + i;
        flag := TRUE;
    ELSE
        x := x - i;
        flag := FALSE;
    END_IF;

    // CASE inside FOR, with nested WHILE / IF / REPEAT
    CASE mode OF

        0:
            // WHILE nested in CASE
            j := 0;
            WHILE (j < 3) AND (NOT done) DO
                sum := sum + (i * 10 + j);

                // IF nested in WHILE
                IF (sum > 50) THEN
                    done := TRUE;
                ELSE
                    done := FALSE;
                END_IF;

                j := j + 1;
            END_WHILE;

            // update mode based on result
            IF done THEN
                mode := 1;
            ELSE
                mode := 2;
            END_IF;

        1:
            // Nested FOR inside CASE branch
            FOR k := 1 TO 3 DO
                y := y + (k * i);

                // Nested IF inside nested FOR
                IF y > 30 THEN
                    y := y - 5;
                END_IF;
            END_FOR;

            // Force transition
            mode := 3;

        2:
            // REPEAT inside CASE branch
            j := 0;
            REPEAT
                j := j + 1;
                sum := sum + j;

                // IF inside REPEAT
                IF (sum MOD 7) = 0 THEN
                    flag := NOT flag;
                END_IF;

            UNTIL (j >= 4) OR flag
            END_REPEAT;

            // Decide next mode
            IF flag THEN
                mode := 1;
            ELSE
                mode := 0;
            END_IF;

        3:
            // Deep nesting: IF -> CASE -> WHILE
            IF x > y THEN
                CASE (x - y) OF
                    0:
                        done := TRUE;

                    1, 2, 3:
                        done := FALSE;

                    ELSE
                        // WHILE nested under CASE ELSE
                        j := 0;
                        WHILE j < 2 DO
                            sum := sum + 100;
                            j := j + 1;
                        END_WHILE;

                        done := (sum > 200);
                END_CASE;
            ELSE
                x := x + 1;
            END_IF;

            // Reset to loop other paths
            mode := 0;

    ELSE
        // CASE default branch
        mode := 0;
    END_CASE;

    // final IF inside FOR to vary paths
    IF done THEN
        sum := sum + 999;
        done := FALSE;
    END_IF;

END_FOR;

// exit state
IF sum > 500 THEN
    flag := TRUE;
ELSE
    flag := FALSE;
END_IF;

END_PROGRAM
