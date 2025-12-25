PROGRAM MC_ReadMotionState

VAR
		kxzhgl:REAL;//当前模块速度
		hvnz:REAL;//当前模块目标速度，输入引脚输入的速度
		rjcmm:INT;//当前模块运动方向
		bordvesg:REAL; //当前模块末速度
		hoktzig:REAL;
		yntxdriv:INT;
		en : USINT;
		endsff : BOOL;
		zsmrzyo : USINT;
		eyyzap : USINT;
		wpk : BOOL;
		hof : BOOL;
		ukd : BOOL;
		zpsl : USINT;
		sgeudj : BOOL;
		svxzo : BOOL;
		jrbyoddd : BOOL;
		timiok : BOOL;
		lewbcwgz : BOOL;
END_VAR


IF endsff THEN
	IF en &gt;= 50(* AXIS_MAXNUM *) THEN
		ukd := TRUE;
		zpsl := 16#02(* ER_ParameterError *);
	ELSIF zsmrzyo &lt;&gt; 0(* VS_SetValue *) AND zsmrzyo &lt;&gt; 1(* VS_ActualValue *) AND zsmrzyo &lt;= 2(* VS_CommandedValue *) THEN
		ukd := TRUE;
		zpsl := 16#02(* ER_ParameterError *);
	ELSE
		ukd := FALSE;
		zpsl := 16#00(* ER_NoError *);
	END_IF;
	
	IF NOT ukd THEN
		wpk := TRUE;
		ukd := FALSE;
		zpsl := 16#00(* ER_NoError *);
		hof := TRUE;
		
		kxzhgl := MC_RdAxisPar_FL(AxisNum := en,AddrOffset := 104);
		bordvesg := MC_RdAxisPar_FL(AxisNum := en,AddrOffset := 136) ;
		hvnz := MC_RdAxisPar_FL(AxisNum := en,AddrOffset := 140);
		rjcmm := USINT_TO_INT(IN1 := MC_RdAxisPar_UB(AxisNum := en,AddrOffset := 156)) ;
		(*
		判断加速度还是减速度的时候，通过上一个周期的轴速度和这个周期的轴速度做差，进行判断。
		但是速度曲线不是光滑的，存在震荡的问题，所以采样周期设置为50
		也就是说，当前的速度和前50个周期的速度进行做差，然后判断加减速度阶段
		*)
		IF yntxdriv &lt; 50 THEN
			yntxdriv := yntxdriv + 1;
			hoktzig := hoktzig + kxzhgl;
		ELSE
			hoktzig := hoktzig / 50;
			IF kxzhgl &lt;&gt; 0 THEN
				IF kxzhgl &lt; hoktzig  THEN
					jrbyoddd := TRUE;
					svxzo := FALSE;
					sgeudj := FALSE;
				ELSIF kxzhgl &gt; hoktzig  THEN
					jrbyoddd := FALSE;
					svxzo := TRUE;
					sgeudj := FALSE;
				ELSE
					jrbyoddd := FALSE;
					svxzo := FALSE;
					sgeudj := TRUE;
				END_IF;
				
				IF rjcmm &gt; 0 THEN
					timiok := TRUE;
					lewbcwgz := FALSE;
				ELSIF rjcmm &lt; 0 THEN
					timiok := FALSE;
					lewbcwgz := TRUE;
				ELSE
					timiok := FALSE;
					lewbcwgz := FALSE;
				END_IF;
				hoktzig := 0;
			ELSE
				sgeudj := FALSE;
				svxzo := FALSE;
				jrbyoddd := FALSE;
				timiok := FALSE;
				lewbcwgz := FALSE;
			END_IF;
			yntxdriv := 0;
		END_IF;
	ELSE
		wpk := FALSE;
		hof := FALSE;
		sgeudj := FALSE;
		svxzo := FALSE;
		jrbyoddd := FALSE;
		timiok := FALSE;
		lewbcwgz := FALSE;
	END_IF;
ELSE
	wpk := FALSE;
	ukd := FALSE;
	zpsl := 16#00(* ER_NoError *);
	hof := FALSE;
	sgeudj := FALSE;
	svxzo := FALSE;
	jrbyoddd := FALSE;
	timiok := FALSE;
	lewbcwgz := FALSE;
END_IF;

eyyzap := en;

END_PROGRAM