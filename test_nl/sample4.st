PROGRAM MC_GroupStop
VAR
		AxesGroup_State : USINT;		// 4(* AG_GEID *) 轴组状态，定义见Group_State(GS_)
		AxesGroup_Axis_Sum_In_Group : USINT;	// 5(* AG_ASUM *) 轴组包含的实际的单轴数量
		AxesGroup_Axis_Num_In_Group : ARRAY[0..7] OF SINT;		// 6(* AG_ANUM *) 轴组里保存的轴号 ***需要初始化为-1***
		AxesGroup_Interrupt_Deceleration : REAL;				  // 228(* AG_ID *) Interrupt减速度，在GroupRunNumber = 0时为Stop/Halt参数
		AxesGroup_Interrupt_Jerk : REAL;						  // 232(* AG_IJ *) Interrupt加加速度，在GroupRunNumber = 0时为Stop/Halt参数
		AxesGroup_Run_Number : USINT;	  // 415(* AG_RN *) 运动编号 从1开始，0为复位状态 1--255 不需要初始化，在运动FB中被赋值，需要复位
		AxesGroup_Busy : USINT;			// 416(* AG_BZ *) 当前轴组忙碌标志
		AxesGroup_Buffer_Busy : USINT;	 // 417(* AG_BBZ *)轴组的缓冲区是否繁忙
		AxesGroup_Buffered_Transition_Mode : USINT;			   // 421(* AG_BTM *) 过渡模式
		AxesGroup_Now_TCP_Velocity : REAL;						// 648(* AG_NTV *) 当前轴组的速度 单位：每周期多少mm TCP_V 都是非负数 初始值为0
		i : USINT;
		stage : USINT;
		WRParRes : BOOL;
		execute_before : BOOL;
		axesGroup : USINT;
		bmode : USINT;
		dec : REAL;
		jerk : REAL;
		AxesGroup : USINT;
		Execute : BOOL;
		Deceleration : REAL;	// 时间单位：s
		Jerk : REAL;			// 时间单位：s
		BufferMode : USINT;
		AxesGroupOut : USINT;
		Done : BOOL;
		Busy : BOOL;
		Active : BOOL;
		CommandAborted : BOOL;
		Error : BOOL;
		ErrorID : UINT;
END_VAR

IF stage = 0(* FBS_WaitExec *) AND NOT Execute THEN
	Done := FALSE;
	Busy := FALSE;
	Active := FALSE;
	CommandAborted := FALSE;
	Error := FALSE;
	ErrorID := 0(* ER_NoError *);
	stage := 0(* FBS_WaitExec *);
END_IF;

IF stage = 0(* FBS_WaitExec *) AND Execute AND NOT execute_before THEN
	IF AxesGroup &lt; 0(* AXESGROUP_MINNUM *) OR AxesGroup &gt; 20(* AXESGROUP_MAXNUM *) THEN
		ErrorID := 16#05(* ER_AxesGroupNum_ParameterError *);
	ELSIF Deceleration &lt;= 0 OR Jerk &lt; 0 THEN
		ErrorID := 16#02(* ER_ParameterError *);
	ELSIF BufferMode &gt; 5 THEN
		ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
	END_IF;
	AxesGroup_State := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 4(* AG_GS *));
	IF AxesGroup_State = 0(* GS_GroupDisabled *) OR AxesGroup_State = 1(* GS_GroupErrorstop *) THEN
		ErrorID := 16#01 (* ER_StateError *);
	END_IF;
	IF AxesGroup_State = 4(* GS_GroupStopping *) THEN
		ErrorID := 16#20(* ER_AxisState_Stopping_Error *);
	END_IF;
	
	IF ErrorID &lt;&gt; 0(* ER_NoError *) THEN
		Error := TRUE;
	ELSE
		axesGroup := AxesGroup;
		dec := Deceleration;
		jerk := Jerk;
		bmode := BufferMode;
		
		Busy := TRUE;
		stage := 1(* FBS_WaitBuffer *);
	END_IF;
END_IF;

IF stage = 1(* FBS_WaitBuffer *) THEN
	IF bmode = 0(* BM_Aborting *) THEN
		stage := 3(* FBS_CalcRoute *);
	ELSE
		AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesGroup,AddrOffset := 417(* AG_BBZ *));
		IF AxesGroup_Buffer_Busy = 0 THEN
			WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 417(* AG_BBZ *),Data := 1);
			
			stage := 2(* FBS_InBuffer *);
		END_IF;
	END_IF;
END_IF;

IF stage = 2(* FBS_InBuffer *) THEN
	AxesGroup_Busy := MC_RdAxesPar_UB(AxesNum := axesGroup,AddrOffset := 416(* AG_BZ *));
	IF AxesGroup_Busy = 0 THEN
		WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 417(* AG_BBZ *),Data := 0);
		stage := 3(* FBS_CalcRoute *);
	END_IF;
END_IF;

IF stage = 3(* FBS_CalcRoute *) THEN
	WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 415(* AG_RN *),Data := 0);
	WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 4(* AG_GEID *),Data := 4(* GS_GroupStopping *));
	
	WRParRes := MC_WrAxesPar_FL(AxesNum := axesGroup,AddrOffset := 228(* AG_ID *),Data := dec);
	WRParRes := MC_WrAxesPar_FL(AxesNum := axesGroup,AddrOffset := 232(* AG_IJ *),Data := jerk);
	
	WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 421(* AG_BTM *),Data := 0);
	
	stage := 4;
END_IF;

IF stage = 4(* FBS_Moving *) THEN
	Busy := TRUE;
	Active := TRUE;
	CommandAborted := FALSE;
	
	AxesGroup_Busy := MC_RdAxesPar_UB(AxesNum := axesGroup,AddrOffset := 416(* AG_BZ *));
	AxesGroup_Now_TCP_Velocity  := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 648(* AG_NTV *));
	IF AxesGroup_Now_TCP_Velocity = 0 AND AxesGroup_Busy = 0 THEN
		Done := TRUE;
		Busy := FALSE;
		Active := FALSE;
		
		IF NOT Execute THEN
			WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 416(* AG_BZ *),Data := 0);
			WRParRes := MC_WrAxesPar_UB(AxesNum := axesGroup,AddrOffset := 4(* AG_GEID *),Data := 2(* GS_GroupStandby *));
			
			stage := 0(* FBS_WaitExec *);
		END_IF;
	END_IF;
END_IF;

AxesGroup_State := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 4(* AG_GS *));
IF stage &lt;&gt; 0(* FBS_WaitExec *) AND AxesGroup_State = 0(* GS_GroupDisabled *) THEN
	stage := 0(* FBS_WaitExec *);
END_IF;

AxesGroupOut := AxesGroup;
execute_before := Execute;

END_PROGRAM