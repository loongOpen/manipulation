#include "openloong_br.h"

uint32_t openloong_br::GetSysTime_us(void)
{
	gettimeofday(&tv, NULL);
	GetSysTime_us_ms = tv.tv_sec * 1000000 + tv.tv_usec; 
	GetSysTime_us_value = GetSysTime_us_ms;
	return GetSysTime_us_value;
}

float openloong_br::Get_T_Trig(void)
{
    return (float)GetSysTime_us() / 1000000.0f;
}

float openloong_br::Get_Cycle_T(int item)
{
	Cycle_T[item][OLD] = Cycle_T[item][NOW];
	Cycle_T[item][NOW] = (float)GetSysTime_us() / 1000000.0f;
	if (Cycle_T[item][NOW] > Cycle_T[item][OLD])
	{
		Cycle_T[item][NEW] = ((Cycle_T[item][NOW] - Cycle_T[item][OLD]));
		Cycle_T[item][DT_LAST] = Cycle_T[item][NEW];
	}
	else
		Cycle_T[item][NEW] = Cycle_T[item][DT_LAST];

	return Cycle_T[item][NEW];
}

void openloong_br::Cycle_Time_Init()
{
	for (int i = 0; i < GET_TIME_NUM; i++)
	{
		Get_Cycle_T(i);
	}
}
