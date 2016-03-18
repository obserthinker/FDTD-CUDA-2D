#include "cuda_runtime.h"
#include "E.h"
#include "H.h"
#include "src.h"
#include <iostream>

using namespace std;

void main()
{
	src source(10,10,400);
	H Hxy(source);
	E Ez(source);

	source.src_checkout();
	//Hxy.Hx_transfer_host_device();
	
	for (int i = 0; i < source.size_time; i++)
	{
		Hxy.Hx_cmp_kernel(Ez, source);
		Hxy.Hy_cmp_kernel(Ez, source);
		Ez.Ez_cmp_kernel(Hxy, source);
		source.src_cmp_kernel(i, Ez);
		//Ez.Ez_boundary_PEC();
		Ez.Ez_boundry_MUR();
		//Hxy.Hx_save2file();
		Ez.Ez_save2file();
	}
	//Hxy.Hx_checkout();
	//Hxy.Hy_checkout();
	//Ez.Ez_checkout();
}