#include "cuda_runtime.h"
#include "E.cuh"
#include "H.cuh"
#include "src.cuh"
#include "cmp.cuh"
#include <iostream>

using namespace std;

int main()
{
	src source(100, 100, 300);
	H Hxy(source);
	E Ez(source);
	int i;

	source.src_checkout();

	cout << Ez.ele_Ez << "\t" << Ez.size_Ez_x << endl;
	for (i = 0; i < source.size_time; i++){
		//cudaMemcpy2D(Ez.dev_Ez, Ez.pitch_Ez, Ez.Ez, Ez.size_Ez_x*sizeof(float), Ez.size_Ez_x*sizeof(float), Ez.size_Ez_y, cudaMemcpyHostToDevice);

		Hy_cmp_kernel << <50, 50 >> >(Ez.dev_Ez, Hxy.dev_Hy, Hxy.size_Hy_x, Hxy.size_Hy_y, Hxy.coe_H, Ez.ele_Ez, Hxy.ele_Hy);
		Hx_cmp_kernel << <50, 50 >> >(Ez.dev_Ez, Hxy.dev_Hx, Hxy.size_Hx_x, Hxy.size_Hx_y, Hxy.coe_H, Ez.ele_Ez, Hxy.ele_Hy);
		Ez_cmp_kernel << <50, 50 >> >(Ez.dev_Ez, Hxy.dev_Hx, Hxy.dev_Hy, Ez.coe_Ez, Ez.size_Ez_x, Ez.size_Ez_y, Ez.ele_Ez, Hxy.ele_Hx, Hxy.ele_Hy);
		//Ez_boundary_PEC<<<1,1>>>(Ez.dev_Ez, Ez.size_Ez_x, Ez.size_Ez_y, Ez.ele_Ez);
		Ez_MUR_u << <50, 50 >> >(Ez.dev_Ez, Ez.E_bd_u, Ez.E_nbd_u, Ez.size_Ez_x, Ez.size_Ez_y, Ez.coe_MUR, Ez.ele_Ez);
		Ez_MUR_d << <50, 50 >> >(Ez.dev_Ez, Ez.E_bd_d, Ez.E_nbd_d, Ez.size_Ez_x, Ez.size_Ez_y, Ez.coe_MUR, Ez.ele_Ez);
		Ez_MUR_lr << <1, Ez.size_Ez_y >> >(Ez.dev_Ez, Ez.E_bd_l, Ez.E_nbd_l, Ez.E_bd_r, Ez.E_nbd_r, Ez.size_Ez_x, Ez.size_Ez_y, Ez.coe_MUR, Ez.ele_Ez);
		src_cmp_kernel << <1, 1 >> >(i, Ez.dev_Ez, Ez.size_Ez_x, Ez.size_Ez_y, source.dt, Ez.ele_Ez);

		cudaMemcpy2D(Ez.Ez, Ez.size_Ez_x*sizeof(float), Ez.dev_Ez, Ez.pitch_Ez, Ez.size_Ez_x*sizeof(float), Ez.size_Ez_y, cudaMemcpyDeviceToHost);
		Ez.Ez_save2file();
	}
	return 0;
}