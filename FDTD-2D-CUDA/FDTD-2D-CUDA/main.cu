#include "cuda_runtime.h"
#include "E.cuh"
#include "H.cuh"
#include "src.cuh"
#include "cmp.cuh"
#include <iostream>

using namespace std;

int main()
{
	src source(30,30,100);
	H Hxy(source);
	E Ez(source);
	size_t pitch_Hx, pitch_Hy, pitch_Ez;
	float *dev_Hx, *dev_Hy, *dev_Ez;
	int i, ele_ex, ele_hx, ele_hy;
	cudaError_t se, shx, shy, err;

	source.src_checkout();

	shx = cudaMallocPitch(&dev_Hx, &pitch_Hx, Hxy.size_Hx_x * sizeof(float), Hxy.size_Hx_y);
	shy = cudaMallocPitch(&dev_Hy, &pitch_Hy, Hxy.size_Hy_x * sizeof(float), Hxy.size_Hy_y);
	se = cudaMallocPitch(&dev_Ez, &pitch_Ez, Ez.size_Ez_x * sizeof(float), Ez.size_Ez_y);
	ele_ex = pitch_Ez / sizeof(float);
	ele_hx = pitch_Hx / sizeof(float);
	ele_hy = pitch_Hy / sizeof(float);

	if (shx == cudaSuccess && shy == cudaSuccess && se == cudaSuccess){
		for (i = 0; i < source.size_time; i++){
			err = cudaMemcpy2D(dev_Ez, pitch_Ez, Ez.Ez, Ez.size_Ez_x*sizeof(float), Ez.size_Ez_x*sizeof(float), Ez.size_Ez_y, cudaMemcpyHostToDevice);
			Hy_cmp_kernel<<<1,1>>>(dev_Ez, dev_Hy, Hxy.size_Hy_x, Hxy.size_Hy_y, Hxy.coe_H, ele_ex, ele_hy);
			Hx_cmp_kernel<<<1,1>>>(dev_Ez, dev_Hx, Hxy.size_Hx_x, Hxy.size_Hx_y, Hxy.coe_H, ele_ex, ele_hy);
			Ez_cmp_kernel<<<1,1>>>(dev_Ez, dev_Hx, dev_Hy, Ez.coe_Ez, Ez.size_Ez_x, Ez.size_Ez_y, ele_ex, ele_hx, ele_hy);
			Ez_boundary_PEC<<<1,1>>>(dev_Ez, Ez.size_Ez_x, Ez.size_Ez_y, ele_ex);
			src_cmp_kernel<<<1,1>>>(i, dev_Ez, Ez.size_Ez_x, Ez.size_Ez_y, source.dt, ele_ex);
			cudaMemcpy2D(Ez.Ez, Ez.size_Ez_x*sizeof(float), dev_Ez, pitch_Ez, Ez.size_Ez_x*sizeof(float), Ez.size_Ez_y, cudaMemcpyDeviceToHost);
			Ez.Ez_save2file();
		}
	}
	else{
		cout << "cudaMalloc failed" << endl;
		return 0;
	}
	return 0;
}