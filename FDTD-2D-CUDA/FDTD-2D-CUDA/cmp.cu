#include "cmp.cuh"

__global__
void src_cmp_kernel(
int current_timestep,
float* dev_Ez,
int size_Ez_x, int size_Ez_y, float dt, int ele_ex
)
{
	float T, T0;
	float vt, val_src, time;
	int src_pos_x, src_pos_y;

	time = current_timestep * dt;

	T = 5e-10f;
	T0 = 3 * T;
	vt = (time - T0) / T;

	val_src = expf(-powf(vt, 2.0f));

	src_pos_x = size_Ez_y / 2;
	src_pos_y = size_Ez_x / 2;

	dev_Ez[src_pos_y * ele_ex + src_pos_x] = val_src;
}

__global__
void Ez_cmp_kernel(
float* Ez, float* Hx, float* Hy,
float coe_Ez, int size_Ez_x, int size_Ez_y,
int ele_ex, int ele_hx, int ele_hy
)
{
	int x, y;
	int tid, number;
	tid = threadIdx.x + blockIdx.x*blockDim.x;
	float dif_Hy, dif_Hx;
	while (tid < ele_ex*size_Ez_y)
	{
		number = tid + 1;
		y = number % ele_ex;//row
		x = number - (y*ele_ex);//column
		//Hy(i,j)	-	Hy(i-1,j)
		dif_Hy = Hy[y*ele_hy + x] - Hy[(y - 1)* ele_hy + x];
		//Hx(i,j-1)	-	Hx(i,j)
		dif_Hx = Hx[y*ele_hx + (x - 1)] - Hx[y*ele_hx + x];
		Ez[y*ele_ex + x] += coe_Ez * (dif_Hx + dif_Hy);
		tid += blockDim.x*gridDim.x;
	}
}

__global__ void Hx_cmp_kernel(
	float* Ez, float* Hx,
	int size_Hx_x, int size_Hx_y,
	float coe_H, int ele_ex, int ele_hx
	)
{
	int x, y;
	int tid, number;
	tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < ele_hx*size_Hx_y)
	{
		number = tid + 1;
		y = number % ele_hx;
		x = number - (y*ele_hx);
		Hx[y*ele_hx + x] += coe_H *(Ez[y*ele_ex + x] - Ez[y*ele_ex + (x + 1)]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void Hy_cmp_kernel(
	float* Ez, float* Hy,
	int size_Hy_x, int size_Hy_y,
	float coe_H, int ele_ex, int ele_hy
	)
{
	int x, y;
	int tid, number;
	tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < ele_hy * size_Hy_y)
	{
		number = tid + 1;
		y = number % ele_hy;
		x = number - (y*ele_hy);
		Hy[y*ele_hy + x] += coe_H * (Ez[(y + 1)*ele_ex + x] - Ez[y*ele_ex + x]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void Ez_boundary_PEC(
	float* Ez,
	int size_Ez_x, int size_Ez_y, int ele_ex
	)
{
	int i, j;

	for (i = 0; i < size_Ez_y; i++){
		if (i == 0 || i == (size_Ez_y - 1)){
			for (j = 0; j < size_Ez_x; j++){
				Ez[i * ele_ex + j] = 0.f;
			}
		}
		else{
			Ez[i* ele_ex + 0] = 0.f;
			Ez[i * ele_ex + size_Ez_x - 1] = 0.f;
		}
	}
}

__global__
void Ez_MUR_u(
float* Ez, float* E_bd_u, float* E_nbd_u,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
)
{
	int tid;
	tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < size_Ez_x){
		Ez[(size_Ez_y - 1) * ele_ex + tid] = E_nbd_u[tid]
			+ coe_MUR * (Ez[(size_Ez_y - 2) * ele_ex + tid]
			- E_bd_u[tid]);
		E_nbd_u[tid] = Ez[(size_Ez_y - 2) * ele_ex + tid];
		E_bd_u[tid] = Ez[(size_Ez_y - 1) * ele_ex + tid];
		tid += blockDim.x * gridDim.x;
	}
}

__global__
void Ez_MUR_d(
float* Ez, float* E_bd_d, float* E_nbd_d,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
)
{
	int tid;
	tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < size_Ez_x)
	{
		Ez[tid] = E_nbd_d[tid] + coe_MUR * (Ez[1 * ele_ex + tid]
			- E_bd_d[tid]);
		E_nbd_d[tid] = Ez[1 * ele_ex + tid];
		E_bd_d[tid] = Ez[tid];
		tid += blockDim.x * gridDim.x;
	}
}

__global__
void Ez_MUR_lr(
float* Ez, float* E_bd_l, float* E_nbd_l,
float* E_bd_r, float* E_nbd_r,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
)
{
	int tid;
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size_Ez_y)
	{
		//left
		Ez[tid*ele_ex + 0] = E_nbd_l[tid] + coe_MUR *
			(Ez[tid * ele_ex + 1] - E_bd_l[tid]);
		E_nbd_l[tid] = Ez[tid * ele_ex + 1];
		E_bd_l[tid] = Ez[tid * ele_ex + 0];
		//right
		Ez[tid* ele_ex + (size_Ez_x - 1)] = E_nbd_r[tid] + coe_MUR *
			(Ez[tid * ele_ex + (size_Ez_x - 2)] - E_bd_r[tid]);
		E_nbd_r[tid] = Ez[tid * ele_ex + (size_Ez_x - 2)];
		E_bd_r[tid] = Ez[tid* ele_ex + (size_Ez_x - 1)];

		tid += blockDim.x * gridDim.x;
	}
}
