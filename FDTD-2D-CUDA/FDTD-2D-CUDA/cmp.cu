#include "cmp.cuh"

__global__ 
void src_cmp_kernel(
int current_timestep, 
float* dev_Ez, 
int size_Ez_x, int size_Ez_y, float dt, int ele_ex
)
{
	float T,T0;
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
	int i, j;
	float dif_Hy, dif_Hx;
	for (i = 1; i < size_Ez_y - 1; i++){
		for (j = 1; j < size_Ez_x - 1; j++){
			//Hy(i,j)	-	Hy(i-1,j)
			dif_Hy = Hy[i*ele_hy + j] - Hy[(i - 1)* ele_hy + j];
			//Hx(i,j-1)	-	Hx(i,j)
			dif_Hx = Hx[i*ele_hx + (j - 1)] - Hx[i*ele_hx + j];
			Ez[i*ele_hx + j] += coe_Ez * (dif_Hx + dif_Hy);
		}
	}
}

__global__ void Hx_cmp_kernel(
	float* Ez, float* Hx, 
	int size_Hx_x, int size_Hx_y,
	float coe_H, int ele_ex, int ele_hx
)
{
	int i, j;

	for ( i = 0; i < size_Hx_y; i++){
		for ( j = 0; j < size_Hx_x; j++){
			Hx[i*ele_hx + j] += coe_H *
				(Ez[i*ele_ex + j] - Ez[i*ele_ex + (j + 1)]);
		}
	}
}

__global__ void Hy_cmp_kernel(
	float* Ez, float* Hy,
	int size_Hy_x, int size_Hy_y,
	float coe_H, int ele_ex, int ele_hy
)
{
	int i, j;

	for ( i = 0; i < size_Hy_y; i++){
		for ( j = 0; j < size_Hy_x; j++){
			Hy[i*ele_hy + j] += 
				coe_H * (Ez[(i + 1)*ele_ex + j] - Ez[i*ele_ex + j]);
		}
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
	int i;
	for ( i = 0; i < size_Ez_x; i++){
		Ez[(size_Ez_y - 1) * ele_ex + i] = E_nbd_u[i]
			+ coe_MUR * (Ez[(size_Ez_y - 2) * ele_ex + i]
			- E_bd_u[i]);
		E_nbd_u[i] = Ez[(size_Ez_y - 2) * ele_ex + i];
		E_bd_u[i] = Ez[(size_Ez_y - 1) * ele_ex + i];
	}
}

__global__
void Ez_MUR_d(
float* Ez, float* E_bd_d, float* E_nbd_d, 
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
)
{
	int i;
	for ( i = 0; i < size_Ez_x; i++){
		Ez[i] = E_nbd_d[i] + coe_MUR * (Ez[1*ele_ex + i]
			- E_bd_d[i]);
		E_nbd_d[i] = Ez[1*ele_ex + i];
		E_bd_d[i] = Ez[i];
	}
}

__global__
void Ez_MUR_lr(
float* Ez, float* E_bd_l, float* E_nbd_l, 
float* E_bd_r, float* E_nbd_r,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
)
{
	for (int i = 0; i < size_Ez_y; i++){
		//left
		Ez[i*ele_ex + 0] = E_nbd_l[i] + coe_MUR *
			(Ez[i * ele_ex + 1] - E_bd_l[i]);
		E_nbd_l[i] = Ez[i * ele_ex + 1];
		E_bd_l[i] = Ez[i * ele_ex + 0];
		//right
		Ez[i* ele_ex + (size_Ez_x - 1)] = E_nbd_r[i] + coe_MUR *
			(Ez[i * ele_ex + (size_Ez_x - 2)] - E_bd_r[i]);
		E_nbd_r[i] = Ez[i * ele_ex + (size_Ez_x - 2)];
		E_bd_r[i] = Ez[i* ele_ex + (size_Ez_x - 1)];
	}
}
