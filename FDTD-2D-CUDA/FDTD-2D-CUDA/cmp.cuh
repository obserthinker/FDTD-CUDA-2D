#pragma once
#ifndef CMP_CUH
#define CMP_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

__global__
void src_cmp_kernel(int current_timestep, float* dev_Ez, int size_Ez_x, int size_Ez_y, float dt, int ele_ex);

__global__ void Hy_cmp_kernel(
	float* Ez, float* Hy,
	int size_Hy_x, int size_Hy_y,
	float coe_H, int ele_ex, int ele_hy
	);

__global__ void Hx_cmp_kernel(
	float* Ez, float* Hx,
	int size_Hx_x, int size_Hx_y,
	float coe_H, int ele_ex, int ele_hx
	);

__global__ void Ez_boundary_PEC(
	float* Ez,
	int size_Ez_x, int size_Ez_y, int ele_ex
	);

__global__
void Ez_cmp_kernel(
float* Ez, float* Hx, float* Hy,
float coe_Ez, int size_Ez_x, int size_Ez_y,
int ele_ex, int ele_hx, int ele_hy
);

__global__
void Ez_MUR_lr(
float* Ez, float* E_bd_l, float* E_nbd_l,
float* E_bd_r, float* E_nbd_r,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
);

__global__
void Ez_MUR_d(
float* Ez, float* E_bd_d, float* E_nbd_d,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
);

__global__
void Ez_MUR_u(
float* Ez, float* E_bd_u, float* E_nbd_u,
int size_Ez_x, int size_Ez_y, float coe_MUR, int ele_ex
);
#endif