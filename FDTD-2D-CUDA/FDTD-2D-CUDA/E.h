#pragma once
#ifndef E_H
#define E_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src.h"
#include "H.h"
#include <iostream>
#include <fstream>

using namespace std;

class src;
class H;

class E
{
public:
	float **Ez, *dev_Ez, coe_Ez, coe_MUR;
	float *E_bd_u, *E_bd_d, *E_bd_l, *E_bd_r;
	float *E_nbd_u, *E_nbd_d, *E_nbd_l, *E_nbd_r;
	int size_Ez, size_Ez_x, size_Ez_y;

public:
	E(src source);
	void Ez_init(src source);
	void Ez_boundary_init(src source);
	void coe_Ez_set(src source);
	void Ez_transfer_host_device();
	void Ez_transfer_device_host();
	void Ez_checkout();
	void Ez_cmp_kernel(H Hxy, src source);
	void Ez_boundary_PEC();
	void Ez_boundry_MUR();
	void Ez_save2file();
	void Ez_MUR_u();
	void Ez_MUR_d();
	void Ez_MUR_lr();
};

#endif
