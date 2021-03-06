#pragma once
#ifndef E_H
#define E_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src.cuh"
#include "H.cuh"
#include <iostream>
#include <fstream>

using namespace std;

class src;
class H;

class E
{
public:
	float *Ez, coe_Ez, coe_MUR, *dev_Ez;
	float *E_bd_u, *E_bd_d, *E_bd_l, *E_bd_r;
	float *E_nbd_u, *E_nbd_d, *E_nbd_l, *E_nbd_r;
	int size_Ez, size_Ez_x, size_Ez_y;
	size_t pitch_Ez;
	int ele_Ez;

public:
	E(src source);
	void Ez_init(src source);
	void Ez_boundary_init(src source);
	void coe_Ez_set(src source);
	void Ez_checkout();
	void Ez_save2file();
};

#endif