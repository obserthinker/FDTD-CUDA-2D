#pragma once
#ifndef H_H
#define H_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src.cuh"
#include "E.cuh"
#include <iostream>
#include <fstream>
using namespace std;
class E;
class src;

class H
{
public:
	float *Hx, *Hy, coe_H, *dev_Hx, *dev_Hy;
	int size_Hx, size_Hy, size_Hx_x, size_Hx_y, size_Hy_x, size_Hy_y;
	const float PI = 3.14159265939f;
	const float mu = (4.0*PI)*1e-7f;
	size_t pitch_Hx, pitch_Hy;
	int ele_Hx, ele_Hy;

public:
	H(src source);
	void Hx_init(src source);
	void Hy_init(src source);
	void coe_H_set(src source);
	void Hx_checkout();
	void Hy_checkout();
	void Hx_save2file();
	void Hy_save2file();
};

#endif
