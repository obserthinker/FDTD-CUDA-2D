#pragma once
#ifndef SRC_H
#define SRC_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "E.cuh"

class E;

class src
{
public:
	int size_space_x, size_space_y, size_time;
	float dt, dz;
	const float C = 3e8f;
public:
	src(int space_x, int space_y, int time);
	void src_init(int space_x, int space_y, int time);
	void src_checkout();
};

#endif