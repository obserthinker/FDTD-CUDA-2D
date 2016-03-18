#include "src.h"

using namespace std;



src::src(int space_x, int space_y, int time)
{
	src_init(space_x, space_y, time);
}

void src::src_init(int space_x, int space_y, int time)
{
	dz = 0.015f;
	dt = dz / (2 * C);

	size_space_x = space_x;
	size_space_y = space_y;
	size_time = time;

	fstream myfile;
	myfile.open("src.txt", ios::out);
	
	myfile.close();
}

void src::src_checkout()
{
	cout << "dz = " << dz <<endl;
	cout << "dt = " << dt <<endl;
	cout << "space size on x: " << size_space_x << endl;
	cout << "space size on y: " << size_space_y << endl;
	cout << "time size: " << size_time << endl;
}

void src::src_cmp_kernel(int current_timestep, E E)
{
	float T, T0, vt, val_src, time;
	int src_position;

	time = current_timestep * dt;

	T = 5e-10f;
	T0 = 3 * T;
	vt = (time - T0) / T;

	val_src = expf(-powf(vt, 2.0f));

	src_position = (E.size_Ez_y / 2) * E.size_Ez_x + (E.size_Ez_x / 2);

	E.Ez[src_position] = val_src;
	fstream myfile;
	myfile.open("src.txt", ios::app);
	myfile << val_src << "\t";
	myfile.close();
	//cout << "source value: " << E.Ez[src_position] << endl;
}