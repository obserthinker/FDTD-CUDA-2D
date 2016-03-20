#include "src.cuh"

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
