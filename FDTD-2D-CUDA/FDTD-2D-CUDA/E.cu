#include "E.cuh"

//using namespace std;

const float epsilon = 8.85e-12f;

E::E(src source)
{
	Ez_init(source);
	coe_Ez_set(source);
}

void E::coe_Ez_set(src source)
{
	coe_Ez = source.dt / (epsilon * source.dz);
}

void E::Ez_init(src source)
{
	int i, j;
	//initialize Ez
	size_Ez_x = source.size_space_x + 1;
	size_Ez_y = source.size_space_y + 1;
	size_Ez = size_Ez_x * size_Ez_y;

	Ez = (float*)malloc(size_Ez * sizeof(float));

	for ( i = 0; i < size_Ez_y; i++){
		for (j = 0; j < size_Ez_x; j++){
			Ez[i * size_Ez_x + j] = 0.f;
		}
	}

	//initialize file
	fstream myfile;
	myfile.open("Ez.txt", ios::out);
	myfile.close();

	//initialize boundary
	Ez_boundary_init(source);
}

void E::Ez_boundary_init(src source)
{
	E_bd_l = (float*)malloc(size_Ez_y * sizeof(float));
	E_bd_r = (float*)malloc(size_Ez_y * sizeof(float));
	E_bd_u = (float*)malloc(size_Ez_x * sizeof(float));
	E_bd_d = (float*)malloc(size_Ez_x * sizeof(float));
	E_nbd_l = (float*)malloc(size_Ez_y * sizeof(float));
	E_nbd_r = (float*)malloc(size_Ez_y * sizeof(float));
	E_nbd_u = (float*)malloc(size_Ez_x * sizeof(float));
	E_nbd_d = (float*)malloc(size_Ez_x * sizeof(float));
	memset(E_bd_l, 0, size_Ez_y * sizeof(float));
	memset(E_bd_r, 0, size_Ez_y * sizeof(float));
	memset(E_bd_u, 0, size_Ez_x * sizeof(float));
	memset(E_bd_d, 0, size_Ez_x * sizeof(float));
	memset(E_nbd_l, 0, size_Ez_y * sizeof(float));
	memset(E_nbd_r, 0, size_Ez_y * sizeof(float));
	memset(E_nbd_u, 0, size_Ez_x * sizeof(float));
	memset(E_nbd_d, 0, size_Ez_x * sizeof(float));

	coe_MUR = (source.C * source.dt - source.dz) / (source.C * source.dt + source.dz);
}

void E::Ez_checkout()
{
	int i, j;
	cout << "Ez size: " << size_Ez << endl;
	for (i = 0; i < size_Ez_y; i++){
		for (j = 0; j < size_Ez_x; j++){
			cout << Ez[i * size_Ez_x + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}



void E::Ez_boundry_MUR()
{
	Ez_MUR_u();
	Ez_MUR_d();
	Ez_MUR_lr();
}

void E::Ez_save2file()
{
	int i, j;
	fstream myfile;
	myfile.open("Ez.txt", ios::app);

	for (i = 0; i < size_Ez_y; i++){
		for (j = 0; j < size_Ez_x; j++){
			myfile << Ez[i * size_Ez_x + j] << "\t";
		}
		myfile << endl;
	}
	myfile << endl;
	myfile.close();
}

void E::Ez_MUR_u()
{
	int i;
	for ( i = 0; i < size_Ez_x; i++){
		Ez[(size_Ez_y - 1) * size_Ez_x + i] = E_nbd_u[i]
			+ coe_MUR * (Ez[(size_Ez_y - 2) * size_Ez_x + i]
			- E_bd_u[i]);
		E_nbd_u[i] = Ez[(size_Ez_y - 2) * size_Ez_x + i];
		E_bd_u[i] = Ez[(size_Ez_y - 1) * size_Ez_x + i];
	}
}

void E::Ez_MUR_d()
{
	int i;
	for ( i = 0; i < size_Ez_x; i++){
		Ez[i] = E_nbd_d[i] + coe_MUR * (Ez[1*size_Ez_x + i]
			- E_bd_d[i]);
		E_nbd_d[i] = Ez[1*size_Ez_x + i];
		E_bd_d[i] = Ez[i];
	}
}

void E::Ez_MUR_lr()
{
	for (int i = 0; i < size_Ez_y; i++){
		//left
		Ez[i*size_Ez_x + 0] = E_nbd_l[i] + coe_MUR *
			(Ez[i * size_Ez_x + 1] - E_bd_l[i]);
		E_nbd_l[i] = Ez[i * size_Ez_x + 1];
		E_bd_l[i] = Ez[i * size_Ez_x + 0];
		//right
		Ez[i* size_Ez_x + (size_Ez_x - 1)] = E_nbd_r[i] + coe_MUR *
			(Ez[i * size_Ez_x + (size_Ez_x - 2)] - E_bd_r[i]);
		E_nbd_r[i] = Ez[i * size_Ez_x + (size_Ez_x - 2)];
		E_bd_r[i] = Ez[i* size_Ez_x + (size_Ez_x - 1)];
	}
}