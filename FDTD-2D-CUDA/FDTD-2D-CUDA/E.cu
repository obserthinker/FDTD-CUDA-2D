#include "E.cuh"

//using namespace std;

const float epsilon = 8.85e-12f;

E::E(src source)
{
	Ez_init(source);
	coe_Ez_set(source);
	Ez_boundary_init(source);
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
	cudaMallocPitch(&dev_Ez, &pitch_Ez, size_Ez_x * sizeof(float), size_Ez_y);
	ele_Ez = pitch_Ez / sizeof(float);

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
	cudaMalloc(&E_bd_l, size_Ez_y * sizeof(float));
	cudaMalloc(&E_bd_r, size_Ez_y * sizeof(float));
	cudaMalloc(&E_bd_u, size_Ez_x * sizeof(float));
	cudaMalloc(&E_bd_d, size_Ez_x * sizeof(float));
	cudaMalloc(&E_nbd_l, size_Ez_y * sizeof(float));
	cudaMalloc(&E_nbd_r, size_Ez_y * sizeof(float));
	cudaMalloc(&E_nbd_u, size_Ez_x * sizeof(float));
	cudaMalloc(&E_nbd_d, size_Ez_x * sizeof(float));
	cudaMemset(E_bd_l, 0, size_Ez_y * sizeof(float));
	cudaMemset(E_bd_r, 0, size_Ez_y * sizeof(float));
	cudaMemset(E_bd_u, 0, size_Ez_x * sizeof(float));
	cudaMemset(E_bd_d, 0, size_Ez_x * sizeof(float));
	cudaMemset(E_nbd_l, 0, size_Ez_y * sizeof(float));
	cudaMemset(E_nbd_r, 0, size_Ez_y * sizeof(float));
	cudaMemset(E_nbd_u, 0, size_Ez_x * sizeof(float));
	cudaMemset(E_nbd_d, 0, size_Ez_x * sizeof(float));

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
