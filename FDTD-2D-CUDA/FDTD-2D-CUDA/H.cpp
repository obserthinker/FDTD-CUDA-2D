#include "H.h"

H::H(src source)
{
	Hx_init(source);
	Hy_init(source);
	coe_H_set(source);
}

void H::Hx_init(src source)
{
	int i, j;
    //size_Hx = source.size_space_x * (source.size_space_y - 1);
	size_Hx_x = source.size_space_x;
	size_Hx_y = source.size_space_y + 1;
	size_Hx = size_Hx_x * size_Hx_y;

    Hx = (float**)malloc(size_Hx_y * sizeof(float*));
	for (i = 0; i < size_Hx_y; i++)
	{
		Hx[i] = (float *)malloc(size_Hx_x * sizeof(float));
	}
    //cudaMalloc(&dev_Hx, size_Hx * sizeof(float));

    for (i = 0; i < size_Hx_y; ++i){
		for ( j = 0; j < size_Hx_x; j++){
			Hx[i][j] = 0.f;
		}
    }

	fstream myfile;
	myfile.open("Hx.txt", ios::out);
	myfile.close();
}

void H::Hy_init(src source)
{
	int i, j;
    //size_Hy = (source.size_space_x - 1) * source.size_space_y;
	size_Hy_x = source.size_space_x + 1;
	size_Hy_y = source.size_space_y;
	size_Hy = size_Hy_x * size_Hy_y;

    Hy = (float**)malloc(size_Hy_y * sizeof(float*));
	for (i = 0; i < size_Hy_y; i++){
		Hy[i] = (float*)malloc(size_Hy_x * sizeof(float));
	}

	//cudaMalloc(&dev_Hy, size_Hy * sizeof(float));

    for (i = 0; i < size_Hy_y; ++i){
    	for (j = 0; j < size_Hy_x; j++){
			Hy[i][j] = 0.f;
		} 
    }

	fstream myfile;
	myfile.open("Hx.txt", ios::out);
	myfile.close();
}

void H::coe_H_set(src source)
{
	coe_H = source.dt / (mu * source.dz);
}
//forbid
void H::Hx_transfer_host_device()
{
	cudaMemcpy(dev_Hx, Hx, size_Hx * sizeof(float), cudaMemcpyHostToDevice);
}
//forbid
void H::Hx_transfer_device_host()
{
	cudaMemcpy(Hx, dev_Hx, size_Hx * sizeof(float), cudaMemcpyDeviceToHost);
}
//firbid
void H::Hy_transfer_host_device()
{
	cudaMemcpy(dev_Hy, Hy, size_Hy * sizeof(float), cudaMemcpyHostToDevice);
}
//forbid
void H::Hy_transfer_device_host()
{
	cudaMemcpy(dev_Hy, Hy, size_Hy * sizeof(float), cudaMemcpyDeviceToHost);
}

//empty, waiting to be test
void H::Hx_cmp_kernel(E E, src source)
{
	int i, j;

	for ( i = 0; i < size_Hx_y; i++){
		for ( j = 0; j < size_Hx_x; j++){
			Hx[i][j] += coe_H *
				(E.Ez[i][j] - E.Ez[i][j + 1]);
		}
	}
}

//empty, waiting to be test
void H::Hy_cmp_kernel(E E, src source)
{
	int i, j;

	for ( i = 0; i < size_Hy_y; i++){
		for ( j = 0; j < size_Hy_x; j++){
			Hy[i][j] += 
				coe_H * (E.Ez[i + 1][j] - E.Ez[i][j]);
		}
	}
}

void H::Hx_checkout()
{
	int i, j;
	cout << "Hx size: " << size_Hx << endl;
	for (i = 0; i < size_Hx_y; i++){
		for (j = 0; j < size_Hx_x; j++){
			cout << Hx[i][j] << "\t";
		}
	}
	cout << endl;
}

void H::Hy_checkout()
{
	int i, j;
	cout << "Hy size: " << size_Hy << endl;
	for (i = 0; i < size_Hy_y; i++)
	{
		for (j = 0; j < size_Hy_x; j++){
			cout << Hy[i][j] << "\t";
		}
	}
	cout << endl;
}

void H::Hx_save2file()
{
	int i, j;
	fstream myfile;
	myfile.open("Hx.txt", ios::app);

	for ( i = 0; i < size_Hx_y; i++){
		for ( j = 0; j < size_Hx_x; j++){
			myfile << Hx[i][j] << "\t";
		}
		myfile << endl;
	}
	myfile << endl;
	myfile.close();
}

void H::Hy_save2file()
{
	int i, j;
	fstream myfile;

	myfile.open("Hy.txt", ios::app);

	for (i = 0; i < size_Hy_y; i++){
		for (j = 0; j < size_Hy_x; j++){
			myfile << Hy[i][j] << "\t";
		}
		myfile << endl;
	}
	myfile << endl;
	myfile.close();
}
