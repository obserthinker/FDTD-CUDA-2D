#include "H.h"

H::H(src source)
{
	Hx_init(source);
	Hy_init(source);
	coe_H_set(source);
}

void H::Hx_init(src source)
{
    //size_Hx = source.size_space_x * (source.size_space_y - 1);
	size_Hx_x = source.size_space_x;
	size_Hx_y = source.size_space_y + 1;
	size_Hx = size_Hx_x * size_Hx_y;

    Hx = (float*)malloc(size_Hx * sizeof(float));
    cudaMalloc(&dev_Hx, size_Hx * sizeof(float));

    for (int i = 0; i < size_Hx; ++i){
    	Hx[i] = 0.f;
    }

	fstream myfile;
	myfile.open("Hx.txt", ios::out);
	myfile.close();
}

void H::Hy_init(src source)
{
    //size_Hy = (source.size_space_x - 1) * source.size_space_y;
	size_Hy_x = source.size_space_x + 1;
	size_Hy_y = source.size_space_y;
	size_Hy = size_Hy_x * size_Hy_y;

    Hy = (float*)malloc(size_Hy * sizeof(float));
    cudaMalloc(&dev_Hy, size_Hy * sizeof(float));

    for (int i = 0; i < size_Hy; ++i){
    	Hy[i] = 0.f;
    }

	fstream myfile;
	myfile.open("Hx.txt", ios::out);
	myfile.close();
}

void H::coe_H_set(src source)
{
	coe_H = source.dt / (mu * source.dz);
}

void H::Hx_transfer_host_device()
{
	cudaMemcpy(dev_Hx, Hx, size_Hx * sizeof(float), cudaMemcpyHostToDevice);
}

void H::Hx_transfer_device_host()
{
	cudaMemcpy(Hx, dev_Hx, size_Hx * sizeof(float), cudaMemcpyDeviceToHost);
}

void H::Hy_transfer_host_device()
{
	cudaMemcpy(dev_Hy, Hy, size_Hy * sizeof(float), cudaMemcpyHostToDevice);
}

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
			Hx[i*size_Hx_x + j] += coe_H *
				(E.Ez[i*E.size_Ez_x + j] - E.Ez[i*E.size_Ez_x + j + 1]);
		}
	}
}

//empty, waiting to be test
void H::Hy_cmp_kernel(E E, src source)
{
	int i, j;

	for ( i = 0; i < size_Hy_y; i++){
		for ( j = 0; j < size_Hy_x; j++)
		{
			Hy[i*size_Hy_x + j] += 
				coe_H * (E.Ez[(i + 1)*E.size_Ez_x + j] - E.Ez[i*E.size_Ez_x+ j]);
		}
	}
}

void H::Hx_checkout()
{
	cout << "Hx size: " << size_Hx << endl;
	for (int i = 0; i < size_Hx; i++)
	{
		cout << Hx[i] << "\t";
	}
	cout << endl;
}

void H::Hy_checkout()
{
	cout << "Hy size: " << size_Hy << endl;
	for (int i = 0; i < size_Hy; i++)
	{
		cout << Hy[i] << "\t";
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
			myfile << Hx[i * size_Hx_x + j] << "\t";
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
			myfile << Hy[i * size_Hy_x + j] << "\t";
		}
		myfile << endl;
	}
	myfile << endl;
	myfile.close();
}