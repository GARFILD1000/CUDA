#include <cufft.h>
#include <stdio.h>

#include <cstdio>
#include <iostream>
#include <fstream>

#include <malloc.h>

#define NX 16384
//#define NX 32
#define BATCH 1
#define pi 3.141592f

__global__ void gInitData(cufftComplex *data){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    float x = i * 2.0f * pi/(NX);
    data[i].x = cosf(x) - 3.0f*sinf(x);
    data[i].y = 0.0f;
}

void loadData(cufftReal* values){
    /* FILE *file = NULL;fopen("wolf.csv", "r");
    for(int i = 0; i < NX; i++) {
		int d;
		fscanf(file, "%*d%d", &d);
		if(d != noData) 
			data[i].x = old = d;
		else 
			data[i].x = old;
		
		data[i].y = 0;
	}
    
    if(file != NULL){
    fclose(file);
    }*/
     std::ifstream in;          // поток для записи
     in.open("wolf.csv"); // окрываем файл для записи
     int counter = 0;
     int val = 0;
     int idx = 0;
     while (in >> val && idx < NX)
     {
        //float x = counter * 2.0f * pi/(NX);
        //if(counter%2==0){
            values[idx] = val;//cosf(x) - 3.0f*sinf(x);
        //}
        //else{
        //    values[idx].y = 0;
            idx++;
        //}
        counter++;
        //values[counter].y = 0.0f;
     }
}

void saveData(cufftComplex* values){
    FILE *file = fopen("wolfFourier.csv", "w");
    if(file == NULL){
        return;
    }
    double intensity;
    for(int i = 0; i < NX / 2 + 1; i++){
        intensity = sqrt((values[i].x/NX * values[i].x/NX) + (values[i].y/NX * values[i].y/NX));
        printf("intensity %g\n", intensity);
        //fwrite(&intensity, sizeof(float), 1, file);
        fprintf(file, "%g\n", intensity);
    }
    
    
     /*std::ifstream in;          // поток для записи
     in.open("wolf.csv"); // окрываем файл для записи
     int counter = 0;
     int val = 0;
     int idx = 0;
     while (in >> val && idx < NX)
     {
        //float x = counter * 2.0f * pi/(NX);
        //if(counter%2==0){
            values[idx] = val;//cosf(x) - 3.0f*sinf(x);
        //}
        //else{
        //    values[idx].y = 0;
            idx++;
        //}
        counter++;
        //values[counter].y = 0.0f;
     }*/
     fclose(file);
}


int main(){
    cufftHandle forwardPlan;
    cufftHandle backwardPlan;
    
    cufftComplex *result_d;
    cufftComplex *result_h;
    
    cufftReal *values_d;
    cufftReal *values_h;
    
    cudaMalloc((void**)&values_d, sizeof(cufftReal)*(NX));
    cudaMallocHost((void**)&values_h, sizeof(cufftReal)*(NX));
    
    cudaMallocHost((void**)&result_h, sizeof(cufftComplex)*(NX/2+1));
    cudaMalloc((void**)&result_d, sizeof(cufftComplex)*(NX/2+1));
    
    loadData(values_h);
    
    printf("Start Values:\n");
    for(int i = 0; i < NX && i < 100; i++){
        printf("real %g\n", values_h[i]);
    }
    
    cudaMemcpy(values_d, values_h, (NX)*sizeof(cufftReal), cudaMemcpyHostToDevice);
    
    if(cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to copy size %d\n", NX);
        return -1;
    }
    
    if(cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate size %d\n", NX);
        return -1;
    }
    
    if (cufftPlan1d(&forwardPlan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: forwardPlan creation failed");
        return -1;
    }
    
    if (cufftExecR2C(forwardPlan, (cufftReal*)values_d, (cufftComplex*)result_d) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Execute CUFFT_FORWARD failed");
        return -1;
    }
    
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize");
        return -1;
    }
    
    cudaMemcpy(result_h, result_d, (NX/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    printf("\nResult Values:\n");
    
    saveData(result_h);
    
        for(int i = 0; i < NX/2+1 && i < 100; i++){
        //printf("real %g ", result_h[i].x/NX);
        //printf("img %g\n", result_h[i].y/NX);
        double intensity = sqrt((result_h[i].x/NX * result_h[i].x/NX) + (result_h[i].y/NX * result_h[i].y/NX));
        printf("intensity %g\n", intensity);
    }
    
    if (cufftPlan1d(&backwardPlan, NX, CUFFT_C2R, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: forwardPlan creation failed");
        return -1;
    }
    
    if (cufftExecC2R(backwardPlan, (cufftComplex*)result_d, (cufftReal*)values_d) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Execute CUFFT_FORWARD failed");
        return -1;
    }
    
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize");
        return -1;
    }
    
    cudaMemcpy(values_h, values_d, (NX)*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    
    printf("\nCheck Values:\n");
    
    for(int i = 0; i < NX && i < 100; i++){
        printf("real %g\n", values_h[i]/NX);
    }

    cufftDestroy(forwardPlan);
    cudaFree(result_d);
    cudaFree(result_h);
    cudaFree(values_h);
    cudaFree(values_d); 
}
