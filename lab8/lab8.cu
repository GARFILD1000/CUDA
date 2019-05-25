#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <stdio.h>
#include <cublas_v2.h>

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if(_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}

struct saxpy_functor 
{
	const float a;
	saxpy_functor(float _a) : a(_a) {}
	__host__ __device__ float operator()(float x, float y) 
	{
		return a * x + y;
	}
};

void saxpy(float a, thrust::device_vector<float>& x, thrust::device_vector<float>& y) 
{
	saxpy_functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

void print_array(float *data1, float *data2, int num_elem, const char *prefix) 
{
	printf("\n%s", prefix);
	for(int i = 0; i < num_elem; i++)
		printf("\n%d: 2.4%f 2.4%f ", i + 1, data1[i], data2[i]);
}


//compile it that way: 
//nvcc lab8.cu -lcublas -o lab8
int main() 
{
	cudaEvent_t start, stop;
	float elapsedTime;
	long vectorSize = 1 << 24;
	float alpha = 2.0f;
	
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	thrust::host_vector<float> h1(vectorSize);
	thrust::host_vector<float> h2(vectorSize);
	thrust::sequence(h1.begin(), h1.end());
	thrust::fill(h2.begin(), h2.end(), 0.4);
	
	printf("Before Thrust SAXPY\n");
	for (int i = 1; i < vectorSize; i = i << 1){
	    printf("h1[%d] = %f\n", i, h1[i]);
	    printf("h2[%d] = %f\n", i, h2[i]);
	}
	
	thrust::device_vector<float> d1 = h1;
	thrust::device_vector<float> d2 = h2;
	
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
	saxpy(alpha, d1, d2);
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	
	h2 = d2;
	h1 = d1;
	
	printf("\nTHRUST Time: %f ms\n", elapsedTime);
	
	
	printf("\nAfter Thrust SAXPY\n");
	for (int i = 1; i < vectorSize; i = i << 1){
	    printf("h1[%d] = %f\n", i, h1[i]);
	    printf("h2[%d] = %f\n", i, h2[i]);
	}

	//const int num_elem = (vectorSize);
	float *A_h, *B_h, *A_dev, *B_dev;
	
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&A_h, vectorSize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&B_h, vectorSize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&A_dev,  vectorSize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&B_dev,  vectorSize * sizeof(float)));
	
	for(int i = 0; i < vectorSize; i++) 
	{
		A_h[i] = (float) i;
		B_h[i] = 0.4f;
	}
	
	printf("\nBefore cuBLAS SAXPY\n");
	for (int i = 1; i < vectorSize; i = i << 1){
	    printf("h1[%d] = %f\n", i, A_h[i]);
	    printf("h2[%d] = %f\n", i, B_h[i]);
	}
	
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	
	const int num_rows = vectorSize;
	const int num_cols = 1;
	const size_t elem_size = sizeof(float);
	
	cublasSetMatrix(num_rows, num_cols, elem_size, A_h, num_rows, A_dev, num_rows);
	cublasSetMatrix(num_rows, num_cols, elem_size, B_h, num_rows, B_dev, num_rows);
	
	const int stride = 1;
	
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
	cublasSaxpy(cublas_handle, vectorSize, &alpha, A_dev, stride, B_dev, stride);
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	
	cublasGetMatrix(num_rows, num_cols, elem_size, A_dev, num_rows, A_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, B_dev, num_rows, B_h, num_rows);
	
	printf("\ncuBLAS Time: %f ms\n", elapsedTime);
	
	const int default_stream = 0;
	CUDA_CHECK_RETURN(cudaStreamSynchronize(default_stream));
	
	printf("\nAfter cuBLAS SAXPY\n");
	for (int i = 1; i < vectorSize; i = i << 1){
	    printf("h1[%d] = %f\n", i, A_h[i]);
	    printf("h2[%d] = %f\n", i, B_h[i]);
	}
	
	cublasDestroy(cublas_handle);
	CUDA_CHECK_RETURN(cudaFreeHost(A_h));  
	CUDA_CHECK_RETURN(cudaFreeHost(B_h)); 
	CUDA_CHECK_RETURN(cudaFree(A_dev));  
	CUDA_CHECK_RETURN(cudaFree(B_dev)); 
	
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));	
		
	return 0;
}
