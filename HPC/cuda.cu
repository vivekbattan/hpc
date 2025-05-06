#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define BLOCK_SIZE 16

__global__ void vectorAdd(int *a, int *b, int *c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        c[idx] = a[idx] + b[idx];
}

__global__ void matrixMul(int *a, int *b, int *c, int m, int k, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < m && col < n){
        int sum = 0;
        for(int i = 0; i < k; i++){
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void fillIntVector(int *arr, int size){
    for(int i = 0; i < size; i++)arr[i] = rand() % 100;
}

int main(){
    srand(time(0));

    // Vector Addition

    int size;
    cout<<"Enter vector size: ";
    cin>>size;

    int *h_vecA = new int[size];
    int *h_vecB = new int[size];
    int *h_vecC = new int[size];

    fillIntVector(h_vecA, size);
    fillIntVector(h_vecB, size);

    int *d_vecA, *d_vecB, *d_vecC;
    cudaMalloc(&d_vecA, size * sizeof(int));
    cudaMalloc(&d_vecB, size * sizeof(int));
    cudaMalloc(&d_vecC, size * sizeof(int));

    cudaMemcpy(d_vecA, h_vecA, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecB, h_vecB, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    cudaEvent_t startVec, stopVec;
    cudaEventCreate(&startVec);
    cudaEventCreate(&stopVec);
    cudaEventRecord(startVec);

    vectorAdd<<<grid, block>>>(d_vecA, d_vecB, d_vecC, size);
    cudaDeviceSynchronize();

    cudaEventRecord(stopVec);
    cudaEventSynchronize(stopVec);
    float vectorAddTime;
    cudaEventElapsedTime(&vectorAddTime, startVec, stopVec);

    cudaMemcpy(h_vecC, d_vecC, size * sizeof(int), cudaMemcpyDeviceToHost);

    cout<<"Time taken for vector addition: "<<vectorAddTime<<"ms\n";
    cout<<"Resultant Vector: \n";
    for(int i = 0; i < size; i++)cout<<h_vecC[i]<<" ";
    cout<<endl;

// Matrix Multiplication

    cout<<"Matrix 1 Dimension (m x k) \nMatrix 2 Dimension (k x n) \nEnter value of m, k, n: ";
    int m,k,n;
    cin>>m>>k>>n;

    int *h_matA = new int[m*k]; 
    int *h_matB = new int[k*n];
    int *h_matC = new int[m*n];

    fillIntVector(h_matA, m*k);
    fillIntVector(h_matB, k*n);

    int *d_matA, *d_matB, *d_matC;
    cudaMalloc(&d_matA, m*k*sizeof(int));
    cudaMalloc(&d_matB, k*n*sizeof(int));
    cudaMalloc(&d_matC, m*n*sizeof(int));

    cudaMemcpy(d_matA, h_matA, m*k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, k*n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1)/ BLOCK_SIZE, (m + BLOCK_SIZE - 1)/ BLOCK_SIZE);

    cudaEvent_t matstart, matstop;
    cudaEventCreate(&matstart);
    cudaEventCreate(&matstop);
    cudaEventRecord(matstart);

    matrixMul<<<gridDim, blockDim>>>(d_matA, d_matB, d_matC, m, k, n);
    cudaDeviceSynchronize();

    cudaEventRecord(matstop);
    cudaEventSynchronize(matstop);
    float mat_time;
    cudaEventElapsedTime(&mat_time, matstart, matstop);

    cudaMemcpy(h_matC, d_matC, m*n*sizeof(int), cudaMemcpyDeviceToHost);

    cout<<"Time taken for matrix multiplication: "<<mat_time<<"ms\n";
    cout<<"Resultant Matrix:\n";
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)cout<<h_matC[i * n + j]<<" ";
        cout<<endl;
    }

    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_vecC);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    delete[] h_vecA;
    delete[] h_vecB;
    delete[] h_vecC;
    delete[] h_matA;
    delete[] h_matB;
    delete[] h_matC;
}
// nvcc cuda.cu -o cuda -lcudart -lcublas -lcublasLt -lcudnn -lcurand
// nvcc cuda.cu -o cuda
// ./cuda