/*
 * author: ck
 * 26.04.2011
 * devised from sgemm.cu of Brian Dushaw
 */

#include "mex.h"
#include "cublas.h"
#include "cutil_inline.h"
#include <iostream>

#define BLOCK_SIZE 16

// Tensor .* operation. Multiply corresponding entries of tensors A,B of same size
// Store the result in tensor C
// C = A .* B






__global__ void
tensorMul( float* C, float* A, float* B, size_t total_size)
{
  // Block index
  size_t bx = blockIdx.x;
  //int by = blockIdx.y;

  // Thread index
  size_t tx = threadIdx.x;
  //int ty = threadIdx.y;

  size_t threadsPerblock = blockDim.x * blockDim.y * blockDim.z;
  size_t thread_id = bx * threadsPerblock + tx;

  if ( thread_id  < total_size ){
    C[thread_id] = A[thread_id] * B[thread_id];
  }

}








void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int col_size = mxGetN(prhs[0]);
  int row_size = mxGetM(prhs[0]);
  int total_size = col_size * row_size;

  // arrange Matlab output storage
    mwSize argMatDims[2];
  argMatDims[0]=row_size;
  argMatDims[1]=col_size;
  plhs[0] = mxCreateNumericArray(2,argMatDims,mxSINGLE_CLASS,mxREAL); 
  float* output = (float*) mxGetData(plhs[0]);

  // allocate device memory
  unsigned int mem_size_mat = sizeof(float) * total_size;
  float* d_A;
  cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_mat));
  float* d_B;
  cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_mat));
  //size_t* d_strides;
  //cutilSafeCall(cudaMalloc((void**) &d_strides, nDims*sizeof(size_t)));

  // copy host memory to device
  cutilSafeCall(cudaMemcpy(d_A, prhs[0], mem_size_mat, cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMemcpy(d_B, prhs[1], mem_size_mat, cudaMemcpyHostToDevice) );
  //cutilSafeCall(cudaMemcpy(d_strides, h_strides, nDims*sizeof(size_t), cudaMemcpyHostToDevice) );

  // allocate device memory for result
  float* d_C;
  cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_mat));

  // setup execution parameters
  //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  //dim3 grid(WC / threads.x, HC / threads.y);
  int blocks=BLOCK_SIZE;
  int threads=512;

  // kernel warmup
  tensorMul<<< blocks, threads >>>(d_C, d_A, d_B, total_size);
  cudaThreadSynchronize();

  // create and start timer
  std::cout << "Run Kernels...\n\n" << std::endl;

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute the kernel
  int nIter = 30;
  for (int j = 0; j < nIter; j++) {
    tensorMul<<< blocks, threads >>>(d_C, d_A, d_B, total_size);
  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  cudaThreadSynchronize();
  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
  double dNumOps = 2.0 * total_size;
  double gflops = 1.0e-9 * dNumOps/dSeconds;
  std::cout << "tensorMul, Throughput = "<< gflops  << " GFlop/s, Time = " << dSeconds << " s, Size = " << dNumOps <<  " Ops, NumDevsUsed = 1, Workgroup = " << threads << "\n" ;

  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(output, d_C, mem_size_mat, cudaMemcpyDeviceToHost) );

  std::cout << std::endl << std::endl << "input A:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" << ((float*)mxGetData(prhs[0]))[i] << std::endl;
  }

  std::cout << std::endl << std::endl << "input B:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" << ((float*)mxGetData(prhs[1]))[i]  << std::endl;
  }

  std::cout << std::endl << std::endl << "output C:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" <<  output[i]  << std::endl;
  }


  // clean up memory
  //free(h_A);
  //free(h_B);
  //free(h_C);
  //free(reference);
  cutilSafeCall(cudaFree(d_A));
  cutilSafeCall(cudaFree(d_B));
  cutilSafeCall(cudaFree(d_C));

  cudaThreadExit();


}


