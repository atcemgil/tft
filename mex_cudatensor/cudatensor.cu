/*
 * author: ck
 * 26.04.2011
 * devised from sgemm.cu of Brian Dushaw
 */

#include "mex.h"
#include "cublas.h"
#include "cutil_inline.h"
#include <iostream>
#include <algorithm>

#define BLOCK_SIZE 16

// Tensor .* operation. Multiply corresponding entries of tensors A,B of same size
// Store the result in tensor C
// C = A .* B



void print( const mxArray *prhs[], float* output, int total_size);


// cuda tensor operation configuration object
struct ct_config{
  // defines how many dimensions are there
  size_t ndims;

  // defines the maximum possible size of each dimension 
  //   for all tensors using this configuration
  // must be allocated dynamically as an array of type size_t
  // size of the array must be equal to ndims
  size_t* cardinalities;

};

// cuda tensor object
struct ct{

  // related configuration object
  ct_config* config;

  // defines size of each dimension for this tensor
  // must be allocated dynamically as an array of type size_t
  // size of the array must be equal to config.ndims
  size_t* cardinalities;

  // points to the values of this tensor
  float* data;
};



/*
// multiply corresponding elemens of A, B tensors, put result in tensor C
__global__ void
tensorDotStar( float* C, float* A, float* B, size_t total_size)
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
*/


void print_ct_config(ct_config* ctc){
  std::cout << "Number of dimensions " << (int) (ctc->ndims) << std::endl
	    << "Cardinalities for each dimension of this configuration " << std::endl;
  size_t i=0;
  for ( i=0; i< ctc->ndims; i++){
    std::cout << ctc->cardinalities[i] << " ";
  }
}


void print_ct(ct* ct, bool print_config=false){

  if (print_config) print_ct_config(ct->config);

  std::cout << std::endl << "Cardinalities for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->config->ndims; i++){
    std::cout << ct->cardinalities[i] << " ";
  }
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /*
  //test block
  int i=1;
  if (i==1){
    std::cout << "get data test " << std::endl;
    int col_size = mxGetN(prhs[0]);
    int row_size = mxGetM(prhs[0]);
    int total_size = col_size * row_size;

    mwSize argMatDims[2];
    argMatDims[0]=row_size;
    argMatDims[1]=col_size;
    plhs[0] = mxCreateNumericArray(2,argMatDims,mxSINGLE_CLASS,mxREAL); 
    double* output = (double*) mxGetPr(plhs[0]);

    print(prhs, output, total_size);

    return;
  }
  */

  // ASSUMES target tensors are of same dimension
  const mxArray* it_mx_A = prhs[0];
  const mxArray* it_mx_B = prhs[1];

  ///// -> bunları fonksiyon yap!

  ct_config ctc;
  ctc.ndims = mxGetNumberOfDimensions(it_mx_A); // dimensions are assumed to be the same
  ctc.cardinalities = (size_t*) malloc(sizeof(size_t)*ctc.ndims);

  ct it_A;
  const mwSize *dims_A = mxGetDimensions(it_mx_A);
  it_A.config = &ctc;
  it_A.cardinalities = (size_t*) malloc(sizeof(size_t)*ctc.ndims);

  ct it_B;
  const mwSize *dims_B = mxGetDimensions(it_mx_B);
  it_B.config = &ctc;
  it_B.cardinalities = (size_t*) malloc(sizeof(size_t)*ctc.ndims);


  // assign cardinalities for config and tensor objects
  for (size_t i=0; i<ctc.ndims; i++){
    ctc.cardinalities[i] = std::max(dims_A[i], dims_B[i]);
    it_A.cardinalities[i] = dims_A[i];
    it_B.cardinalities[i] = dims_B[i]; 
  }

  print_ct_config(&ctc);  
  print_ct(&it_A);
  print_ct(&it_B);

  
  // arrange Matlab output storage
  // ??? zero pad input for matching cardinalities ???
  // ??? zero pad output ???
  // now zero pads output with maximum cardinalities

  // calculate total cardinalities for all objects
  size_t card_total = 1;
  size_t card_A = 1;
  size_t card_B = 1;

  mwSize argMatDims[ctc.ndims];
  for (size_t i=0; i<ctc.ndims; i++){
    size_t tmp = std::max(dims_A[i], dims_B[i]);
    argMatDims[i] = tmp;
    card_total *= tmp;
    card_A *= dims_A[i];
    card_B *= dims_B[i];
  }
  plhs[0] = mxCreateNumericArray(ctc.ndims,argMatDims,mxSINGLE_CLASS,mxREAL); 
  float* output = (float*) mxGetPr(plhs[0]);


  // allocate device memory

  // copy data
  size_t mem_size_A = sizeof(float) * card_A;
  float* d_A;
  cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
  float* d_B;
  size_t mem_size_B = sizeof(float) * card_B;
  cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));
  //size_t* d_strides;
  //cutilSafeCall(cudaMalloc((void**) &d_strides, nDims*sizeof(size_t)));

  // copy host memory to device
  cutilSafeCall(cudaMemcpy(d_A, (float*) mxGetPr(prhs[0]), mem_size_mat, cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMemcpy(d_B, (float*) mxGetPr(prhs[1]), mem_size_mat, cudaMemcpyHostToDevice) );
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
  //tensorMul<<< blocks, threads >>>(d_C, d_A, d_B, total_size);
  cudaThreadSynchronize();

  // create and start timer
  std::cout << "Run Kernels...\n\n" << std::endl;

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute the kernel
  int nIter = 30;
  for (int j = 0; j < nIter; j++) {
    //tensorDotStar<<< blocks, threads >>>(d_C, d_A, d_B, total_size);
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

  //std::cout << " AFTER " << std::endl;
  //print(prhs, output, total_size);

  // clean up memory
  //free(h_A);
  //free(h_B);
  //free(h_C);
  //free(reference);
  cutilSafeCall(cudaFree(d_A));
  cutilSafeCall(cudaFree(d_B));
  cutilSafeCall(cudaFree(d_C));

  cudaThreadExit();


  // required to avoid memory leak?
  delete ctc.cardinalities;
  delete it_A.cardinalities;
  delete it_B.cardinalities;

}


void print( const mxArray *prhs[], float* output, int total_size){
  std::cout << "\ntotal_size " << total_size << std::endl; 

  std::cout << std::endl << std::endl << "input A:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" << ((float*)mxGetPr(prhs[0]))[i] << std::endl;
  }

  std::cout << std::endl << std::endl << "input B:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" << ((float*)mxGetPr(prhs[1]))[i]  << std::endl;
  }

  std::cout << std::endl << std::endl << "output C:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" <<  output[i]  << std::endl;
  }
}
