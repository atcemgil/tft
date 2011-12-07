/*
 * author: ck
 * 09.04.2011
 * devised from CUDA's matrixMul.cu
 */

// Utilities and system includes
#include <shrUtils.h>
#include "cutil_inline.h"
#include <iostream>
// Thread block size
#define BLOCK_SIZE 16


// Tensor .* operation. Multiply corresponding entries of tensors A,B of same size 
// Store the result in tensor C
// C = A .* B : 

// specify dimensions common to all tensors (A,B,C)
size_t crdlies[] = {2,3,2,2};
size_t nDims = 4;

// includes, kernels
#include <tensorMul_kernel.cu>

static char *sSDKsample = "tensorMul";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void incrementalInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

//extern "C"
//void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  printf("[ %s ]\n", sSDKsample);

  shrSetLogFileName ("tensorMul.txt");
  shrLog("%s Starting...\n\n", argv[0]);

  runTest(argc, argv);

  shrEXIT(argc, (const char**)argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
  if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
      cutilDeviceInit(argc, argv);
    }
  else
    {
      cudaSetDevice(cutGetMaxGflopsDeviceId());
    }

  int devID;
  cudaDeviceProp props;

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

  // set seed for rand()
  srand(2011);

  //int tmp=1;
  //shrGetCmdLineArgumenti(argc, (const char**)argv, "sizemult", &tmp);

  std::cout << "sizes:";
  for (int i=0; i<nDims; i++){
    std::cout << " " << crdlies[i] ;
  }
  std::cout << "\n\n";
  //shrLog("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n",
  //       WA, HA, WB, HB, WC, HC);


  size_t h_strides[nDims];
  size_t total_size = 1;
  for (size_t i=0; i< nDims; i++) {
    h_strides[i] = crdlies[i]>1 ? total_size : 0;
    total_size *= crdlies[i];
  };
  std::cout << "total size " << total_size << std::endl;
  

  // allocate host memory for matrices A and B
  unsigned int size_A = total_size;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)malloc(mem_size_A);
  unsigned int size_B = total_size;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)malloc(mem_size_B);


  // initialize host memory
  incrementalInit(h_A, size_A);
  incrementalInit(h_B, size_B);


  std::cout << "h_strides:\n";
  for (int i=0; i<nDims; i++){
    std::cout << h_strides[i] << std::endl;
  }

  // allocate device memory
  //size_t pitch_A;
  float* d_A;
  cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
  //cutilSafeCall( cudaMallocPitch(&d_A, &pitch_A, WA*sizeof(float), HA) );
  //size_t pitch_B;
  float* d_B;
  cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));
  //cutilSafeCall( cudaMallocPitch(&d_B, &pitch_B, WB*sizeof(float), HB) );

  size_t* d_strides;
  cutilSafeCall(cudaMalloc((void**) &d_strides, nDims*sizeof(size_t)));

  // copy host memory to device
  cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMemcpy(d_strides, h_strides, nDims*sizeof(size_t), cudaMemcpyHostToDevice) );

  // allocate device memory for result
  unsigned int size_C = total_size;
  unsigned int mem_size_C = sizeof(float) * size_C;
  //size_t pitch_C;
  float* d_C;
  cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));
  //cutilSafeCall( cudaMallocPitch(&d_C, &pitch_C, WC*sizeof(float), HC) );

  // allocate host memory for the result
  float* h_C = (float*) malloc(mem_size_C);




  // setup execution parameters
  //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  //dim3 grid(WC / threads.x, HC / threads.y);
  int blocks=BLOCK_SIZE;
  int threads=512;

  // kernel warmup
  tensorMul<<< blocks, threads >>>(d_C, d_A, d_B, d_strides, nDims, total_size);
  cudaThreadSynchronize();

  // create and start timer
  shrLog("Run Kernels...\n\n");
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute the kernel
  int nIter = 30;
  for (int j = 0; j < nIter; j++) {
    tensorMul<<< blocks, threads >>>(d_C, d_A, d_B, d_strides, nDims, total_size);
  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  cudaThreadSynchronize();
  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
  double dNumOps = 2.0 * total_size;
  double gflops = 1.0e-9 * dNumOps/dSeconds;

  //Log througput, etc
  shrLogEx(LOGBOTH | MASTER, 0, "tensorMul, Throughput = %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops, NumDevsUsed = %d, Workgroup = %u\n",
           gflops, dSeconds, dNumOps, 1, threads);
  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C,
                           cudaMemcpyDeviceToHost) );

  std::cout << std::endl << std::endl << "input A:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" <<  h_A[i]  << std::endl;
  }

  std::cout << std::endl << std::endl << "input B:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" <<  h_B[i]  << std::endl;
  }

  std::cout << std::endl << std::endl << "output C:" << std::endl;
  for (int i=0; i<total_size ; i++){
    std::cout << i << "\t" <<  h_C[i]  << std::endl;
  }

  // compute reference solution
  /*    shrLog("\nCheck against Host computation...\n\n");
        float* reference = (float*)malloc(mem_size_C);
        computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

        // check result
        shrBOOL res = shrCompareL2fe(reference, h_C, size_C, 1.0e-6f);
        if (res != shrTRUE)
        {
        printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
        }
        shrLog("%s \n\n", (shrTRUE == res) ? "PASSED" : "FAILED");
  */
  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  //free(reference);
  cutilSafeCall(cudaFree(d_A));
  cutilSafeCall(cudaFree(d_B));
  cutilSafeCall(cudaFree(d_C));

  cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void incrementalInit(float*data, int size)
{
  for (int i=1 ; i<=size; i++){
    data[i] = i;
  }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
  shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
  int i,j,k;
  int error_count=0;
  for (j = 0; j < height; j++)
    {
      if (error_count < iListLength)
        {
          shrLog("\n  Row %d:\n", j);
        }
      for (i = 0; i < width; i++)
        {
          k = j * width + i;
          float fDiff = fabs(data1[k] - data2[k]);
          if (fDiff > fListTol)
            {
              if (error_count < iListLength)
                {
                  shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
              error_count++;
            }
        }
    }
  shrLog(" \n  Total Errors = %d\n\n", error_count);
}
