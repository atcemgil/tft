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

// setup execution parameters
//dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//dim3 grid(WC / threads.x, HC / threads.y);
int blocks=BLOCK_SIZE;
int threads=512;


// Tensor .* operation. Multiply corresponding entries of tensors A,B of same size
// Store the result in tensor C

// two operators are available 
// dot product: multiplies each element of input objects elementwise
// C = A .* B

// contract product: performs matrix multiplication if elements are 2 dimensional
// C = A * B




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

  // total size of the related objects
  // multiplication of all cardinalities
  size_t total_cardinality;
};

// cuda tensor object
struct ct{

  // related configuration object
  ct_config* config;

  // defines size of each dimension for this tensor
  // must be allocated dynamically as an array of type size_t
  // size of the array must be equal to config.ndims
  size_t* cardinalities;

  // size of the corresponding data
  size_t mem_size;

  // points to the values of this tensor
  float* data;
};

// compact structure carying pointers to elements of a cudatensor on the device
struct dev_ct_ptrs{
  ct* ct;
  ct_config* ctc;
  size_t* cardinalities;
  float* data;
};


// multiply corresponding elemens of A, B tensors, put result in tensor C
__global__ void
tensorDotStar( ct* C, ct* A, ct* B)
{
  // Block index
  size_t bx = blockIdx.x;
  //int by = blockIdx.y;

  // Thread index
  size_t tx = threadIdx.x;
  //int ty = threadIdx.y;

  size_t threadsPerblock = blockDim.x * blockDim.y * blockDim.z;
  size_t thread_id = bx * threadsPerblock + tx;

  if ( thread_id  < A->config->total_cardinality ){
    C->data[thread_id] = A->data[thread_id] * B->data[thread_id];
  }

}

__global__ void
assignCudatensor( ct* c, ct_config* ctc, size_t* cards, size_t mem_size, float* data){
  c->config = ctc;
  c->cardinalities = cards;
  c->mem_size = mem_size;
  c->data = data;
}



void print_ct_config(char* txt, ct_config* ctc){
  std::cout << txt << std::endl;

  std::cout << "Number of dimensions " << (int) (ctc->ndims) << std::endl;

  std::cout << "Cardinalities for each dimension of this configuration " << std::endl;
  size_t i=0;
  for ( i=0; i< ctc->ndims; i++){
    std::cout << ctc->cardinalities[i] << " ";
  }
  std::cout << "\nTotal cardinality: " << ctc->total_cardinality << std::endl;
  std::cout << std::endl;
}


void print_ct(char* txt, ct* ct, bool print_config=false, bool printdata=false){
  
  std::cout << txt << std::endl;

  if (print_config) print_ct_config(txt, ct->config);

  std::cout << "Mem size " << ct->mem_size << std::endl;

  std::cout << "Cardinalities for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->config->ndims; i++){
    std::cout << ct->cardinalities[i] << " ";
  }
  std::cout << std::endl;

  if (printdata){
    std::cout << "Data" << std::endl;
    for (size_t i=0; i< ct->config->total_cardinality; i++){
	std::cout << ct->data[i] << " ";
    }
  }

  std::cout << std::endl << std::endl;
}

// returns a dev_ct_ptrs struct with information about the cudatensor generated on the device
dev_ct_ptrs prepareDeviceTensor(ct_config* h_ctc, ct_config* d_ctc, ct* h_ct, const mxArray* data){

  // generate h_ct
  
  h_ct->config = h_ctc;
  h_ct->cardinalities = (size_t*) malloc(sizeof(size_t)*h_ctc->ndims);

  // assign cardinalities for the tensor objects
  const mwSize *dims_c = mxGetDimensions(data);
  for (size_t i=0; i<h_ctc->ndims; i++){
    h_ct->cardinalities[i] = dims_c[i];
    std::cout << "dim "<< i << " cardinality assignment: " << h_ct->cardinalities[i] << "<-" << dims_c[i] << std::endl;
  }


  // assign h_ct host data
  //size_t elnum = (size_t) mxGetNumberOfElements(data);
  h_ct->mem_size= sizeof(float) * h_ctc->total_cardinality;
  h_ct->data = (float*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (float*)mxGetData(data), h_ct->mem_size);

  print_ct("prepareDeviceTensor h_ct",h_ct,true,true);


  // allocate d_ct
  ct* d_ct;
  cutilSafeCall(cudaMalloc((void**) &d_ct, sizeof(ct)));

  // allocate d_ct contents
  // config -> d_ctc
  size_t* tmp_card;
  cutilSafeCall(cudaMalloc((void**)&tmp_card, sizeof(size_t)*h_ctc->ndims));
  cutilSafeCall(cudaMemcpy(tmp_card, h_ct->cardinalities, sizeof(size_t)*h_ctc->ndims  ,cudaMemcpyHostToDevice));

  float* tmp_data;
  cutilSafeCall(cudaMalloc((void**)&tmp_data, h_ct->mem_size));
  cutilSafeCall(cudaMemcpy(tmp_data, h_ct->data, h_ct->mem_size, cudaMemcpyHostToDevice));

  // put contents of d_ct in their places on the device
  assignCudatensor<<<1, 1>>>(d_ct, d_ctc, tmp_card, h_ct->mem_size, tmp_data);


  dev_ct_ptrs dcp;
  dcp.ct=d_ct;
  dcp.ctc=d_ctc;
  dcp.cardinalities=tmp_card;
  dcp.data=tmp_data;
  return dcp;
}

ct_config* prepareDeviceTensorConfig(ct_config* ctc, const mxArray* sampleObject){
  ctc->ndims = mxGetNumberOfDimensions(sampleObject);
  ctc->cardinalities = (size_t*) malloc(sizeof(size_t)*ctc->ndims);
  const mwSize *dims = mxGetDimensions(sampleObject);
  ctc->total_cardinality = 1;
  for (size_t i=0; i<ctc->ndims; i++){
    ctc->cardinalities[i] = dims[i];
    ctc->total_cardinality *= dims[i];
  }

  // transfer to device
  ct_config* d_ctc;
  cutilSafeCall(cudaMalloc((void**) &d_ctc, sizeof(ct_config) ));
  cutilSafeCall(cudaMemcpy( d_ctc , ctc, sizeof(ct_config), cudaMemcpyHostToDevice) );

  return d_ctc;
}

void print_device_ct(char* txt,dev_ct_ptrs* dcp, ct* host_ct){
  ct tmp_ct; 
  cutilSafeCall(cudaMemcpy(&tmp_ct, dcp->ct, sizeof(ct), cudaMemcpyDeviceToHost));

  tmp_ct.config = (ct_config*) malloc( sizeof(ct_config) );
  tmp_ct.cardinalities = (size_t*) malloc( host_ct->config->ndims );
  tmp_ct.data = (float*) malloc(host_ct->mem_size);

  cutilSafeCall(cudaMemcpy(tmp_ct.data, dcp->data, host_ct->mem_size, cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(tmp_ct.config, dcp->ctc, sizeof(ct_config), cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(tmp_ct.cardinalities, dcp->cardinalities, sizeof(size_t)*host_ct->config->ndims, cudaMemcpyDeviceToHost));

  print_ct(txt,&tmp_ct,false,true);
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  // ASSUMES target tensors are of the same dimension
  ct_config h_ctc;
  ct_config* d_ctc = prepareDeviceTensorConfig(&h_ctc,prhs[0]);

  print_ct_config("Host ctc",&h_ctc);

  ct_config tmp_ctc;
  cutilSafeCall(cudaMemcpy(&tmp_ctc, d_ctc, sizeof(ct_config), cudaMemcpyDeviceToHost));
  print_ct_config("Device tmp ctc",&tmp_ctc);


  // input tensor A
  ct h_it_A;
  dev_ct_ptrs d_A=prepareDeviceTensor(&h_ctc, d_ctc, &h_it_A, prhs[0]);

  // input tensor B
  ct h_it_B;
  dev_ct_ptrs d_B=prepareDeviceTensor(&h_ctc, d_ctc, &h_it_B, prhs[1]);

  // output tensor C
  ct h_ot_C;
  // prepare MATLAB storage
  // calculate total cardinalities for all objects
  mwSize argMatDims[h_ctc.ndims];
  for (size_t i=0; i<h_ctc.ndims; i++){
    argMatDims[i] = h_ctc.cardinalities[i];
  }
  plhs[0] = mxCreateNumericArray(h_ctc.ndims,argMatDims,mxSINGLE_CLASS,mxREAL); 
  float* m_C = (float*) mxGetPr(plhs[0]);
  dev_ct_ptrs d_C=prepareDeviceTensor(&h_ctc, d_ctc, &h_ot_C, plhs[0]);

  bool printdata=true;
  print_ct("Host A",&h_it_A,false,printdata);
  print_ct("Host B",&h_it_B,false,printdata);
  print_ct("Host C",&h_ot_C,false,printdata);


  print_device_ct("Device A",&d_A, &h_it_A);
  print_device_ct("Device B",&d_B, &h_it_B);
  print_device_ct("Device C",&d_C, &h_ot_C);
  


  // allocate device memory for result
  // kernel warmup
  tensorDotStar<<< blocks, threads >>>(d_C.ct, d_A.ct, d_B.ct);

  cudaThreadSynchronize();

  // create and start timer
  std::cout << "Run Kernels...\n\n" << std::endl;

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute the kernel
  int nIter = 30;
  for (int j = 0; j < nIter; j++) {
    tensorDotStar<<< blocks, threads >>>(d_C.ct, d_A.ct, d_B.ct);
  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  cudaThreadSynchronize();
  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  //double dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
  //double dNumOps = 2.0 * total_size;
  //double gflops = 1.0e-9 * dNumOps/dSeconds;
  //std::cout << "tensorMul, Throughput = "<< gflops  << " GFlop/s, Time = " << dSeconds << " s, Size = " << dNumOps <<  " Ops, NumDevsUsed = 1, Workgroup = " << threads << "\n" ;

  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(m_C, d_C.data, h_ot_C.mem_size, cudaMemcpyDeviceToHost) ); // assumes same size

  // clean up memory
  //free(h_A);
  //free(h_B);
  //free(h_C);
  //free(reference);

  // wrong
  //cutilSafeCall(cudaFree());
  //cutilSafeCall(cudaFree(d_it_B));
  //cutilSafeCall(cudaFree(d_it_A)); //->C

  print_device_ct("Result\nDevice C",&d_C, &h_ot_C);

  cudaThreadExit();


  // required to avoid memory leak?
  delete h_ctc.cardinalities;
  delete h_it_A.cardinalities;
  delete h_it_B.cardinalities;

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
