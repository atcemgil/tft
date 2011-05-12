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

#include "cuPrintf.cu"

#define BLOCK_SIZE 16

// setup execution parameters
//dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//dim3 grid(WC / threads.x, HC / threads.y);
int blocks=BLOCK_SIZE;
int threads=128;


// Tensor .* operation. Multiply corresponding entries of tensors A,B of same size
// Store the result in tensor C

// two operators are available
// hadamard product: multiplies each element of input objects elementwise
// C = A .* B
// requires two input tensors A, B as input

// contract product: performs matrix multiplication if elements are 2 dimensional
// C = A * B
// requires five input arguments A, A_cardinalities, B, B_cardinalities, C_cardinalities
// objects (A,B,C) must have same number of dimensions








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
  // maximum of cardinality of input objects
  // cardinality for an object is found by multiplying object's cardinalities of each dimension
  size_t total_cardinality;

  // number of elements in the data
  size_t element_number;

  // index of the dimension to contract over
  //size_t contract_dim;
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
tensorHadamard( ct* C, ct* A, ct* B)
{
  // Block index
  size_t bx = blockIdx.x;
  //int by = blockIdx.y;

  // Thread index
  size_t tx = threadIdx.x;
  //int ty = threadIdx.y;

  size_t threadsPerblock = blockDim.x * blockDim.y * blockDim.z;
  size_t thread_id = bx * threadsPerblock + tx;

  if ( thread_id  < A->config->element_number ){
    C->data[thread_id] = A->data[thread_id] * B->data[thread_id];
  }
}

// multiply corresponding elements and contract along specified dimension
__global__ void
tensorContract( ct* C_full, ct* C, ct* A, ct* B )
{
  size_t thread_id = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.x * threadIdx.y * blockDim.y);
  size_t block_id = blockIdx.x + (blockIdx.y * gridDim.x);

  // assumes same total dimensions and cardinalities for all objects
  size_t tot_card = A->config->total_cardinality;
  size_t ndims = A->config->ndims;

  if ( thread_id  < tot_card && block_id == 0){

    //size_t uclu[3];
    //for (size_t i=0; i<3; i++) {uclu[i]=0; }

    int index_number_A=0;
    int index_number_B=0;
    int index_number_C=0;

    for (size_t obj=0; obj<2; obj++){
      ct* p;
      if      (obj==0)  p = A;
      else if (obj==1)  p = B;
      //else if (obj==2)  p = C;

      size_t t_id_rem = thread_id;
      size_t cumulative_offset_ind = 1;
      size_t cumulative_offset_elnum = 1;
      size_t cur_card_index=0;
      for (size_t card_index=0; card_index < ndims; card_index++){
        if ( t_id_rem == 0 ) break;

        //uclu[card_index] = (t_id_rem % p->config->cardinalities[card_index]) * cumulative_offset_ind;
        cur_card_index = (t_id_rem % p->config->cardinalities[card_index]) * cumulative_offset_ind;

        t_id_rem = (size_t) t_id_rem / p->config->cardinalities[card_index];

        if (p->cardinalities[card_index] != 0){
	  // int olmazsa patliyor?
	  if      (obj==0)  index_number_A += (int)cur_card_index * (int)cumulative_offset_elnum; 
	  else if (obj==1)  index_number_B += (int)cur_card_index * (int)cumulative_offset_elnum;
	  //else if (obj==2)  index_number_C += (int)cur_card_index * (int)cumulative_offset_elnum;

          // increment cumulative offset with current dimension cardinality for next loop
          // -1 for cardinalities are indexed from 1
          cumulative_offset_ind *= p->config->cardinalities[card_index] - 1 ;
          cumulative_offset_elnum *= p->config->cardinalities[card_index] ;
        }
      }
    }

    size_t tmpB = B->data[index_number_B];
    size_t tmpA= A->data[index_number_A];
    size_t tmpC_full= C_full->data[thread_id];
    cuPrintf("C[%d] %d += A[%d] %d * B[%d] %d\n",  thread_id, tmpC_full,  index_number_A, tmpA, index_number_B, tmpB);

    //~/arastir/cuda2/cudainstall/3.2/sdk/C/src/reduction/doc/reduction.pdf
    // extern __shared__ int sdata[];
    // // each thread loads one element from global to shared mem
    // unsigned int tid = threadIdx.x;
    // unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // sdata[tid] = C->data[i];
    // __syncthreads();

    // sdata[index_number_C] += A->data[index_number_A] * B->data[index_number_B];



    //cuPrintf("C_full->data[%d] = %d ", thread_id, tmpA * tmpB);
    C_full->data[thread_id] = A->data[index_number_A] * B->data[index_number_B];


    __syncthreads();

    // contract on dimensions with zero cardinality
    size_t cum_card=1;
    for (size_t card_index=0; card_index<ndims; card_index++){
      size_t current_card=C->cardinalities[card_index];

      if( current_card == 0 ) {
	// contract on this dimension

	size_t C_ind=0;
	for (size_t C_full_ind=0; C_full_ind < tot_card-1;){

	  size_t tmp1 = C->data[C_ind];
	  size_t tmp2 = C_full->data[C_full_ind];
	  size_t tmp3 = C_full->data[C_full_ind + cum_card];
	  cuPrintf("C[%d] %d +=  C_full[%d] %d + C_full[%d] %d \n", C_ind, tmp1, C_full_ind, tmp2 , C_full_ind+cum_card , tmp3);

	  C->data[C_ind] = C_full->data[C_full_ind] + C_full->data[C_full_ind+cum_card];

	  C_ind++;
	  if (C_full_ind % cum_card == (cum_card-1) ){
	    C_full_ind += cum_card+1;
	  }else{
	    C_full_ind++;
	  }
	}

	// size_t C_full_ind=0;
	// for (size_t C_elnum=0; C_elnum< tot_card/2; C_elnum++){
	//   C->data[C_elnum] = C_full->data[C_full_ind] + C_full->data[C_full_ind+cum_card];
	//   C_full_ind += cum_card;
	//}
      }

      cum_card *= current_card;
    }


    // size_t tmpS= sdata[tid];
    // cuPrintf("C %d\n",tmpS);

    //tmpC= C->data[index_number_C];
    //cuPrintf("C %d\n",tmpC);
    //cuPrintf("uclu %d %d %d index_number %d\n", uclu[0], uclu[1], uclu[2], (int)index_number);
    //cuPrintf("A %d B %d C %d\n", index_number_A, index_number_B, index_number_C);
  }
}


__global__ void
assignCudatensorConfig( ct_config* ctc, size_t* cards ){
  ctc->cardinalities = cards;
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

  //std::cout << "Contract dimension " << (int) (ctc->contract_dim) << std::endl;

  std::cout << "Cardinalities for each dimension of this configuration " << std::endl;
  size_t i=0;
  for ( i=0; i< ctc->ndims; i++){
    std::cout << ctc->cardinalities[i] << " ";
  }
  std::cout << "\nTotal cardinality: " << ctc->total_cardinality << std::endl;
  std::cout << "\nElement number: " << ctc->element_number << std::endl;
  std::cout << std::endl << std::endl << std::endl;
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
    for (size_t i=0; i< ct->config->element_number; i++){
      std::cout << ct->data[i] << " ";
    }
  }
  std::cout << std::endl << std::endl << std::endl;
}

// returns a dev_ct_ptrs struct with information about the cudatensor generated on the device
dev_ct_ptrs prepareDeviceTensor(ct_config* h_ctc, ct_config* d_ctc, ct* h_ct,
                                const mxArray* data, const mxArray* tensor_card = NULL){

 // generate h_ct

  h_ct->config = h_ctc;
  h_ct->cardinalities = (size_t*) malloc(sizeof(size_t)*h_ctc->ndims);

  // assign cardinalities for the tensor objects
  const mwSize* dims_c = mxGetDimensions(data);
  for (size_t i=0; i<h_ctc->ndims; i++){
    if (tensor_card==NULL){
      // we are doing hadamard multiplication, all tensors have same cardinalities
      // or we are doing output tensor object, which as maximum cardinalities on all dimensions
      h_ct->cardinalities[i] = dims_c[i];
      std::cout << "H dim "<< i << " cardinality assignment: "
                << h_ct->cardinalities[i]
                << " <- " << dims_c[i]
                << std::endl;
    }else{
      // we are doing tensor contraction, tensors may have different cardinalities
      h_ct->cardinalities[i] = ((float *)mxGetData(tensor_card))[i];
      std::cout << "TC dim "<< i << " cardinality assignment: "
                << h_ct->cardinalities[i]
                << " <- " << ((float *)mxGetData(tensor_card))[i] << std::endl;
    }
  }


  // assign h_ct host data
  size_t elnum = (size_t) mxGetNumberOfElements(data);
  std::cout << " prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(float) * elnum;
  h_ct->data = (float*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (float*)mxGetData(data), h_ct->mem_size);

  print_ct("prepareDeviceTensor h_ct",h_ct,false,true);


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

ct_config* ctcToDevice(ct_config* h_ctc){
  // transfer to device
  size_t* tmp_card;
  cutilSafeCall(cudaMalloc((void**)&tmp_card, sizeof(size_t)*h_ctc->ndims));
  cutilSafeCall(cudaMemcpy(tmp_card, h_ctc->cardinalities, sizeof(size_t)*h_ctc->ndims ,cudaMemcpyHostToDevice));

  ct_config* d_ctc;
  cutilSafeCall(cudaMalloc((void**) &d_ctc, sizeof(ct_config) ));
  cutilSafeCall(cudaMemcpy( d_ctc , h_ctc, sizeof(ct_config), cudaMemcpyHostToDevice) );

  assignCudatensorConfig<<<1,1>>>(d_ctc, tmp_card);

  return d_ctc;
}

ct_config* prepareDeviceTensorConfig(ct_config* h_ctc, const mxArray* sampleObject){
  h_ctc->ndims = mxGetNumberOfDimensions(sampleObject);
  h_ctc->cardinalities = (size_t*) malloc(sizeof(size_t)*h_ctc->ndims);
  const mwSize *dims = mxGetDimensions(sampleObject);
  h_ctc->total_cardinality = 1;
  for (size_t i=0; i<h_ctc->ndims; i++){
    h_ctc->cardinalities[i] = dims[i];
    if(dims[i] != 0)
      h_ctc->total_cardinality *= dims[i];
  }
  return ctcToDevice(h_ctc);
}

ct_config* getDeviceTensorContractConfig(ct_config* h_ctc, const mxArray* tensor1, const mxArray* tensor1_card, const mxArray* tensor2, const mxArray* tensor2_card){
  h_ctc->ndims = mxGetNumberOfElements(tensor1_card); // assumes both objects of same size
  h_ctc->cardinalities = (size_t*) malloc(sizeof(size_t)*h_ctc->ndims);
  h_ctc->element_number = 0;
  h_ctc->total_cardinality = 1;

  float tmpcard1[h_ctc->ndims];
  float tmptotalcard1=1;

  float tmpcard2[h_ctc->ndims];
  float tmptotalcard2=1;

  for (size_t i=0; i<h_ctc->ndims; i++){
    // assumes same total cardinality for all objects
    if ( ((float*)mxGetData(tensor1_card))[i] != 0 )
      h_ctc->total_cardinality *= ((float*)mxGetData(tensor1_card))[i];
    else if (((float*)mxGetData(tensor2_card))[i] != 0)
      h_ctc->total_cardinality *= ((float*)mxGetData(tensor2_card))[i];

    tmpcard1[i] = ((float*)mxGetData(tensor1_card))[i];
    if ( ((float*)mxGetData(tensor1_card))[i] != 0 )
      tmptotalcard1 *= ((float*)mxGetData(tensor1_card))[i];

    tmpcard2[i] = ((float*)mxGetData(tensor2_card))[i];
    if ( ((float*)mxGetData(tensor2_card))[i] != 0 )
      tmptotalcard2 *= ((float*)mxGetData(tensor2_card))[i];
  }

  if (tmptotalcard1 != tmptotalcard2){
    std::cout << "input arguments have different number of elements, exiting" << std::endl;
  }
  std::cout << "element number <- " << tmptotalcard1 << std::endl;
  h_ctc->element_number = tmptotalcard1;

  for (size_t i=0; i<h_ctc->ndims; i++){
    h_ctc->cardinalities[i] = std::max( ((float*)mxGetData(tensor1_card))[i] ,
                                        ((float*)mxGetData(tensor2_card))[i] );
  }

  return ctcToDevice(h_ctc);
}

void print_device_ctc(char* txt, ct_config* d_ctc){
  ct_config tmp_ctc;
  cutilSafeCall(cudaMemcpy(&tmp_ctc, d_ctc, sizeof(ct_config), cudaMemcpyDeviceToHost));
  //print_ct_config(txt,&tmp_ctc); // must return pointer set from config copy operation and use that
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


enum tensor_operation{
  hadamard,
  contract
};


void operate(ct_config* h_ctc, ct_config* d_ctc, const mxArray *prhs[], mxArray *plhs[], tensor_operation operation){

  // input tensor A
  ct h_it_A; dev_ct_ptrs d_A;
  // input tensor B
  ct h_it_B; dev_ct_ptrs d_B;

  if (operation==hadamard){
    // we are doing hadamard multiplication, all tensors have same cardinalities
    std::cout << "d_A prepareDeviceTensor " << std::endl;
    d_A=prepareDeviceTensor(h_ctc, d_ctc, &h_it_A, prhs[0]);
    std::cout << "d_B prepareDeviceTensor " << std::endl;
    d_B=prepareDeviceTensor(h_ctc, d_ctc, &h_it_B, prhs[1]);
  }else if (operation==contract){
    // we are doing tensor contraction, tensors may have different cardinalities
    std::cout << "d_A prepareDeviceTensor " << std::endl;
    d_A=prepareDeviceTensor(h_ctc, d_ctc, &h_it_A, prhs[0], prhs[1]);
    std::cout << "d_B prepareDeviceTensor " << std::endl;
    d_B=prepareDeviceTensor(h_ctc, d_ctc, &h_it_B, prhs[2], prhs[3]);
  }

  // output tensor C
  ct h_ot_C;
  ct h_ot_C_full;
  //ct_config h_ctc_full;
  //ct_config d_ctc_full;

  dev_ct_ptrs d_C;
  dev_ct_ptrs d_C_full;
  mxArray* full_data = mxCreateNumericArray(h_ctc->ndims,h_ctc->cardinalities,mxSINGLE_CLASS,mxREAL);
  mxArray* full_cardinalities = mxCreateNumericArray(h_ctc->ndims,h_ctc->cardinalities,mxSINGLE_CLASS,mxREAL);
  float* f_c_ptr = (float*)mxGetData(full_cardinalities);
  for(size_t i=0; i<h_ctc->ndims; i++ ){
    f_c_ptr[i]=h_ctc->cardinalities[i];
  }


  // prepare MATLAB storage
  float* m_C;
  //size_t m_C_mem_size=1;
  // calculate total cardinalities for all objects
  if(operation == hadamard){
    mwSize argMatDims[h_ctc->ndims];
    for (size_t i=0; i<h_ctc->ndims; i++){
      argMatDims[i] = h_ctc->cardinalities[i];
    }
    plhs[0] = mxCreateNumericArray(h_ctc->ndims,argMatDims,mxSINGLE_CLASS,mxREAL);
    m_C = (float*) mxGetPr(plhs[0]);
    std::cout << "d_C prepareDeviceTensor " << std::endl;
    d_C=prepareDeviceTensor(h_ctc, d_ctc, &h_ot_C, plhs[0]);
  }

  else if (operation == contract){
    size_t non_zero_dim_number=0;
    for (size_t i=0; i<h_ctc->ndims; i++){
      //std::cout << " non_zero_dim_number loop " << i ;
      float tmpdimcard = ((float*)mxGetData(prhs[4]))[i];
      if(tmpdimcard != 0) {
        non_zero_dim_number++;
        //std::cout  << " tmpdimcard " << tmpdimcard << std::endl;
        //m_C_mem_size *= tmpdimcard;
      }
    }

    mwSize argMatDims[non_zero_dim_number];
    size_t argMatDims_ind=0;
    //std::cout << "C tensor init argMatDims with size " << non_zero_dim_number << std::endl;
    //<< " m_C_mem_size " << m_C_mem_size << std::endl;

    for (size_t i=0; i<h_ctc->ndims; i++){
      float val=((float*)mxGetData(prhs[4]))[i];
      //std::cout << "C tensor argMatDims[" << i << "] = " << val << " ";
      if ( val != 0){ // skip dimensions with 0 cardinality
        //std::cout << " assign " << std::endl;
        argMatDims[argMatDims_ind] = val;
        argMatDims_ind++;
      }else{
        //std::cout << " not assign " << std::endl;
      }
    }

    plhs[0] = mxCreateNumericArray(non_zero_dim_number,argMatDims,mxSINGLE_CLASS,mxREAL);
    //std::cout << "SELAM  " <<  (size_t) mxGetNumberOfElements(plhs[0]) << std::endl;
    m_C = (float*) mxGetPr(plhs[0]);

    std::cout << "d_C prepareDeviceTensor " << std::endl;
    d_C=prepareDeviceTensor(h_ctc, d_ctc, &h_ot_C, plhs[0], prhs[4]);

    std::cout << "SELAAM bu " << h_ctc->element_number << std::endl;
    std::cout << "d_C_full prepareDeviceTensor " << std::endl;
    d_C_full=prepareDeviceTensor(h_ctc, d_ctc, &h_ot_C_full, full_data, full_cardinalities);
  }


  bool printdata=true;
  print_ct("Host A",&h_it_A,false,printdata);
  print_ct("Host B",&h_it_B,false,printdata);
  print_ct("Host C",&h_ot_C,false,printdata);
  print_ct("Host C_full",&h_ot_C_full,false,printdata);


  print_device_ct("Device A",&d_A, &h_it_A);
  print_device_ct("Device B",&d_B, &h_it_B);
  print_device_ct("Device C",&d_C, &h_ot_C);

  print_device_ct("Device C_full",&d_C_full, &h_ot_C_full);


  cudaPrintfInit();

  // allocate device memory for result
  // kernel warmup

  // if (operation == hadamard){
  //   tensorHadamard<<< blocks, threads >>>(d_C.ct, d_A.ct, d_B.ct);
  // }else if (operation == contract){
  //   tensorContract<<< blocks, threads >>>(d_C.ct, d_A.ct, d_B.ct);
  // }



  cudaThreadSynchronize();

  // create and start timer
  std::cout << "Run Kernels...\n\n" << std::endl;

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute the kernel
  int nIter = 30;
  //for (int j = 0; j < nIter; j++) {
  if (operation == hadamard){
    tensorHadamard<<< blocks, threads >>>(d_C.ct, d_A.ct, d_B.ct);
  }else if (operation == contract){
    tensorContract<<< blocks, threads >>>(d_C_full.ct, d_C.ct, d_A.ct, d_B.ct);
  }
  //}

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
  float h_C_full_data[h_ctc->total_cardinality];

  if(operation==hadamard){
    cutilSafeCall(cudaMemcpy(m_C, d_C.data, h_ot_C.mem_size, cudaMemcpyDeviceToHost) ); // assumes same size
  }
  else if(operation==contract){
    cutilSafeCall(cudaMemcpy(m_C, d_C.data, h_ot_C.mem_size, cudaMemcpyDeviceToHost) ); // assumes same size
    cutilSafeCall(cudaMemcpy(h_C_full_data, d_C_full.data, h_ot_C_full.mem_size, cudaMemcpyDeviceToHost) );
  }

  // print C_full 
  for (size_t i=0; i<h_ctc->total_cardinality; i++){
    std::cout << "C_full[" << i << "] = " << h_C_full_data[i] << std::endl;
  }


  // clean up memory
  //free(h_A);
  //free(h_B);
  //free(h_C);
  //free(reference);

  // wrong
  //cutilSafeCall(cudaFree());
  //cutilSafeCall(cudaFree(d_it_B));
  //cutilSafeCall(cudaFree(d_it_A)); //->C

  print_device_ct("Result\nDevice C",&d_C_full, &h_ot_C);
  print_device_ct("Result\nDevice C",&d_C, &h_ot_C);

  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();

  cudaThreadExit();


  // required to avoid memory leak?
  delete h_ctc->cardinalities;
  delete h_it_A.cardinalities;
  delete h_it_B.cardinalities;
}



void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  std::cout << "mex: found " << nrhs << " number of arguments " << std::endl;
  if (nrhs == 2){
    // hadamard multiplication
    std::cout << "mex: applying hadamard multiplication " << std::endl;

    // ASSUMES target tensors are of the same dimension
    ct_config h_ctc;
    ct_config* d_ctc = prepareDeviceTensorConfig(&h_ctc,prhs[0]);

    print_ct_config("Host ctc",&h_ctc);

    operate(&h_ctc, d_ctc, prhs, plhs, hadamard);

  }else if(nrhs==5){
    // tensor contraction operation
    std::cout << "mex: applying tensor contraction " << std::endl;

    ct_config h_ctc;
    ct_config* d_ctc = getDeviceTensorContractConfig(&h_ctc,prhs[0],prhs[1],prhs[2],prhs[3]);

    print_ct_config("Host ctc", &h_ctc);

    print_device_ctc("Device tmp ctc",d_ctc);

    operate(&h_ctc, d_ctc, prhs, plhs, contract);


  }else{
    std::cout << "mex: wrong number of arguments " << std::endl;
  }


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
