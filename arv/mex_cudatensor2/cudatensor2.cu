/*
 * author: ck
 * 06.06.2011
 * advisor: atc
 */

#include "mex.h"
#include "cublas.h"
#include "cutil_inline.h"

#include <iostream>
#include <algorithm>

#include "cuPrintf.cu"

#include "tensor.h"


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

  std::cout << "Strides for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->config->ndims; i++){
    std::cout << ct->strides[i] << " ";
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





void prepareHostConfig(ct_config* h_ctc, const mxArray* tensor1, const mxArray* tensor1_card, const mxArray* tensor2, const mxArray* tensor2_card){
  h_ctc->ndims = mxGetNumberOfElements(tensor1_card); // assumes both objects of same size
  h_ctc->cardinalities = (size_t*) malloc(sizeof(size_t)*h_ctc->ndims);
  h_ctc->element_number = 0;
  h_ctc->total_cardinality = 1;

  double tmptotalcard1=1;
  double tmptotalcard2=1;

  for (size_t i=0; i<h_ctc->ndims; i++){
    // assumes same total cardinality for all objects
    if ( ((double*)mxGetData(tensor1_card))[i] != 0 )
      h_ctc->total_cardinality *= ((double*)mxGetData(tensor1_card))[i];
    else if (((double*)mxGetData(tensor2_card))[i] != 0)
      h_ctc->total_cardinality *= ((double*)mxGetData(tensor2_card))[i];

    if ( ((double*)mxGetData(tensor1_card))[i] != 0 )
      tmptotalcard1 *= ((double*)mxGetData(tensor1_card))[i];

    if ( ((double*)mxGetData(tensor2_card))[i] != 0 )
      tmptotalcard2 *= ((double*)mxGetData(tensor2_card))[i];
  }

  if (tmptotalcard1 != tmptotalcard2){
    std::cout << "input arguments have different number of elements, exiting" << std::endl;
  }
  std::cout << "element number <- " << tmptotalcard1 << std::endl;
  h_ctc->element_number = tmptotalcard1;

  for (size_t i=0; i<h_ctc->ndims; i++){
    h_ctc->cardinalities[i] = std::max( ((double*)mxGetData(tensor1_card))[i] ,
                                        ((double*)mxGetData(tensor2_card))[i] );
  }
  //return ctcToDevice(h_ctc);
}



void prepareHostTensor(ct_config* h_ctc, ct* h_ct, const mxArray* m_data, const mxArray* tensor_card){
  h_ct->config = h_ctc;
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  //h_ct->cur_ind       = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  //h_ct->cur_anti_ind       = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  //h_ct->increment_error = 0;

  // assign cardinalities for the tensor objects and init cur_ind values
  const mwSize* dims_c = mxGetDimensions(m_data);
  size_t cum_sum=1;
  for (size_t i=0; i<h_ctc->ndims; i++){
    h_ct->cardinalities[i] = ((double *)mxGetData(tensor_card))[i];
    std::cout << "TC dim "<< i << " cardinality assignment: "
              << h_ct->cardinalities[i]
              << " <- " << ((double *)mxGetData(tensor_card))[i] << std::endl;

    //h_ct->cur_ind[i] = 0;
    //h_ct->cur_anti_ind[i] = 0;

    if ( h_ct->cardinalities[i] == 0){
      h_ct->strides[i]=0;
    }else{
      h_ct->strides[i]=cum_sum;
      cum_sum *= h_ct->cardinalities[i];
    }
  }

  // assign h_ct host data
  size_t elnum = (size_t) mxGetNumberOfElements(m_data);
  std::cout << " prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->data = (double*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (double*)mxGetData(m_data), h_ct->mem_size);

  print_ct("prepareDeviceTensor h_ct",h_ct,false,true);
}








// Multiply incoming vector pair by pair and write the result in second input vector
__global__ void pairmul( double* pairmul, size_t pairmul_elnum, double* pairmul_result ){
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if ( thread_id < pairmul_elnum/2  ) { // odd number of elements?
    pairmul_result[thread_id] = pairmul[(thread_id+1)*2-2] * pairmul[(thread_id+1)*2-1];
    cuPrintf("pairmul_result[%d] = pairmul[%d] * pairmul[%d]\n", thread_id, (thread_id+1)*2-2, (thread_id+1)*2-1);
  }
}






double get_element(ct* h_ct, size_t* global_index, char* str=""){
  std::cout << "get_element: " << str << " cur_ind ";
  size_t cur_ind=0;
  for (size_t dim=0; dim<h_ct->config->ndims; dim++){
    std::cout << global_index[dim] << " ";
    cur_ind += h_ct->strides[dim] * global_index[dim];
  }
  std::cout << " index " << cur_ind << " val " << h_ct->data[cur_ind] 
	    << std::endl;
  return h_ct->data[cur_ind];
}




void increment_cur_index(ct_config* h_ctc, size_t* global_index){
  for (size_t dim=0; dim<h_ctc->ndims; dim++){
    // if we have NOT reached limit of this dimension
    if( global_index[dim] != (h_ctc->cardinalities[dim]-1) ){
      // increment this dimension
      global_index[dim]++;
      break;
    }else{
      // we have reached limit of this dimension

      // if next dimension is at limit as well, skip this dimension, operation will take place in next dimension
      if( dim != (h_ctc->ndims-1) && global_index[dim+1] == (h_ctc->cardinalities[dim+1]-1)){
	//std::cout << "skip" << std::endl;
	continue;
      }else{

	// if this is the last dimension (and it is full) no increment is possible increment error
	//if (dim == h_ctc->ndims-1){
	//  h_ct->increment_error = 1;
	//  break;
	//}

	// make this and all previous dimensions zero
	for (int dim_prev=dim; dim_prev>=0 ; dim_prev--){
	  global_index[dim_prev] = 0;
	}
	// increment next dimension
	global_index[dim+1]++;
	break;
      }
    }
  }
}




void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  std::cout << "mex: found " << nrhs << " number of arguments " << std::endl;
  if (nrhs!=5){
    std::cout << "mex: cudatensor2 requires 5 arguments. A, dimensions of A, B, dimensions of B, dimensions of C " << std::endl;
    return;
  }



  ct_config h_ctc;
  prepareHostConfig(&h_ctc,prhs[0],prhs[1],prhs[2],prhs[3]);



  // prepare global index //////////////////////////////////////////////////////////////////////
  // used to address data fields of tensor objects
  size_t global_index[h_ctc.ndims];
  for (size_t i=0; i<h_ctc.ndims; i++){
    global_index[i]=0;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////



  // prepare output tensor ////////////////////////////////////////////////////////////////////

  size_t non_zero_dim_number=0;
  for (size_t i=0; i<h_ctc.ndims; i++){
    //std::cout << " non_zero_dim_number loop " << i ;
    double tmpdimcard = ((double*)mxGetData(prhs[4]))[i];
    if(tmpdimcard != 0) {
      non_zero_dim_number++;
      //std::cout  << " tmpdimcard " << tmpdimcard << std::endl;
      //m_C_mem_size *= tmpdimcard;
    }
  }

  mwSize argMatDims[non_zero_dim_number];
  size_t argMatDims_ind=0;
  size_t C_elnum=1;
  //std::cout << "C tensor init argMatDims with size " << non_zero_dim_number << std::endl;
  //<< " m_C_mem_size " << m_C_mem_size << std::endl;

  for (size_t i=0; i<h_ctc.ndims; i++){
    double val=((double*)mxGetData(prhs[4]))[i];
    //std::cout << "C tensor argMatDims[" << i << "] = " << val << " ";
    if ( val != 0){ // skip dimensions with 0 cardinality
      //std::cout << " assign " << std::endl;
      argMatDims[argMatDims_ind] = val;
      argMatDims_ind++;
      C_elnum *= val;
    }else{
      //std::cout << " not assign " << std::endl;
    }
  }

  plhs[0] = mxCreateNumericArray(non_zero_dim_number,argMatDims,mxDOUBLE_CLASS,mxREAL);
  //m_C = (double*) mxGetPr(plhs[0]);

  ///////////////////////////////////////////////////////////////////////////////////////////


  // prepare pairmul vector ////////////////////////////////////////////////////////////////


  ct h_A; ct h_B; //ct h_C;
  prepareHostTensor(&h_ctc, &h_A, prhs[0], prhs[1]);
  prepareHostTensor(&h_ctc, &h_B, prhs[2], prhs[3]);
  //prepareHostTensor(&h_ctc, &h_C, plhs[0], prhs[4]);

  // prepare memory for pairmul vector
  double* h_pairmul = (double*) malloc( sizeof(double) * C_elnum * 2 );

  // populate pairmul vector
  //double* m_C_card = ((double*)mxGetData(prhs[4]));

  size_t cur_pairmul_ind=0;

  // make operation for all element in all dimensions
  for ( size_t tot_card=0; tot_card < h_ctc.total_cardinality; tot_card++){
    std::cout << "pairmul gen: tot_card " << tot_card << std::endl;

    h_pairmul[cur_pairmul_ind] = get_element(&h_A, global_index, "h_A");
    cur_pairmul_ind++;

    h_pairmul[cur_pairmul_ind] = get_element(&h_B, global_index, "h_B");
    cur_pairmul_ind++;

    increment_cur_index(&h_ctc, global_index);
  }


  std::cout << "h_pairmul" << std::endl;
  for (size_t i=0; i<C_elnum * 2; i++){
    std::cout << " i "<< i << " = " << h_pairmul[i] << std::endl;
  }


  ///////////////////////////////////////////////////////////////////////////////////////////


  // run kernels //////////////////////////////////////////////////////////////////////////////


  double* d_pairmul, *d_pairmul_result;
  cutilSafeCall(cudaMalloc((void**)&d_pairmul, sizeof(double)*C_elnum*2));
  cutilSafeCall(cudaMalloc((void**)&d_pairmul_result, sizeof(double)*C_elnum));

  cutilSafeCall(cudaMemcpy(d_pairmul, h_pairmul, sizeof(double)*C_elnum*2, cudaMemcpyHostToDevice));


  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  cudaPrintfInit();

  std::cout << " Running kernels " << std::endl << std::endl;
  pairmul<<<100,100>>>(d_pairmul, C_elnum*2, d_pairmul_result);

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaThreadSynchronize();
  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  cutilCheckError(cutDeleteTimer(timer));

  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
  

  double* m_C = (double*) mxGetPr(plhs[0]);
  cutilSafeCall(cudaMemcpy(m_C, d_pairmul_result, sizeof(double)*C_elnum, cudaMemcpyDeviceToHost));
  std::cout << "plhs elnum " << mxGetNumberOfElements(plhs[0]) << std::endl;
  std::cout << "C_elnum " << C_elnum << std::endl;

  cudaThreadExit();
  ///////////////////////////////////////////////////////////////////////////////////////////
}
