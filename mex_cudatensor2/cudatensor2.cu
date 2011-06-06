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
  h_ct->cur_ind       = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );

  // assign cardinalities for the tensor objects and init cur_ind values
  const mwSize* dims_c = mxGetDimensions(data);
  for (size_t i=0; i<h_ctc->ndims; i++){
    h_ct->cardinalities[i] = ((double *)mxGetData(tensor_card))[i];
    std::cout << "TC dim "<< i << " cardinality assignment: "
              << h_ct->cardinalities[i]
              << " <- " << ((double *)mxGetData(tensor_card))[i] << std::endl;

    h_ct->cur_ind[i] = 0;
  }

  // assign h_ct host data
  size_t elnum = (size_t) mxGetNumberOfElements(data);
  std::cout << " prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->data = (double*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (double*)mxGetData(data), h_ct->mem_size);

  print_ct("prepareDeviceTensor h_ct",h_ct,false,true);
}








// Multiply incoming vector pair by pair and write the result in second input vector
__global__ void pairmul( double* pairmul, size_t pairmul_elnum, double* pairmul_result ){
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if ( thread_id < pairmul_elnum/2  ) { // odd number of elements?
    pairmul_result[thread_id] = pairmul[thread_id*2] * pairmul[thread_id*2-1];
  }
}




double get_element(ct* h_ct){
  size_t cur_ind=0;
  for (size_t dim=0; dim<h_ct->config->ndims; dim++){
    cur_ind += h_ct->cardinalities[dim] * h_ct->cur_ind[dim];
  }
  return h_ct->data[cur_ind];
}


void increment_cur_index(ct* h_ct){
  for (size_t dim=0; dim<h_ct->config->ndims; dim++){  
    // if we reached limit of this dimension
    if( h_ct->cur_ind[dim] == h_ct->cardinalities[dim]){
      
    }
  }
  return h_ct->cur_ind++;
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



  ct h_A; ct h_B; //ct h_C;
  prepareHostTensor(&h_ctc, &h_A, prhs[0], prhs[1]);
  prepareHostTensor(&h_ctc, &h_B, prhs[2], prhs[3]);
  //prepareHostTensor(&h_ctc, &h_C, plhs[0], prhs[4]);

  // prepare memory for pairmul vector
  double* h_pairmul = (double*) malloc( sizeof(double) * C_elnum * 2 );

  // populate pairmul vector
  //double* m_C_card = ((double*)mxGetData(prhs[4]));

  size_t cur_pairmul_ind=0;

  for ( size_t element; element < h_ctc.total_cardinality; element++){
    
    h_pairmul[cur_pairmul_ind] = get_element(&h_A);
    increment_cur_index(h_A);
    cur_pairmul_ind++;

    h_pairmul[cur_pairmul_ind] = get_element(&h_B);
    increment_cur_index(h_B);
    cur_pairmul_ind++;

  }


  double* d_;
  cutilSafeCall(cudaMalloc((void**)&d_v1, sizeof(double)*h_ctc.total_cardinality));
  cutilSafeCall(cudaMalloc((void**)&d_v2, sizeof(double)*(h_ctc.total_cardinality/2)));



}
