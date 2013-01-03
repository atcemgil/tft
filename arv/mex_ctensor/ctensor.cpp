// matlab interface to atc's tensor.h
// author ck
// 08.2011

#include "tensor.h"
#include "mex.h"
#include <iostream>
#include <vector>

Tensor<double>* genTensor(const mxArray* data_ptr, const mxArray* card_ptr){
  size_t ndims = mxGetNumberOfElements(card_ptr);
  size_t sizes[ndims];

  for ( size_t i = 0 ; i < ndims; i++)
    sizes[i] = ((double *)mxGetData(card_ptr))[i];

  Tensor<double>* t = new Tensor<double>(mxGetNumberOfElements(card_ptr), (size_t*)sizes);

  if (data_ptr != NULL){
    valarray<double>* va = t->Data_ptr();

    for ( size_t i = 0 ; i < mxGetNumberOfElements(data_ptr); i++)
      (*va)[i] = ((double *)mxGetData(data_ptr))[i];

  }
  return t;
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  std::cout << "mex: found " << nrhs << " number of arguments " << std::endl;

  if (nrhs!=5){
    std::cout << "mex: cudatensor3 requires 5 arguments. A, dimensions of A, B, dimensions of B, dimensions of C " << std::endl;
    return;
  }


  const mxArray* m_A_data = prhs[0];
  const mxArray* m_A_card = prhs[1];

  const mxArray* m_B_data = prhs[2];
  const mxArray* m_B_card = prhs[3];

  const mxArray* m_C_card = prhs[4];


  // assume same size cardinalities for all objects
  size_t ndims = mxGetNumberOfElements(m_A_card);

  // full_cardinalities define maximum possible cardinalities for all dimensions
  size_t* h_full_cardinalities = (size_t*) calloc(ndims, sizeof(size_t));
  for( size_t i=0; i<ndims; i++){
    h_full_cardinalities[i] = std::max(
                                       std::max( ((double *)mxGetData(m_A_card))[i] ,
                                                 ((double *)mxGetData(m_B_card))[i] ) ,
                                       ((double *)mxGetData(m_C_card))[i] );
  }



  // prepare output tensor in matlab  //////////////////////////////////////////////////////

    size_t non_zero_dim_number=0;
  for (size_t i=0; i<ndims; i++){
    //std::cout << " non_zero_dim_number loop " << i ;
    double tmpdimcard = ((double*)mxGetData(m_C_card))[i];
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

  for (size_t i=0; i<ndims; i++){
    double val=((double*)mxGetData(m_C_card))[i];
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
  double* m_C = (double*) mxGetPr(plhs[0]);

  
  Tensor<double>* A = genTensor(m_A_data, m_A_card);
  Tensor<double>* B = genTensor(m_B_data, m_B_card);
  Tensor<double>* C = genTensor(NULL, m_C_card);

  cout << "A";
  A->Print();
  cout << "B";
  B->Print();
  cout << "C";
  C->Print();
  Tmult_contract(*C, *A, *B, true);
  cout << "C";
  C->Print();

}
