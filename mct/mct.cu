/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */

//#include "mex.h"
#include "cublas.h"

#include <iostream>
#include <algorithm>
//#include <vector>

#include <string.h>

#include "mct_tensorop_utils.cuh"

enum op_type {
  tensor_op,
  nmf_op,
  num_of_op_types
};


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if ( COUT ) std::cout << "mct: found " << nrhs << " number of arguments " << std::endl;

  if( nrhs == 0 ){
    std::cout << "mct: not enough arguments" << std::endl;
    // print help;
    return;
  }

  if (!mxIsChar(prhs[0]) || (mxGetM(prhs[0]) != 1 ) ){
    mexErrMsgTxt("mct: first argument must be a string.");
    // print help;
    return;
  }
  mwSize buflen = mxGetN(prhs[0])*sizeof(mxChar)+1;
  char* op_name = (char*) mxMalloc(buflen);
  op_type opt;
  int status = mxGetString(prhs[0], op_name, buflen);


  if (strcmp(op_name, "nmf") == 0){
    std::cout << "selecting NMF operation" << std::endl;
    opt=nmf_op;
  }else if (strcmp(op_name, "tensor") == 0){
    std::cout << "selecting tensor operation" << std::endl;
    opt=tensor_op;
  }else{
    std::cout << "unknown operation: " << op_name << std::endl;
    // print help;
    return;
  }


  if ( opt == nmf_op ){

  }else if ( opt == tensor_op ){

    if ( nrhs!= (7+1) ){
      std::cout << "mct: tensor operation requires 7 arguments. "
                << "A, dimensions of A, B, dimensions of B, dimensions of C, "
                << " use_c_code(1 uses c,0 uses gpu), use_multiplication(1 uses multiplication, 0 uses division  "
                << std::endl;
      return;
    }

    const mxArray* m_A_data = prhs[1];
    const mxArray* m_A_card = prhs[2];

    const mxArray* m_B_data = prhs[3];
    const mxArray* m_B_card = prhs[4];

    const mxArray* m_C_card = prhs[5];

    size_t use_c_code = ((double*)mxGetData(prhs[6]))[0];

    size_t use_multiplication = ((double *)mxGetData(prhs[7]))[0];

    // assume same size cardinalities for all objects
    size_t ndims = mxGetNumberOfElements(m_A_card);

    // if all cardinalities of A,B,C are the same, we have hadamard operation
    bool isHadamard=true;

    // full_cardinalities define maximum possible cardinalities for all dimensions
    size_t* h_full_cardinalities = (size_t*) calloc(ndims, sizeof(size_t));
    for( size_t i=0; i<ndims; i++){
      double m_A_card_i = ((double *)mxGetData(m_A_card))[i];
      double m_B_card_i = ((double *)mxGetData(m_B_card))[i];
      double m_C_card_i = ((double *)mxGetData(m_C_card))[i];

      h_full_cardinalities[i] = std::max(std::max(m_A_card_i,m_B_card_i), m_C_card_i);

      if ( m_A_card_i != m_B_card_i || m_B_card_i != m_C_card_i ) isHadamard=false;
    }

    if (COUT) if (isHadamard) std::cout << "HADAMARD OPERATION" << std::endl;


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


    /*
      for (size_t i=0; i<ndims; i++){
      double val=((double*)mxGetData(m_C_card))[i];
      //std::cout << "C tensor argMatDims[" << i << "] = " << val << " ";
      //std::cout << " assign " << std::endl;
      if ( val == 0){
      argMatDims[argMatDims_ind] = 1;
      }else{
      argMatDims[argMatDims_ind] = val;
      C_elnum *= val;
      }
      argMatDims_ind++;
      }
    */

    plhs[0] = mxCreateNumericArray(non_zero_dim_number,argMatDims,mxDOUBLE_CLASS,mxREAL);

    // mwSize alldims[ndims];
    // for (size_t i=0; i<ndims; i++){
    //   alldims[i]=((double*)mxGetData(m_C_card))[i];
    // }

    // plhs[0] = mxCreateNumericArray(ndims,alldims,mxDOUBLE_CLASS,mxREAL);

    double* m_C = (double*) mxGetPr(plhs[0]);

    ///////////////////////////////////////////////////////////////////////////////////////////

    // prepare host memory for tensors  ///////////////////////////////////////////////////////

    ct h_A, h_B, h_C, h_F;
    prepareHostTensor(&h_A, m_A_data, m_A_card, "Host A");
    prepareHostTensor(&h_B, m_B_data, m_B_card, "Host B");
    // NULL initiates data with zero
    prepareHostTensorFromCpp(&h_F, NULL, h_full_cardinalities, ndims, "Host F");

    // read C cardinalities from matlab side
    size_t* tmp_arr = (size_t*) malloc(sizeof(size_t)*ndims);
    for ( size_t i=0; i<ndims; i++) tmp_arr[i] = (size_t) (((double*) mxGetData(m_C_card))[i]);

    prepareHostTensorFromCpp(&h_C, NULL, tmp_arr, ndims, "Host C");



    // prepare range permutation vector //////////////////////////////////////////////////////
    size_t h_zero_cardinality_dim_tuple_size_C = 0;
    size_t h_zero_cardinality_dim_tuples_C_element_number = 0;
    size_t* h_zero_cardinality_dim_tuples_C = NULL;

    if ( isHadamard == false){
      std::vector<size_t> zero_cardinality_dims;
      //std::vector<size_t> non_zero_cardinality_dims;
      for ( size_t dim=0; dim<ndims; dim++ ){
        if ( h_C.cardinalities[dim] == 0 && h_F.cardinalities[dim] != 0 ){
          zero_cardinality_dims.push_back(h_F.cardinalities[dim]);
        }
        // else{
        //   non_zero_cardinality_dims.push_back(h_F.cardinalities[dim]);
        // }
      }

      // std::cout << "non_zero_cardinality_dims" << std::endl;
      // for ( size_t j=0; j<non_zero_cardinality_dims.size(); j++){
      //   std::cout << non_zero_cardinality_dims.at(j) << std::endl;
      // }

      if ( COUT ) {
        std::cout << "zero_cardinality_dims" << std::endl;
        for ( size_t j=0; j<zero_cardinality_dims.size(); j++){
          std::cout << zero_cardinality_dims.at(j) << std::endl;
        }
      }

      h_zero_cardinality_dim_tuple_size_C = zero_cardinality_dims.size();


      //h_zero_cardinality_dim_tuples_C_element_number; // set by gen_range_permutation

      h_zero_cardinality_dim_tuples_C =
        gen_range_permutation(zero_cardinality_dims,
                              &(h_zero_cardinality_dim_tuples_C_element_number));
    }



    if ( use_c_code == 0 ) {
      //mct_tensorop_gpu();
    }else{  // operate on CPU
      //mct_tensorop_c();
    }
  }
}
