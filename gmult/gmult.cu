/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#include <string.h>
#include <sstream>

#include "utils.cuh"

void tensorop(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool is_parallel){
  if( nlhs != 1 ){
    std::cout << "gmult: tensor operation requires exactly one output argument" << std::endl;
    // print help;
    return;
  }

  if ( nrhs != 6 ){
    std::cout << "gmult: tensor operation requires 6 arguments. "
              << "A, dimensions of A, B, dimensions of B, dimensions of C,"
              << " use_multiplication(1 uses multiplication, 0 uses division)"
              << std::endl;
    return;
  }


  const mxArray* m_A_data = prhs[0];
  const mxArray* m_A_card = prhs[1];

  const mxArray* m_B_data = prhs[2];
  const mxArray* m_B_card = prhs[3];

  const mxArray* m_C_card = prhs[4];

  size_t use_multiplication = ((double *)mxGetData(prhs[5]))[0];

  // assume same size cardinalities for all objects
  size_t ndims = mxGetNumberOfElements(m_A_card);

  // if all cardinalities of A,B,C are the same, we have hadamard operation
  bool isHadamard=true;

  // full_cardinalities define maximum possible cardinalities for all dimensions
  h_full_cardinalities = (size_t*) calloc(ndims, sizeof(size_t)); // defined in mct_tensorop_utils.cuh
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


  if (non_zero_dim_number==0){
    // contraction on all dimensions
    // will result in a single number
    mwSize argMatDim[1] = {1};
    plhs[0] = mxCreateNumericArray(1,argMatDim,mxDOUBLE_CLASS,mxREAL);
  }else{
    plhs[0] = mxCreateNumericArray(non_zero_dim_number,argMatDims,mxDOUBLE_CLASS,mxREAL);
  }


  double* m_C = (double*) mxGetPr(plhs[0]);

  ///////////////////////////////////////////////////////////////////////////////////////////

  // prepare host memory for tensors  ///////////////////////////////////////////////////////

  ct h_A, h_B, h_C, h_F;

  size_t A_card[ndims];
  size_t B_card[ndims];
  for (size_t i=0; i<ndims; i++){
    A_card[i] = ((double *)mxGetData(m_A_card))[i];
    B_card[i] = ((double *)mxGetData(m_B_card))[i];
  }

  prepareHostTensorFromCpp(&h_A, mxGetPr(m_A_data), A_card, ndims, (const char*) "Host A");
  prepareHostTensorFromCpp(&h_B, mxGetPr(m_B_data), B_card, ndims, (const char*)"Host B");
  // NULL initiates data with zero
  prepareHostTensorFromCpp(&h_F, NULL, h_full_cardinalities, ndims, "Host F");

  // read C cardinalities from matlab side
  size_t* tmp_arr = (size_t*) malloc(sizeof(size_t)*ndims);
  for ( size_t i=0; i<ndims; i++) tmp_arr[i] = (size_t) (((double*) mxGetData(m_C_card))[i]);

  prepareHostTensorFromCpp(&h_C, NULL, tmp_arr, ndims, "Host C");
  free(tmp_arr);


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


  if ( is_parallel) 
    tensorop_par(isHadamard, h_A, h_B, h_C, m_C, h_F, ndims, h_zero_cardinality_dim_tuples_C_element_number, h_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuple_size_C, use_multiplication);
  else
    tensorop_seq(isHadamard, h_A, h_B, h_C, m_C, h_F, ndims, h_zero_cardinality_dim_tuples_C_element_number, h_zero_cardinality_dim_tuples_C);

  if (is_parallel)
    resetDevice();

  free_ct(&h_A);
  free_ct(&h_B);
  free_ct(&h_C);
  //free_ct(&h_F);
}
