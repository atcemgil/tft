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

#include "mct_tensorop_gpu.cuh"
#include "mct_tensorop_cpp.cuh"

#include <time.h>

#define REGISTER_CT(obj) register_ct(#obj, &obj)


enum op_type {
  tensor_gpu,
  tensor_cpp,
  nmf_gpu,
  nmf_cpp,
  num_of_op_types
};


void tensorop(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], op_type opt){
  if( nlhs != 1 ){
    std::cout << "mct: tensor operation requires exactly one output argument" << std::endl;
    // print help;
    return;
  }

  if ( nrhs != (6+1) ){
    std::cout << "mct: tensor operation requires 6 arguments. "
              << "A, dimensions of A, B, dimensions of B, dimensions of C,"
              << " use_multiplication(1 uses multiplication, 0 uses division)"
              << std::endl;
    return;
  }

  const mxArray* m_A_data = prhs[1];
  const mxArray* m_A_card = prhs[2];

  const mxArray* m_B_data = prhs[3];
  const mxArray* m_B_card = prhs[4];

  const mxArray* m_C_card = prhs[5];

  size_t use_multiplication = ((double *)mxGetData(prhs[6]))[0];

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
  prepareHostTensor(&h_A, m_A_data, m_A_card, (const char*) "Host A");
  prepareHostTensor(&h_B, m_B_data, m_B_card, (const char*)"Host B");
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



  if ( opt == tensor_gpu ) {
    mct_tensorop_gpu(isHadamard, h_A, h_B, h_C, m_C, h_F, ndims, h_full_cardinalities, h_zero_cardinality_dim_tuples_C_element_number, h_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuple_size_C, use_multiplication);
  }else{  // operate on CPU
    mct_tensorop_cpp(isHadamard, h_A, h_B, h_C, m_C, h_F, ndims, h_full_cardinalities, h_zero_cardinality_dim_tuples_C_element_number, h_zero_cardinality_dim_tuples_C);
  }

}











void nmfop(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], op_type opt){
  if( nlhs != 2 ){
    std::cout << "mct: NMF operation requires exactly two output arguments" << std::endl;
    // print help;
    return;
  }

  if ( nrhs != (2+1) ){
    std::cout << "mct: NMF operation requires 2 arguments. "
              << "X, M"
              << std::endl;
    return;
  }

  const mxArray* m_X_data = prhs[1];
  const mxArray* m_M_data = prhs[2];

  if (mxGetNumberOfDimensions(m_X_data) != 2 ){
    std::cout << "mct: NMF requires two dimensional data (matrix) input" << std::endl;
    std::cout << "mct: found data dimenstions: " << mxGetNumberOfDimensions(m_X_data) << std::endl;
    // print help;
    return;
  }

  // prepare output tensor in matlab  //////////////////////////////////////////////////////

  // assume working with square matrices
  size_t card = (mxGetDimensions(m_X_data))[0];
  mwSize argMatDims[2] = { card, card };

  //argMatDims[0] = card;
  //argMatDims[1] = card;

  plhs[0] = mxCreateNumericArray(2,argMatDims,mxDOUBLE_CLASS,mxREAL);
  plhs[1] = mxCreateNumericArray(2,argMatDims,mxDOUBLE_CLASS,mxREAL);

  double* m_Z1 = (double*) mxGetPr(plhs[0]);
  double* m_Z2 = (double*) mxGetPr(plhs[1]);

  // prepare host memory for tensors  ///////////////////////////////////////////////////////

  // full_cardinalities define maximum possible cardinalities for all dimensions
  size_t full_cardinalities[3] = { card, card, card}; 
  if(COUT)
    for (int i=0; i<3; i++)
      std::cout << "full_cardinalities " << i << " " << full_cardinalities[i] << std::endl;


  // initialize random seed for random initialization of objects
  //srand((unsigned)time(NULL));
  srand(123);
  ct Z1, Z2, Xhat, X, D1_z1, D1_z2, D2_z1, D2_z2, A, M, F;

  mwSize m_X_card_size = 3;

  mxArray* m_X_card = mxCreateNumericArray(1,&m_X_card_size,mxDOUBLE_CLASS,mxREAL);

  size_t X_card[3] = {card, 0, card};
  for (size_t i=0; i<3; i++)
    mxGetPr(m_X_card)[i] = X_card[i];


  size_t ndims = 3;

  prepareHostTensor(&X, m_X_data, m_X_card, (const char*) "Host X");
  prepareHostTensor(&M, m_M_data, m_X_card, (const char*) "Host M");
  prepareHostTensorFromCpp(&A, NULL, X_card, ndims, (const char*) "Host A"); // init with 0
  prepareHostTensorFromCpp(&Xhat, NULL, X_card, ndims, (const char*) "Host Xhat");
  prepareHostTensorFromCpp(&F, NULL, full_cardinalities, ndims, "Host F");

  size_t Z1_card[3] = {card , card, 0};
  prepareHostTensorFromCpp(&Z1, NULL, Z1_card, ndims, (const char*) "Host Z1", true);
  size_t Z2_card[3] = {0 , card, card};
  prepareHostTensorFromCpp(&Z2, NULL, Z2_card, ndims, (const char*) "Host Z2", true);

  prepareHostTensorFromCpp(&D1_z1, NULL, Z1_card, ndims, (const char*) "Host D1_z1", true);
  prepareHostTensorFromCpp(&D1_z2, NULL, Z2_card, ndims, (const char*) "Host D1_z2", true);

  prepareHostTensorFromCpp(&D2_z1, NULL, Z1_card, ndims, (const char*) "Host D2_z1", true);
  prepareHostTensorFromCpp(&D2_z2, NULL, Z2_card, ndims, (const char*) "Host D2_z2", true);

  print_ct("random Z1 init", &Z1, true);
  print_ct("random Z2 init", &Z2, true);
  print_ct("target X (cpp side)", &X, true);

  
  ///////////////////////////////////////////////////////////////////////////////////////////


  // transfer objects to device /////////////////////////////////////////////////////////////

  REGISTER_CT(Z1); REGISTER_CT(Z2); REGISTER_CT(Xhat); REGISTER_CT(X); REGISTER_CT(D1_z1); REGISTER_CT(D1_z2); REGISTER_CT(D2_z1); REGISTER_CT(D2_z2); REGISTER_CT(A); REGISTER_CT(M); REGISTER_CT(F);

  transferToDevice(full_cardinalities, ndims);
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  // perform NMF operation //////////////////////////////////////////////////////////////////

  for (int iter=0; iter<1; iter++){
    // z1 update
    mct_tensorop_gpu_keys(false, 1, ndims, "Z1", "Z2", "Xhat");
    mct_tensorop_gpu_keys(true , 0, ndims, "X", "Xhat", "A");
    mct_tensorop_gpu_keys(true , 1, ndims, "M", "A", "A");

    //mct_tensorop_gpu_keys(false, 1, ndims, "A", "Z2", "D1_z1");
    //mct_tensorop_gpu_keys(false, 1, ndims, "M", "Z2", "D2_z1");
    mct_tensorop_gpu_keys(false, 1, ndims, "A", "Z2", "D1_z1", "F", NOSWAP, NOSWAP, 1, 2);
    mct_tensorop_gpu_keys(false, 1, ndims, "M", "Z2", "D2_z1", "F", NOSWAP, NOSWAP, 1, 2);

    mct_tensorop_gpu_keys(true , 0, ndims, "D1_z1", "D2_z1", "D1_z1");
    mct_tensorop_gpu_keys(true , 1, ndims, "Z1", "D1_z1", "Z1");

    // z2 update
    mct_tensorop_gpu_keys(false, 1, ndims, "Z1", "Z2", "Xhat");
    mct_tensorop_gpu_keys(true , 0, ndims, "X", "Xhat", "A");
    mct_tensorop_gpu_keys(true , 1, ndims, "M", "A", "A");

    //mct_tensorop_gpu_keys(false, 1, ndims, "A", "Z1", "D1_z2");
    //mct_tensorop_gpu_keys(false, 1, ndims, "M", "Z1", "D2_z2");
    mct_tensorop_gpu_keys(false, 1, ndims, "A", "Z1", "D1_z2", "F", 0, 2, NOSWAP, NOSWAP);
    mct_tensorop_gpu_keys(false, 1, ndims, "M", "Z1", "D2_z2", "F", 0, 2, NOSWAP, NOSWAP);

    mct_tensorop_gpu_keys(true , 0, ndims, "D1_z2", "D2_z2", "D1_z2");
    mct_tensorop_gpu_keys(true , 1, ndims, "Z2", "D1_z1", "Z2");

    //if (iter % 10 == 0 || iter==99 || iter == 98){ // ???
      std::cout << "iter " << iter << std::endl;
      transferFromDevice(Z1.data, "Z1");
      print_ct("current Z1", &Z1, true);
      transferFromDevice(Z2.data, "Z2");
      print_ct("current Z2", &Z2, true);
      //}
  }
  

  ///////////////////////////////////////////////////////////////////////////////////////////

  // transfer results to matlab /////////////////////////////////////////////////////////////

  transferFromDevice(m_Z1, "Z1");
  transferFromDevice(m_Z2, "Z2");

  /*
  transferFromDevice(Z1.data, "Z1");
  transferFromDevice(Z2.data, "Z2");
  std::cout << "bu ne ya" << std::endl;
  for (int i=0; i<4; i++)
    {
      std::cout << (((double*) mxGetData(plhs[0]))[i]) << std::endl;
      std::cout << m_Z1[i] << std::endl;
      std::cout << Z1.data[i] << std::endl;
      std::cout << std::endl;
    }
  */

  //if ( PRINT_CT ) print_ct("result on C side (after)", &h_C,true);

  ///////////////////////////////////////////////////////////////////////////////////////////


  // reset device
  resetDevice();
}



void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if ( COUT ) std::cout << "mct: found " << nrhs << " number of arguments " << std::endl;

  if( nrhs == 0 ){
    std::cout << "mct: not enough input arguments" << std::endl;
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


  if (strcmp(op_name, "nmf_gpu") == 0){
    if ( COUT ) std::cout << "selecting NMF operation on GPU" << std::endl;
    opt=nmf_gpu;
  }else if (strcmp(op_name, "nmf_cpp") == 0){
    if ( COUT ) std::cout << "selecting tensor operation on CPU" << std::endl;
    opt=nmf_cpp;
  }else if (strcmp(op_name, "tensor_gpu") == 0){
    if ( COUT ) std::cout << "selecting tensor operation on GPU" << std::endl;
    opt=tensor_gpu;
  }else if (strcmp(op_name, "tensor_cpp") == 0){
    if ( COUT ) std::cout << "selecting tensor operation on CPU" << std::endl;
    opt=tensor_cpp;
  }else{
    std::cout << "mct: unknown operation: " << op_name << std::endl;
    // print help;
    return;
  }

  if ( opt == nmf_gpu || opt == nmf_cpp ){
    nmfop(nlhs, plhs, nrhs, prhs, opt);
  }else if ( opt == tensor_gpu || opt == tensor_cpp ){
    tensorop(nlhs, plhs, nrhs, prhs, opt);
  }
}
