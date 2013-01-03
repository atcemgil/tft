/*
 * author: ck
 * created: 26.03.2012
 * advisor: atc
 */

#include <string.h>
#include <sstream>

#include "utils.cuh"
#include "../common/cuPrintf.cuh"
#include "cutil_inline.h"
#include "../common/kernels.cuh"

#include <sys/time.h>

//void tensorop(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool is_parallel){
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  read_config();
  //print_config();


  if( nlhs != 2 ){
    std::cout << "test_gmult: tensor operation requires exactly 2 output argument, given " << nlhs << std::endl;
    // print help;
    return;
  }

  if ( nrhs != 9 ){
    std::cout << "test_gmult: tensor operation requires 7 arguments. "
              << "A, dimensions of A, B, dimensions of B, dimensions of C,"
              << " use_multiplication(1 uses multiplication, 0 uses division) "
              << " use_F episode_num iter_num"
              << std::endl;
    return;
  }

  const mxArray* m_A_data = prhs[0];
  const mxArray* m_A_card = prhs[1];

  const mxArray* m_B_data = prhs[2];
  const mxArray* m_B_card = prhs[3];

  const mxArray* m_C_card = prhs[4];

  size_t use_multiplication = ((double *)mxGetData(prhs[5]))[0];

  bool use_F = ((double *)mxGetData(prhs[6]))[0];

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


  //tensorop_par(isHadamard, h_A, h_B, h_C, m_C, h_F, ndims, h_zero_cardinality_dim_tuples_C_element_number, h_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuple_size_C, use_multiplication);










  // prepare device memory for tensors  /////////////////////////////////////////////////////

  //clock_t start_mem, end_mem;
  timeval start_mem, end_mem;
  gettimeofday(&start_mem, NULL);
  //start_mem = clock();
  

  dev_ptrs dp = prepareDeviceParameters(ndims, &h_A, &h_B, &h_C, &h_F,
                                        h_zero_cardinality_dim_tuple_size_C,
                                        h_zero_cardinality_dim_tuples_C_element_number,
                                        h_zero_cardinality_dim_tuples_C,
                                        isHadamard);





  // operation arguments for mops
  operands ops;

  size_t operand_elnum = 2;

  size_t** h_strides_operand_pointers = (size_t**) malloc( operand_elnum * sizeof(size_t*) );
  size_t** h_cards_operand_pointers   = (size_t**) malloc( operand_elnum * sizeof(size_t*) );
  double** h_operand_pointers         = (double**) malloc( operand_elnum * sizeof(double*) );

  size_t* d_cards_A;
  cutilSafeCall(cudaMalloc((void**)&(d_cards_A), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(d_cards_A, h_A.cardinalities, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));

  size_t* d_cards_B;
  cutilSafeCall(cudaMalloc((void**)&(d_cards_B), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(d_cards_B, h_B.cardinalities, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));


  h_strides_operand_pointers[0] = dp.d_strides_A;
  h_cards_operand_pointers[0] = d_cards_A;
  h_operand_pointers[0] = dp.d_data_A;

  h_strides_operand_pointers[1] = dp.d_strides_B;
  h_cards_operand_pointers[1] = d_cards_B;
  h_operand_pointers[1] = dp.d_data_B;


  // copy to device
  //cur_mem += sizeof(size_t*)*operand_elnum;
  //std::cout << "   cur_mem increment by " << sizeof(size_t*)*operand_elnum << " new cur_mem " << cur_mem;
  cutilSafeCall(cudaMalloc((void**)&(ops.d_strides_operand_pointers), sizeof(size_t*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops.d_strides_operand_pointers, h_strides_operand_pointers, sizeof(size_t*)*operand_elnum, cudaMemcpyHostToDevice));

  //cur_mem += sizeof(size_t*)*operand_elnum;
  //std::cout << "   cur_mem increment by " << sizeof(size_t*)*operand_elnum << " new cur_mem " << cur_mem;
  cutilSafeCall(cudaMalloc((void**)&(ops.d_cards_operand_pointers), sizeof(size_t*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops.d_cards_operand_pointers, h_cards_operand_pointers, sizeof(size_t*)*operand_elnum, cudaMemcpyHostToDevice));

  //cur_mem += sizeof(double*)*operand_elnum;
  //std::cout << "   cur_mem increment by " << sizeof(double*)*operand_elnum << " new cur_mem " << cur_mem;
  cutilSafeCall(cudaMalloc((void**)&(ops.d_operand_pointers), sizeof(double*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops.d_operand_pointers, h_operand_pointers, sizeof(double*)*operand_elnum, cudaMemcpyHostToDevice));

  // if( h_to_power != NULL ){
  //   //cur_mem += sizeof(int)*operand_elnum;
  //   //std::cout << "   cur_mem increment by " << sizeof(int)*operand_elnum << " new cur_mem " << cur_mem;
  //   cutilSafeCall(cudaMalloc((void**)&(ops.d_to_power), sizeof(int)*operand_elnum));
  //   cutilSafeCall(cudaMemcpy(ops.d_to_power, h_to_power, sizeof(int)*operand_elnum, cudaMemcpyHostToDevice));
  // }else{

  // }

  ops.d_to_power = NULL;

  //std::cout << " gen_operation_arguments elnum " << operand_elnum << " curmem " << cur_mem << std::endl;

  gettimeofday(&end_mem, NULL);
  //end_mem = clock();
  timeval diff_mem;
  timersub(&end_mem, &start_mem, &diff_mem);

  std::cout << "mem time " << diff_mem.tv_sec << " sec " << diff_mem.tv_usec << " usec " << std::endl;
  //std::cout << "mem time " << (end_mem - start_mem) / (float)CLOCKS_PER_SEC << " seconds" << std::endl;
  



  ///////////////////////////////////////////////////////////////////////////////////////////









  if (CUPRINTF == true){
    cudaPrintfInit();
  }



  if ( COUT ) std::cout << " tensorop_gpu Running kernels " << std::endl << std::endl;


  //int episode_num  = ((double *)mxGetData(prhs[7]))[0];; // first episode is used for warm up
  int iter_num = ((double *)mxGetData(prhs[8]))[0];;

  int total_iter=0;
  timeval start;
  timeval end;
  //clock_t start, end;

  if ( isHadamard ){
    std::cout << " will use hadamard " << std::endl;
  } else if ( use_F ){
    std::cout << " will use F " << std::endl;
  }else{
    std::cout << " will use calculate_C_mops " << std::endl;
  }


  gettimeofday(&start, NULL);
  //start = clock();

  //for( int episode=0; episode<episode_num; episode++ ){
  for( int iter=0; iter<iter_num; iter++) {
    total_iter++;

    //if( episode != 0 ) total_iter++;
    //if( episode == 1 && iter == 0 ) gettimeofday(&start, NULL);

    //std::cout << " episode " << episode << " iter " << iter << " total_iter " << total_iter << std::endl;

    if ( isHadamard ){

      if (use_multiplication == 1)
        hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(dp.d_data_A,dp.d_data_B,dp.d_data_C,h_C.element_number, CUPRINTF);
      else
        hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(dp.d_data_A,dp.d_data_B,dp.d_data_C,h_C.element_number, CUPRINTF);



    } else if ( use_F ){


      // run kernels //////////////////////////////////////////////////////////////////////////////


      // unsigned int timer = 0;
      // cutilCheckError(cutCreateTimer(&timer));
      // cutilCheckError(cutStartTimer(timer));


      bool got_zeros = false;


      // generate the full output
      genFullResult<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(dp.d_full_cardinalities, ndims,
                                                      dp.d_strides_A, dp.d_strides_B, dp.d_strides_F,
                                                      dp.d_data_A, dp.d_data_B, dp.d_data_F,
                                                      h_F.element_number, h_A.element_number, h_B.element_number,
                                                      use_multiplication, CUPRINTF);

      // test full result
      if ( PRINT_CT ) {
        cutilSafeCall(cudaMemcpy(h_F.data, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
        print_ct("genFullResult", &h_F,true);
      }

      // if no contraction is required, result is already stored in F, return that
      got_zeros = false;
      for ( size_t dim=0; dim<ndims; dim++){
        if ( h_C.cardinalities[dim] == 0 && h_F.cardinalities[dim] != 0){
          if ( COUT ) std::cout << " GOT ZEROS found zero on h_C dimension " << dim << std::endl;
          got_zeros = true;
          break;
        }
      }

      if ( got_zeros ){
        // contract on required dimensions
        if ( COUT ) std::cout << "performing contaction" << std::endl;
        contractFintoC<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(ndims,
                                                         dp.d_strides_F, dp.d_strides_C,
                                                         dp.d_data_F, dp.d_data_C,
                                                         h_C.element_number,
                                                         dp.d_zero_cardinality_dim_tuples_C,
                                                         dp.zero_cardinality_dim_tuple_size_C,
                                                         dp.zero_cardinality_dim_tuples_C_element_number,
                                                         CUPRINTF);
      }
    }else{
      /// DO NOT USE F


      calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                          2,

                                                          (size_t**) (ops.d_strides_operand_pointers),

                                                          (size_t*) (dp.d_strides_C),
                                                          (size_t*) (dp.d_full_cardinalities),

                                                          (size_t**) (ops.d_cards_operand_pointers),
                                                          (double**) (ops.d_operand_pointers),

                                                          (double*) (dp.d_data_C),
                                                          //(double*) (get_d_obj_data()["Z0"]),
                                                          (size_t) (h_C.element_number),
                                                          (size_t) 1,
                                                          CUPRINTF,1);


    }

  }

  //}


  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaThreadSynchronize();
  // stop and destroy timer
  // cutilCheckError(cutStopTimer(timer));
  // cutilCheckError(cutDeleteTimer(timer));

  if (CUPRINTF == true){
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
  }


  // if ( isHadamard ){
  cutilSafeCall(cudaMemcpy(m_C, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
  // }else{
  //   if ( got_zeros ){
  //     cutilSafeCall(cudaMemcpy(m_C, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
  //     cutilSafeCall(cudaMemcpy(h_C.data, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
  //     if ( PRINT_CT ) print_ct("result on C side (after)", &h_C,true);
  //   }else{
  //     cutilSafeCall(cudaMemcpy(m_C, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
  //   }
  // }


  cudaThreadExit();
  ///////////////////////////////////////////////////////////////////////////////////////////


  gettimeofday(&end, NULL);
  timeval diff;
  timersub(&end, &start, &diff);
  std::cout << "total time " << diff.tv_sec << " sec " << diff.tv_usec << " usec " << std::endl;
  std::cout << "average time " << diff.tv_sec/(float)total_iter << " sec " << diff.tv_usec/(float)total_iter << " usec " << std::endl;
  // end = clock();
  // std::cout << "total time " << (end-start)/(float)CLOCKS_PER_SEC << " seconds" << std::endl;


  mwSize argMatDim[1] = {1};
  plhs[1] = mxCreateNumericArray(1,argMatDim,mxDOUBLE_CLASS,mxREAL);
  double* m_t = (double*) mxGetPr(plhs[1]);
  *m_t = diff.tv_sec/(float)total_iter + (0.000001 * (diff.tv_usec/(float)total_iter));
  //*m_t = (end-start)/(float)CLOCKS_PER_SEC;



  free_ct(&h_A);
  free_ct(&h_B);
  free_ct(&h_C);
  free_ct(&h_F);

  // timeval tmpt;
  // gettimeofday(&tmpt, NULL);
  // timersub(&tmpt, &end, &diff);
  // std::cout << "end to final " << diff.tv_sec << " sec " << diff.tv_usec << " usec " << std::endl;
}
