/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */

#include "mct_tensorop_utils.cuh"
#include "cutil_inline.h"
#include "mct_kernels.cuh"

#if CUPRINTF == true
#include "cuPrintf.cu"
#endif

dev_ptrs prepareDeviceParameters(const size_t* h_full_cardinalities, size_t ndims, const ct* h_A, const ct* h_B, ct* h_C, ct* h_F,
                                 size_t zero_cardinality_dims_elnum, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C,
                                 bool isHadamard){
  dev_ptrs dp;

  cutilSafeCall(cudaMalloc((void**)&(dp.d_full_cardinalities), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(dp.d_full_cardinalities, h_full_cardinalities, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));


  cutilSafeCall(cudaMalloc((void**)&(dp.d_strides_A), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(dp.d_strides_A, h_A->strides, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(dp.d_strides_B), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(dp.d_strides_B, h_B->strides, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(dp.d_strides_C), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(dp.d_strides_C, h_C->strides, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(dp.d_strides_F), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(dp.d_strides_F, h_F->strides, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));



  cutilSafeCall(cudaMalloc((void**)&(dp.d_data_A), h_A->mem_size));
  cutilSafeCall(cudaMemcpy(dp.d_data_A, h_A->data, h_A->mem_size, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(dp.d_data_B), h_B->mem_size));
  cutilSafeCall(cudaMemcpy(dp.d_data_B, h_B->data, h_B->mem_size, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(dp.d_data_F), h_F->mem_size));
  cutilSafeCall(cudaMemcpy(dp.d_data_F, h_F->data, h_F->mem_size, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(dp.d_data_C), h_C->mem_size));
  cutilSafeCall(cudaMemcpy(dp.d_data_C, h_C->data, h_C->mem_size, cudaMemcpyHostToDevice));


  dp.zero_cardinality_dim_tuple_size_C = zero_cardinality_dims_elnum;
  dp.zero_cardinality_dim_tuples_C_element_number = h_zero_cardinality_dim_tuples_C_element_number;

  if ( isHadamard == false){
    cutilSafeCall(cudaMalloc((void**)&(dp.d_zero_cardinality_dim_tuples_C),
                             sizeof(size_t)*dp.zero_cardinality_dim_tuples_C_element_number));
    cutilSafeCall(cudaMemcpy(dp.d_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuples_C,
                             sizeof(size_t)*dp.zero_cardinality_dim_tuples_C_element_number, cudaMemcpyHostToDevice));
  }

  return dp;
}

void mct_tensorop_gpu(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, const size_t* h_full_cardinalities,size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, size_t h_zero_cardinality_dim_tuple_size_C, size_t use_multiplication){

  //std::cout  << " mct_tensorop_gpu SELAM " << std::endl;

    // prepare device memory for tensors  /////////////////////////////////////////////////////

    dev_ptrs dp = prepareDeviceParameters(h_full_cardinalities, ndims, &h_A, &h_B, &h_C, &h_F,
                                          h_zero_cardinality_dim_tuple_size_C,
                                          h_zero_cardinality_dim_tuples_C_element_number,
                                          h_zero_cardinality_dim_tuples_C,
                                          isHadamard);


    ///////////////////////////////////////////////////////////////////////////////////////////


    // run kernels //////////////////////////////////////////////////////////////////////////////


    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

#if CUPRINTF == true
    cudaPrintfInit();
#endif

    if ( COUT ) std::cout << " Running kernels " << std::endl << std::endl;

    bool got_zeros = false;

    if ( isHadamard ){

      if (use_multiplication == 1)
	hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(dp.d_data_A,dp.d_data_B,dp.d_data_C,h_C.element_number);
      else
	hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(dp.d_data_A,dp.d_data_B,dp.d_data_C,h_C.element_number);

    }else{

      // generate the full output
      genFullResult<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(dp.d_full_cardinalities, ndims, 
						      dp.d_strides_A, dp.d_strides_B, dp.d_strides_F, 
						      dp.d_data_A, dp.d_data_B, dp.d_data_F, 
						      h_F.element_number,
						      use_multiplication);

      // test full result
      cutilSafeCall(cudaMemcpy(h_F.data, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
      if ( PRINT_CT ) print_ct("genFullResult", &h_F,true);

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
                                                         dp.zero_cardinality_dim_tuples_C_element_number);
      }
    }

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");
    cudaThreadSynchronize();
    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    cutilCheckError(cutDeleteTimer(timer));

#if CUPRINTF == true
      cudaPrintfDisplay(stdout, true);
      cudaPrintfEnd();
#endif


    if ( isHadamard ){
      cutilSafeCall(cudaMemcpy(m_C, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
    }else{
      if ( got_zeros ){
        cutilSafeCall(cudaMemcpy(m_C, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(h_C.data, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
        if ( PRINT_CT ) print_ct("result on C side (after)", &h_C,true);
      }else{
        cutilSafeCall(cudaMemcpy(m_C, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
      }
    }

    cudaThreadExit();
    ///////////////////////////////////////////////////////////////////////////////////////////

}
