/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */

#include "mct_tensorop_utils.cuh"
#include "mct_tensorop_gpu.cuh"

#include "cutil_inline.h"
#include "mct_kernels.cuh"

#if CUPRINTF == true
#include "cuPrintf.cu"
#endif


// old functions /////////////////////////////////////////////////////////////////////////


dev_ptrs prepareDeviceParameters(const size_t* h_full_cardinalities, size_t ndims,
                                 const ct* h_A, const ct* h_B, ct* h_C, ct* h_F,
                                 size_t zero_cardinality_dims_elnum,
                                 size_t h_zero_cardinality_dim_tuples_C_element_number,
                                 const size_t* h_zero_cardinality_dim_tuples_C,
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

  if ( COUT ) std::cout << " mct_tensorop_gpu Running kernels " << std::endl << std::endl;

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




//////////////////////////////////////////////////////////////////////////////////////////////
















#include <map>
std::map<std::string,ct*> h_objs;
std::map<std::string,size_t*> d_obj_strides;
std::map<std::string,double*> d_obj_data;
size_t* d_full_cardinalities;


void register_ct(std::string key, ct* obj){
  h_objs[key] = obj;
}

void transferToDevice(const size_t* h_full_cardinalities, size_t ndims){
  std::map<std::string,ct*>::iterator it;

  // copy h_full_cardinalities to device manually
  cutilSafeCall(cudaMalloc((void**)&(d_full_cardinalities), sizeof(size_t)*ndims));
  cutilSafeCall(cudaMemcpy(d_full_cardinalities, h_full_cardinalities, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));

  // copy registered objects to device
  for(it=h_objs.begin(); it != h_objs.end(); it++){
    std::string key = it->first;
    ct*         obj = it->second;

    // initialize temporary storage structures for device memory
    size_t* d_strides;
    double* d_data;

    // copy to device
    cutilSafeCall(cudaMalloc((void**)&(d_strides), sizeof(size_t)*ndims));
    cutilSafeCall(cudaMemcpy(d_strides, obj->strides, sizeof(size_t)*ndims, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMalloc((void**)&(d_data), obj->mem_size));
    cutilSafeCall(cudaMemcpy(d_data, obj->data, obj->mem_size, cudaMemcpyHostToDevice));

    // store pointers in global storage
    d_obj_strides[key] = d_strides;
    d_obj_data   [key] = d_data;

  }
}


// perform tensor operation with specified objects
bool mct_tensorop_gpu_keys(bool isHadamard,
                           size_t use_multiplication,
                           size_t ndims,
                           std::string A, std::string B, std::string C, std::string F){

  // check requested objects are registered
  if ( h_objs.find(A) == h_objs.end() ||
       h_objs.find(B) == h_objs.end() ||
       h_objs.find(C) == h_objs.end() ||
       h_objs.find(F) == h_objs.end() ){
    std::cout << "mct_tensorop_gpu_keys: all of requested keys should be registered "
	      << " requested keys: A " << A << " B " << B << " C " << C << " F " << F
	      << std::endl;
    return false;
  }

  // prepare range permutation vector //////////////////////////////////////////////////////
  size_t zero_cardinality_dim_tuple_size_C = 0;
  size_t zero_cardinality_dim_tuples_C_element_number = 0;
  size_t* h_zero_cardinality_dim_tuples_C = NULL;
  size_t* d_zero_cardinality_dim_tuples_C = NULL;

  if ( isHadamard == false){
    std::vector<size_t> zero_cardinality_dims;
    for ( size_t dim=0; dim<ndims; dim++ ){
      if ( h_objs[C]->cardinalities[dim] == 0 && h_objs[F]->cardinalities[dim] != 0 ){
        zero_cardinality_dims.push_back(h_objs[F]->cardinalities[dim]);
      }
    }

    if ( COUT ) {
      std::cout << "zero_cardinality_dims" << std::endl;
      for ( size_t j=0; j<zero_cardinality_dims.size(); j++){
        std::cout << zero_cardinality_dims.at(j) << std::endl;
      }
    }

    zero_cardinality_dim_tuple_size_C = zero_cardinality_dims.size();

    h_zero_cardinality_dim_tuples_C =
      gen_range_permutation(zero_cardinality_dims,
                            &(zero_cardinality_dim_tuples_C_element_number));

    // transfer to device
    cutilSafeCall(cudaMalloc((void**)&(d_zero_cardinality_dim_tuples_C),
                             sizeof(size_t)*zero_cardinality_dim_tuples_C_element_number));
    cutilSafeCall(cudaMemcpy(d_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuples_C,
                             sizeof(size_t)*zero_cardinality_dim_tuples_C_element_number, cudaMemcpyHostToDevice));

  }

  
  
  ////////////////////////////////////////////////////////////////////////////////////////


#if CUPRINTF == true
  cudaPrintfInit();
#endif

  if ( COUT ) std::cout << " mct_tensorop_gpu_keys Running kernels " << std::endl << std::endl;

  bool got_zeros = false;

  if ( isHadamard ){

    if (use_multiplication == 1)
      hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(d_obj_data[A],
						      d_obj_data[B],
						      d_obj_data[C],
						      h_objs[C]->element_number);
    else
      hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(d_obj_data[A],
						      d_obj_data[B],
						      d_obj_data[C],
						      h_objs[C]->element_number);

  }else{

    // generate the full output
    genFullResult<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(d_full_cardinalities, ndims,
                                                    d_obj_strides[A], d_obj_strides[B], d_obj_strides[F],
                                                    d_obj_data[A], d_obj_data[B], d_obj_data[F],
                                                    h_objs[F]->element_number,
                                                    use_multiplication);

    // test full result
    cutilSafeCall(cudaMemcpy(h_objs[F]->data, d_obj_data[F], h_objs[F]->mem_size, cudaMemcpyDeviceToHost));
    if ( PRINT_CT ) print_ct("genFullResult", h_objs[F], true);

    // if no contraction is required, result is already stored in F, return that
    got_zeros = false;
    for ( size_t dim=0; dim<ndims; dim++){
      if ( h_objs[C]->cardinalities[dim] == 0 && h_objs[F]->cardinalities[dim] != 0){
        if ( COUT ) std::cout << " GOT ZEROS found zero on h_objs[C] dimension " << dim << std::endl;
        got_zeros = true;
        break;
      }
    }

    if ( got_zeros ){
      // contract on required dimensions
      if ( COUT ) std::cout << "performing contaction" << std::endl;
      contractFintoC<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(ndims,
                                                       d_obj_strides[F], d_obj_strides[C],
                                                       d_obj_data[F], d_obj_data[C],
                                                       h_objs[C]->element_number,
                                                       d_zero_cardinality_dim_tuples_C,
                                                       zero_cardinality_dim_tuple_size_C,
                                                       zero_cardinality_dim_tuples_C_element_number);
    }
  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaDeviceSynchronize();

#if CUPRINTF == true
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif


  ///////////////////////////////////////////////////////////////////////////////////////////

  return true;
}


void resetDevice(){
  cudaDeviceReset();
}


void transferFromDevice(double* matlab_storage, std::string d_storage_key){
  cutilSafeCall(cudaMemcpy(matlab_storage, 
			   d_obj_data[d_storage_key],
			   h_objs[d_storage_key]->mem_size,
			   cudaMemcpyDeviceToHost));
}
