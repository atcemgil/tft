/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#include "../common/utils.cuh"
#include "tensorop_par.cuh"

#include "cutil_inline.h"
#include "../common/kernels.cuh"

#include "../common/cuPrintf.cuh"
//#include "../common/cuPrintf.cu"

#include "cuda.h"

std::map<std::string,size_t*> d_obj_strides;
std::map<std::string,size_t*> d_obj_cards;
std::map<std::string,double*> d_obj_data;

size_t* d_full_cardinalities;

std::map<std::string,size_t*> get_d_obj_strides(){ return d_obj_strides; }
std::map<std::string,size_t*> get_d_obj_cards()  { return d_obj_cards; }
std::map<std::string,double*> get_d_obj_data()   { return d_obj_data; }



// old functions /////////////////////////////////////////////////////////////////////////


dev_ptrs prepareDeviceParameters(size_t ndims,
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

void tensorop_par(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, size_t h_zero_cardinality_dim_tuple_size_C, size_t use_multiplication){
  
  // prepare device memory for tensors  /////////////////////////////////////////////////////

  dev_ptrs dp = prepareDeviceParameters(ndims, &h_A, &h_B, &h_C, &h_F,
                                        h_zero_cardinality_dim_tuple_size_C,
                                        h_zero_cardinality_dim_tuples_C_element_number,
                                        h_zero_cardinality_dim_tuples_C,
                                        isHadamard);


  ///////////////////////////////////////////////////////////////////////////////////////////


  // run kernels //////////////////////////////////////////////////////////////////////////////


  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  if (CUPRINTF == true)
    cudaPrintfInit();
  if ( COUT ) std::cout << " tensorop_gpu Running kernels " << std::endl << std::endl;

  bool got_zeros = false;

  if ( isHadamard ){

    if (use_multiplication == 1)
      hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(dp.d_data_A,dp.d_data_B,dp.d_data_C,h_C.element_number, CUPRINTF);
    else
      hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(dp.d_data_A,dp.d_data_B,dp.d_data_C,h_C.element_number, CUPRINTF);

  }else{

    // generate the full output
    genFullResult<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(dp.d_full_cardinalities, ndims,
                                                    dp.d_strides_A, dp.d_strides_B, dp.d_strides_F,
                                                    dp.d_data_A, dp.d_data_B, dp.d_data_F,
                                                    h_F.element_number, h_A.element_number, h_B.element_number, 
                                                    use_multiplication, CUPRINTF);

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
                                                       dp.zero_cardinality_dim_tuples_C_element_number,
						       CUPRINTF);
    }
  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaThreadSynchronize();
  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  cutilCheckError(cutDeleteTimer(timer));

  if (CUPRINTF == true){
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
  }


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










static unsigned long inKB(unsigned long bytes)
{ return bytes/1024; }

static unsigned long inMB(unsigned long bytes)
{ return bytes/(1024*1024); }

void printStats(unsigned long free, unsigned long total)
{
  printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));
  printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));
  printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
}


void printGPUstats(){
  size_t free, total;
  int gpuCount, i;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;

  cuInit(0);

  cuDeviceGetCount(&gpuCount);
  printf("Detected %d GPU\n",gpuCount);

  for (i=0; i<gpuCount; i++)
    {
      cuDeviceGet(&dev,i);
      cuCtxCreate(&ctx, 0, dev);
      res = cuMemGetInfo(&free, &total);
      if(res != CUDA_SUCCESS)
	printf("!!!! cuMemGetInfo failed! (status = %x)", res);
      printf("^^^^ Device: %d\n",i);
      printStats(free, total);
      cuCtxDetach(ctx);
    }
}




size_t transferToDevice(size_t full_ndims, size_t cur_mem){
  std::map<std::string,ct*>::iterator it;

  // copy h_full_cardinalities to device manually
  cur_mem += sizeof(size_t)*full_ndims;
  cutilSafeCall(cudaMalloc((void**)&(d_full_cardinalities), sizeof(size_t)*full_ndims));

  cutilSafeCall(cudaMemcpy(d_full_cardinalities, h_full_cardinalities, sizeof(size_t)*full_ndims, cudaMemcpyHostToDevice));

  // copy registered objects to device
  for(it=h_objs.begin(); it != h_objs.end(); it++){
    std::string key = it->first;
    ct*         obj = it->second;
    std::cout << " transferToDevice key " << key << " cur_mem " << cur_mem << " will allocate " << sizeof(size_t)*obj->ndims + sizeof(size_t)*obj->ndims + obj->mem_size << " more " << std::endl;
    //printGPUstats();

    // initialize temporary storage structures for device memory
    size_t* d_strides;
    size_t* d_cards;
    double* d_data;

    // copy to device
    cur_mem += sizeof(size_t)*obj->ndims;
    cutilSafeCall(cudaMalloc((void**)&(d_strides), sizeof(size_t)*obj->ndims));
    cutilSafeCall(cudaMemcpy(d_strides, obj->strides, sizeof(size_t)*obj->ndims, cudaMemcpyHostToDevice));

    cur_mem += sizeof(size_t)*obj->ndims;
    cutilSafeCall(cudaMalloc((void**)&(d_cards), sizeof(size_t)*obj->ndims));
    cutilSafeCall(cudaMemcpy(d_cards, obj->cardinalities, sizeof(size_t)*obj->ndims, cudaMemcpyHostToDevice));

    // std::cout << "printing data of " << key << " mem size " << obj->mem_size << std::endl;
    // for(int i=0; i<obj->element_number; i++)
    //   std::cout << " selam" << obj->data[i] << std::endl;

    if( obj->data != NULL ){
      cur_mem += obj->mem_size;
      cutilSafeCall(cudaMalloc((void**)&(d_data), obj->mem_size));
      cutilSafeCall(cudaMemcpy(d_data, obj->data, obj->mem_size, cudaMemcpyHostToDevice));
    }

    // store pointers in global storage
    d_obj_strides[key] = d_strides;
    d_obj_cards  [key] = d_cards;

    if( obj->data != NULL ){
      d_obj_data   [key] = d_data;
    }

    //printData<<<1,1>>>(d_data, obj->mem_size, 123123123);
  }

  return cur_mem;
}







// perform tensor operation with specified objects
// ct* result contains a pointer to the ct objects containing the result. It may be input C or F.
bool tensorop_par_keys(operation_type op_type,
		       size_t ndims,
		       bool* result_in_F,
		       std::string A, std::string B, std::string C,
		       std::string F, int to_power_A, int to_power_B
		       ){
  if (check_input_keys(A,B,C,F) == false){
    return false;
  }


  if( CUPRINTF )
    cudaPrintfInit();

  if ( COUT ) std::cout << " tensorop_gpu_keys Running kernels " << std::endl << std::endl;

  bool got_zeros = false;

  if ( is_hadamard(op_type) ){

    if (is_multiplication(op_type) )
      hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(d_obj_data[A],
                                                      d_obj_data[B],
                                                      d_obj_data[C],
                                                      h_objs[C]->element_number,
						      CUPRINTF,
						      to_power_A,
						      to_power_B);
    else if (is_division(op_type) )
      hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(d_obj_data[A],
                                                      d_obj_data[B],
                                                      d_obj_data[C],
                                                      h_objs[C]->element_number,
						      CUPRINTF,
						      to_power_A,
						      to_power_B);
    else if (is_addition(op_type) )
      hadamard_sum<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(d_obj_data[A],
                                                      d_obj_data[B],
                                                      d_obj_data[C],
                                                      h_objs[C]->element_number,
						      CUPRINTF,
						      to_power_A,
						      to_power_B);

    if ( PRINT_CT ) {
      transferFromDevice(h_objs[A]->data, A);
      print_ct("tensorop gpu A", h_objs[A], true);
      transferFromDevice(h_objs[B]->data, B);
      print_ct("tensorop gpu B", h_objs[B], true);
      transferFromDevice(h_objs[C]->data, C);
      print_ct("tensorop gpu C ", h_objs[C], true);
    }


    *result_in_F=false;
  }else{

    // generate the full output
    genFullResult<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(d_full_cardinalities, ndims,
                                                    d_obj_strides[A], d_obj_strides[B], d_obj_strides[F],
                                                    d_obj_data[A], d_obj_data[B], d_obj_data[F],
                                                    h_objs[F]->element_number, h_objs[A]->element_number, h_objs[B]->element_number,
                                                    is_multiplication(op_type),
						    CUPRINTF,
						    to_power_A, to_power_B);

    // test full result
    cutilSafeCall(cudaMemcpy(h_objs[F]->data, d_obj_data[F], h_objs[F]->mem_size, cudaMemcpyDeviceToHost));
    if ( PRINT_CT ) {
      transferFromDevice(h_objs[A]->data, A);
      print_ct("tensorop gpu A", h_objs[A], true);

      transferFromDevice(h_objs[B]->data, B);
      print_ct("tensorop gpu B", h_objs[B], true);

      transferFromDevice(h_objs[F]->data, F);
      print_ct("tensorop gpu F", h_objs[F], true);
    }

    // if no contraction is required, result is already stored in F, return that
    got_zeros = false;
    for ( size_t dim=0; dim<ndims; dim++){
      if ( h_objs[C]->cardinalities[dim] == 0 && h_objs[F]->cardinalities[dim] != 0){
        if ( COUT ) std::cout << " GOT ZEROS found zero on h_objs[C] dimension " << dim << std::endl;
        got_zeros = true;
        break;
      }
    }

    *result_in_F=true;

    if ( got_zeros ){

      // prepare range permutation vector //////////////////////////////////////////////////////
      size_t zero_cardinality_dim_tuple_size_C = 0;
      size_t zero_cardinality_dim_tuples_C_element_number = 0;
      size_t* h_zero_cardinality_dim_tuples_C = NULL;
      size_t* d_zero_cardinality_dim_tuples_C = NULL;

      if ( is_hadamard(op_type) == false){
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



      // contract on required dimensions
      if ( COUT ) std::cout << "performing contaction" << std::endl;
      contractFintoC<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(ndims,
                                                       d_obj_strides[F], d_obj_strides[C],
                                                       d_obj_data[F], d_obj_data[C],
                                                       h_objs[C]->element_number,
                                                       d_zero_cardinality_dim_tuples_C,
                                                       zero_cardinality_dim_tuple_size_C,
                                                       zero_cardinality_dim_tuples_C_element_number,
						       CUPRINTF);

    if ( PRINT_CT ) {
      transferFromDevice(h_objs[C]->data, C);
      print_ct("tensorop gpu C ", h_objs[C], true);
    }


      *result_in_F=false;
    }
  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaDeviceSynchronize();

  if ( CUPRINTF == true ){
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
  }


  ///////////////////////////////////////////////////////////////////////////////////////////


  return true;
}


















void resetDevice(){
  cudaDeviceReset();
}

void transferFromDevice(double* matlab_storage, std::string d_storage_key){

  if ( COUT )
    std::cout << " transferFromDevice copying "
              << " data mem size " << h_objs[d_storage_key]->mem_size
              << " element number " << h_objs[d_storage_key]->element_number
              << " for key " << d_storage_key
	      << " matlab_storage " << matlab_storage
	      << " d_obj_data[d_storage_key] " << d_obj_data[d_storage_key]
              << std::endl;

  cutilSafeCall(cudaMemcpy(matlab_storage,
                           d_obj_data[d_storage_key],
                           h_objs[d_storage_key]->mem_size,
                           cudaMemcpyDeviceToHost));
}
