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

#include <vector>

void print_ct(char* txt, ct* ct, bool printdata=false){ //bool print_config=false,

  std::cout << txt << std::endl;

  //if (print_config) print_ct_config(txt, ct->config);

  std::cout << "Mem size " << ct->mem_size << std::endl;

  std::cout << "Element number " << ct->element_number << std::endl;

  std::cout << "Cardinalities for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->ndims; i++){
    std::cout << ct->cardinalities[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Strides for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->ndims; i++){
    std::cout << ct->strides[i] << " ";
  }
  std::cout << std::endl;

  if (printdata){
    std::cout << "Data" << std::endl;
    for (size_t i=0; i< ct->element_number; i++){
      std::cout << ct->data[i] << " ";
    }
  }
  std::cout << std::endl << std::endl << std::endl;
}




void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims){
  h_ct->ndims = ndims;
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * h_ct->ndims );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * h_ct->ndims );

  size_t elnum=1;
  // assign cardinalities for the tensor objects and init cur_ind values
  size_t cum_sum=1;
  for (size_t i=0; i<h_ct->ndims; i++){
    elnum *= tensor_card[i];

    h_ct->cardinalities[i] = tensor_card[i];
    std::cout << "TC dim "<< i << " cardinality assignment: "
              << h_ct->cardinalities[i] << " <- " << tensor_card[i] << std::endl;

    if ( h_ct->cardinalities[i] == 0){
      h_ct->strides[i]=0;
    }else{
      h_ct->strides[i]=cum_sum;
      cum_sum *= h_ct->cardinalities[i];
    }
  }

  // assign h_ct host data
  std::cout << " prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;

  if (data == NULL){
    h_ct->data = (double*)calloc( h_ct->mem_size, sizeof(double) );
  }else{
    h_ct->data = (double*)malloc( h_ct->mem_size );
    memcpy(h_ct->data, data, h_ct->mem_size);
  }

  print_ct("prepareDeviceTensor h_ct",h_ct,true);

}

void prepareHostTensor(ct* h_ct, const mxArray* m_data, const mxArray* tensor_card){
  h_ct->ndims = mxGetNumberOfElements(tensor_card); // assumes both objects of same size
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );

  // assign cardinalities for the tensor objects and init cur_ind values
  size_t cum_sum=1;
  for (size_t i=0; i<h_ct->ndims; i++){
    h_ct->cardinalities[i] = ((double *)mxGetData(tensor_card))[i];
    std::cout << "TC dim "<< i << " cardinality assignment: "
              << h_ct->cardinalities[i]
              << " <- " << ((double *)mxGetData(tensor_card))[i] << std::endl;

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
  h_ct->element_number = elnum;

  h_ct->data = (double*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (double*)mxGetData(m_data), h_ct->mem_size);

  print_ct("prepareDeviceTensor h_ct",h_ct,true);
}


// Recursive function which generates all permutations of a given list
void gen_range_permutation_helper(std::vector<size_t> iter_dims, std::vector<size_t> cur_perm, std::vector<size_t>* acc){
  if ( iter_dims.size() == 0 ){
    acc->insert(acc->end(), cur_perm.begin(), cur_perm.end());
  }else{
    // pick one dimension from iter_dims and iterate for each element in its cardinality
    size_t one_dim = iter_dims.front();
    // remove that dim for the remaining recursions
    iter_dims.erase(iter_dims.begin());

    for ( size_t i=0; i<one_dim ; i++){
      std::vector<size_t> tmp_vec (cur_perm.begin(), cur_perm.end());
      tmp_vec.push_back(i);
      gen_range_permutation_helper( iter_dims, tmp_vec, acc );
    }
  }
}

// generates range-permutation of numbers given in permutation_list
// Example
// (2 3) -> ( 0,0,   0,1,  0,2,   1,0,   1,1,  1,2 )
size_t* gen_range_permutation(std::vector<size_t> permutation_list, size_t* elnum){

  std::vector<size_t> empty_list;
  std::vector<size_t> acc;

  gen_range_permutation_helper(permutation_list, empty_list, &acc);

  size_t* acc_array = (size_t*) malloc(sizeof(size_t)*acc.size());
  std::copy(acc.begin(), acc.end(), acc_array);
  (*elnum) = acc.size();
  return acc_array;
}



dev_ptrs prepareDeviceParameters(size_t* h_full_cardinalities, size_t ndims, ct* h_A, ct* h_B, ct* h_C, ct* h_F){
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


  std::vector<size_t> zero_cardinality_dims;
  for ( size_t dim=0; dim<ndims; dim++ ){
    if ( h_C->cardinalities[dim] == 0 ){
      zero_cardinality_dims.push_back(dim);
    }
  }

  size_t h_zero_cardinality_dim_tuples_C_elnum;
  size_t* h_zero_cardinality_dim_tuples_C = gen_range_permutation(zero_cardinality_dims, &h_zero_cardinality_dim_tuples_C_elnum);

  cutilSafeCall(cudaMalloc((void**)&(dp.d_zero_cardinality_dim_tuples_C), sizeof(size_t)*h_zero_cardinality_dim_tuples_C_elnum));
  cutilSafeCall(cudaMemcpy(dp.d_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuples_C,
                           sizeof(size_t)*h_zero_cardinality_dim_tuples_C_elnum, cudaMemcpyHostToDevice));

  return dp;
}



/*
// Multiply incoming vector pair by pair, sum elements with mod SG_SIZE and write the result in second input vector
__global__ void pairmulsum( double* pairmul, size_t pairmul_elnum, double* pairmul_result ){
size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

if ( thread_id < pairmul_elnum/2 ) { // odd number of elements?
pairmul_result[thread_id] = pairmul[(thread_id+1)*2-2] * pairmul[(thread_id+1)*2-1];
cuPrintf("pairmul_result[%d] = pairmul[%d] * pairmul[%d]\n", thread_id, (thread_id+1)*2-2, (thread_id+1)*2-1);
}
}
*/

// generates the full result tensor
__global__ void genFullResult(size_t* d_total_cards, size_t ndims,
                              size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_F,
                              double* d_A, double* d_B, double* d_F, size_t F_element_number){

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t d_inds_F;// = (size_t*) malloc(sizeof(size_t)*ndims);

  if (tid < F_element_number){

    // for each element of the full result tensor
    //      multiply corresponding elements of input tensors A and B

    //cuPrintf("tid %d \n",tid);

    size_t F_ind=0;
    size_t A_ind=0;
    size_t B_ind=0;
    for ( size_t dim=ndims-1; ; dim--){

      if ( tid / d_strides_F[dim] > 0 ){
        d_inds_F = tid / d_strides_F[dim];
        tid -= d_inds_F*d_strides_F[dim];
      }else{
        d_inds_F = 0;
      }

      F_ind += d_strides_F[dim] * d_inds_F;
      A_ind += d_strides_A[dim] * d_inds_F;
      B_ind += d_strides_B[dim] * d_inds_F;

      if(dim == 0) break;
    }

    d_F[F_ind] = d_A[A_ind] * d_B[B_ind];
    cuPrintf("tid %d: d_F[%d] = d_A[%d] * d_B[%d]\n", tid, F_ind, A_ind, B_ind);

  }
}


// for each element of d_C (tid corresponds to a single iteration)
//    loop over every zero cardinality dimension summing in tmp_sum
//    store tmp_sum as corresponding element of d_C
__global__ void contractFintoC(size_t* d_cards_F, size_t ndims,
                               size_t* d_strides_F, size_t* d_strides_C,
                               double* d_F, double* d_C,
                               size_t C_element_number,

                               size_t* d_zero_cardinality_dim_tuples_C,
                               size_t zero_cardinality_dim_tuple_size_C,
                               size_t d_zero_cardinality_dim_tuples_C_element_size) {

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  size_t* d_inds_C = (size_t*) malloc(sizeof(size_t)*ndims);


  if ( tid < C_element_number ){

    // calculate index for this tid
    size_t C_ind=0;
    for ( size_t dim=ndims-1; ; dim--){
      if ( tid / d_strides_C[dim] > 0 ){
        d_inds_C[dim] = tid / d_strides_C[dim];
        tid -= d_inds_C[dim]*d_strides_C[dim];
      }else{
        d_inds_C[dim] = 0;
      }

      C_ind += d_strides_C[dim] * d_inds_C[dim];

      if(dim == 0) break;
    }


    // calculate contraction value for this index of output tensor C
    double tmp_sum=0;

    // d_zero_cardinality_dim_tuples_C contains tuples of size zero_cardinality_dim_tuple_size_C
    // these correspond to the set of all possible indices over zero cardinality indices of tensor C

    for ( size_t iter=0;
          iter < d_zero_cardinality_dim_tuples_C_element_size;
          iter += zero_cardinality_dim_tuple_size_C ){

      size_t F_ind = 0;
      for ( size_t dim=0 ; dim<ndims; dim++){
        if ( d_cards_C[dim] == 0 ){
          F_ind += d_strides_F[dim] * d_zero_cardinality_dim_tuples_C_element_size[iter+dim];
        }else{
          F_ind += d_strides_F[dim] * d_inds_C[dim];
        }
      }

      tmp_sum += d_F[F_ind];
    }



    //d_F[F_ind] = d_A[A_ind] * d_B[B_ind];
    //cuPrintf("tid %d: d_F[%d] = d_A[%d] * d_B[%d]\n", tid, F_ind, A_ind, B_ind);



    // store this element of d_C
    d_C[C_ind] = tmp_sum;
  }
}


double get_element(ct* h_ct, size_t* global_index, char* str=""){
  std::cout << "get_element: " << str << " cur_ind ";
  size_t cur_ind=0;
  for (size_t dim=0; dim<h_ct->ndims; dim++){
    std::cout << global_index[dim] << " ";
    cur_ind += h_ct->strides[dim] * global_index[dim];
  }
  std::cout << " index " << cur_ind << " val " << h_ct->data[cur_ind]
            << std::endl;
  return h_ct->data[cur_ind];
}




void increment_cur_index(size_t ndims, size_t* h_full_cardinalities, size_t* global_index){
  for (size_t dim=0; dim<ndims; dim++){
    // if we have NOT reached limit of this dimension
    if( global_index[dim] != (h_full_cardinalities[dim]-1) ){
      // increment this dimension
      global_index[dim]++;
      break;
    }else{
      // we have reached limit of this dimension

      // if next dimension is at limit as well, skip this dimension, operation will take place in next dimension
      if( dim != (ndims-1) && global_index[dim+1] == (h_full_cardinalities[dim+1]-1)){
        //std::cout << "skip" << std::endl;
        continue;
      }else{

        // if this is the last dimension (and it is full) no increment is possible increment error
        //if (dim == ndims-1){
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

  ///////////////////////////////////////////////////////////////////////////////////////////


  // prepare host and device memory for tensors  ///////////////////////////////////////////

  ct h_A, h_B, h_C, h_F;
  prepareHostTensor(&h_A, m_A_data, m_A_card);
  prepareHostTensor(&h_B, m_B_data, m_B_card);
  // NULL initiates data with zero
  prepareHostTensorFromCpp(&h_F, NULL, h_full_cardinalities, ndims);
  size_t* tmp_arr = (size_t*) malloc(sizeof(size_t)*ndims);
  for ( size_t i=0; i<ndims; i++) tmp_arr[i] = (size_t) (((double*) mxGetData(m_C_card))[i]);
  prepareHostTensorFromCpp(&h_C, NULL, tmp_arr, ndims);

  dev_ptrs dp = prepareDeviceParameters(h_full_cardinalities, ndims, &h_A, &h_B, &h_C, &h_F);

  ///////////////////////////////////////////////////////////////////////////////////////////


  // run kernels //////////////////////////////////////////////////////////////////////////////


  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  cudaPrintfInit();

  std::cout << " Running kernels " << std::endl << std::endl;

  //pairmul<<<100,100>>>(d_pairmul, C_elnum*2, d_pairmul_result);

  // generate the full output
  genFullResult<<<1,100>>>(dp.d_full_cardinalities, ndims, dp.d_strides_A, dp.d_strides_B, dp.d_strides_F, dp.d_data_A, dp.d_data_B, dp.d_data_F, h_F.element_number);

  // test full result
  cutilSafeCall(cudaMemcpy(h_F.data, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
  print_ct("genFullResult", &h_F,true);

  // if no contraction is required, result is already stored in F, return that
  bool got_zeros=false;
  for ( size_t dim=0; dim<ndims; dim++){
    if ( h_C.cardinalities[dim] == 0 ){
      //std::cout << " GOT ZEROS found zero on h_C dimension " << dim << std::endl;
      got_zeros=true;
      break;
    }
  }

  if ( got_zeros ){
    // contract on required dimensions
    contractFintoC<<<1000,100>>>();
  }


  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaThreadSynchronize();
  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  cutilCheckError(cutDeleteTimer(timer));

  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();


  if ( got_zeros ){
    cutilSafeCall(cudaMemcpy(m_C, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
  }else{
    cutilSafeCall(cudaMemcpy(m_C, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
  }
  std::cout << "plhs elnum " << mxGetNumberOfElements(plhs[0]) << std::endl;
  std::cout << "C_elnum " << C_elnum << std::endl;

  cudaThreadExit();
  ///////////////////////////////////////////////////////////////////////////////////////////
}
