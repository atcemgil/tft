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

#define CUPRINTF false
#define COUT false
#define PRINT_CT false

#define NUM_BLOCKS 53000
#define THREADS_FOR_BLOCK 512

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




void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims, char* txt=NULL){
  h_ct->ndims = ndims;
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * h_ct->ndims );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * h_ct->ndims );

  size_t elnum=1;
  // assign cardinalities for the tensor objects and init cur_ind values
  size_t cum_sum=1;
  for (size_t i=0; i<h_ct->ndims; i++){
    if ( tensor_card[i] != 0 ){
      elnum *= tensor_card[i];
    }

    h_ct->cardinalities[i] = tensor_card[i];
    // std::cout << "TC dim "<< i << " cardinality assignment: "
    //           << h_ct->cardinalities[i] << " <- " << tensor_card[i] << std::endl;

    if ( h_ct->cardinalities[i] == 0){
      h_ct->strides[i]=0;
    }else{
      h_ct->strides[i]=cum_sum;
      cum_sum *= h_ct->cardinalities[i];
    }
  }

  // assign h_ct host data
  if ( txt != NULL && COUT == true){
    std::cout << txt << std::endl;
  }
  if ( COUT ) std::cout << "prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;

  if (data == NULL){
    //h_ct->data = (double*)calloc( h_ct->element_number, sizeof(double) );
    h_ct->data = (double*) malloc(h_ct->mem_size);
    for (size_t i=0; i<h_ct->element_number; i++ ) h_ct->data[i]=(double)0;
  }else{
    h_ct->data = (double*)malloc( h_ct->mem_size );
    memcpy(h_ct->data, data, h_ct->mem_size);
  }

  if ( PRINT_CT ) print_ct("prepareDeviceTensor h_ct",h_ct,true);

}

void prepareHostTensor(ct* h_ct, const mxArray* m_data, const mxArray* tensor_card, char* txt=NULL){
  h_ct->ndims = mxGetNumberOfElements(tensor_card); // assumes both objects of same size
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );

  // assign cardinalities for the tensor objects and init cur_ind values
  size_t cum_sum=1;
  for (size_t i=0; i<h_ct->ndims; i++){
    h_ct->cardinalities[i] = ((double *)mxGetData(tensor_card))[i];
    //std::cout << "TC dim "<< i << " cardinality assignment: "
    //          << h_ct->cardinalities[i]
    //          << " <- " << ((double *)mxGetData(tensor_card))[i] << std::endl;

    if ( h_ct->cardinalities[i] == 0){
      h_ct->strides[i]=0;
    }else{
      h_ct->strides[i]=cum_sum;
      cum_sum *= h_ct->cardinalities[i];
    }
  }

  // assign h_ct host data
  size_t elnum = (size_t) mxGetNumberOfElements(m_data);
  if ( txt != NULL && COUT ){
    std::cout << txt << std::endl;
  }
  if ( COUT ) std::cout << "prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;

  h_ct->data = (double*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (double*)mxGetData(m_data), h_ct->mem_size);

  if ( PRINT_CT ) print_ct("prepareDeviceTensor h_ct",h_ct,true);
}


// Recursive function which generates all permutations of a given list
void gen_range_permutation_helper(std::vector<size_t> iter_dims, std::vector<size_t> cur_perm, std::vector<size_t>* acc){
  if ( iter_dims.size() == 0 ){
    if (COUT){
      std::cout << "final cur_perm" <<  std::endl;
      for ( size_t j=0; j<cur_perm.size(); j++){
	std::cout << cur_perm.at(j) << std::endl;
      }
    }
    acc->insert(acc->end(), cur_perm.begin(), cur_perm.end());
  }else{
    // pick one dimension from iter_dims and iterate for each element in its cardinality
    size_t one_dim = iter_dims.front();
    // remove that dim for the remaining recursions
    iter_dims.erase(iter_dims.begin());

    for ( size_t i=0; i<one_dim ; i++){
      std::vector<size_t> tmp_vec (cur_perm.begin(), cur_perm.end());
      tmp_vec.push_back(i);
      if ( COUT ){
	std::cout << " tmp_vec " << std::endl;
	for ( size_t j=0; j<tmp_vec.size(); j++){
	  std::cout << tmp_vec.at(j) << std::endl;
	}
      }
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

  if ( COUT ){
    std::cout << "gen_range_permutation \n acc:" << std::endl;
    for ( size_t i=0; i<acc.size(); i++){
      std::cout << acc.at(i) << std::endl;
    }
    std::cout << "elnum " << *elnum << std::endl;
  }

  return acc_array;
}


dev_ptrs prepareDeviceParameters(size_t* h_full_cardinalities, size_t ndims, ct* h_A, ct* h_B, ct* h_C, ct* h_F,
                                 size_t zero_cardinality_dims_elnum, size_t h_zero_cardinality_dim_tuples_C_element_number, size_t* h_zero_cardinality_dim_tuples_C){
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


  cutilSafeCall(cudaMalloc((void**)&(dp.d_zero_cardinality_dim_tuples_C),
                           sizeof(size_t)*dp.zero_cardinality_dim_tuples_C_element_number));
  cutilSafeCall(cudaMemcpy(dp.d_zero_cardinality_dim_tuples_C, h_zero_cardinality_dim_tuples_C,
                           sizeof(size_t)*dp.zero_cardinality_dim_tuples_C_element_number, cudaMemcpyHostToDevice));

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

    cuPrintf("tid %d \n",tid);

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

      //size_t tmp= d_strides_F[dim];
      //cuPrintf("F_ind %d d_strides_F %d d_inds_F %d\n", F_ind, F_ind, tmp);


      if(dim == 0) break;
    }

    d_F[F_ind] = d_A[A_ind] * d_B[B_ind];
    //cuPrintf("tid %d: d_F[%d] = d_A[%d] * d_B[%d]\n", tid, F_ind, A_ind, B_ind);

  }
}


// for each element of d_C (tid corresponds to a single iteration)
//    loop over every zero cardinality dimension summing in tmp_sum
//    store tmp_sum as corresponding element of d_C
__global__ void contractFintoC(size_t ndims,
                               size_t* d_strides_F, size_t* d_strides_C,
                               double* d_F, double* d_C,
                               size_t C_element_number,
                               size_t* d_zero_cardinality_dim_tuples_C,
                               size_t zero_cardinality_dim_tuple_size_C,
                               size_t zero_cardinality_dim_tuples_C_element_number) {

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  size_t d_inds_C[20]; // 20 dimensions limit


  if ( tid < C_element_number ){
    // calculate index for this tid
    size_t C_ind=0;
    for ( size_t dim=ndims-1; ; dim--){
      if (d_strides_C[dim] != 0){
        if ( tid / d_strides_C[dim] > 0 ){
          d_inds_C[dim] = tid / d_strides_C[dim];
          tid -= d_inds_C[dim]*d_strides_C[dim];
        }else{
          d_inds_C[dim] = 0;
        }
      }


      C_ind += d_strides_C[dim] * d_inds_C[dim];

      // size_t tmp= d_strides_C[dim];
      // size_t tmp1= d_inds_C[dim];
      // cuPrintf("dim %d C_ind %d d_strides_C %d d_inds_C %d\n",dim, C_ind, tmp, tmp1);


      if(dim == 0) break;
    }

    // for(size_t i=0; i<ndims; i++){
    //  size_t tmp=d_inds_C[i];
    //  cuPrintf("d_inds_C %d\n",tmp);
    //   }


    // calculate contraction value for this index of output tensor C
    double tmp_sum=0;

    // d_zero_cardinality_dim_tuples_C contains tuples of size zero_cardinality_dim_tuple_size_C
    // these correspond to the set of all possible indices over zero cardinality indices of tensor C

    if (CUPRINTF){
      cuPrintf("zero_cardinality_dim_tuples_C_element_number %d\n",zero_cardinality_dim_tuples_C_element_number);
      cuPrintf("zero_cardinality_dim_tuple_size_C %d\n",zero_cardinality_dim_tuple_size_C);
    }
    for ( size_t iter=0;
          iter < zero_cardinality_dim_tuples_C_element_number; ){

      size_t F_ind = 0;
      for ( size_t dim=0 ; dim<ndims; dim++){
        if ( d_strides_F[dim] == 0 ){
          continue;
        }

        if ( d_strides_C[dim] == 0 ){
          F_ind += d_strides_F[dim] * d_zero_cardinality_dim_tuples_C[iter];
          //cuPrintf();

          // size_t tmp = d_strides_F[dim] * d_zero_cardinality_dim_tuples_C[iter];
          // size_t tmp1 = d_strides_F[dim];
          // size_t tmp2 = d_zero_cardinality_dim_tuples_C[iter];
          //cuPrintf("F_ind val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );

          iter++;
        }else{
          F_ind += d_strides_F[dim] * d_inds_C[dim];
          // size_t tmp = d_strides_F[dim] * d_inds_C[dim];
          // size_t tmp1 = d_strides_F[dim];
          // size_t tmp2 = d_inds_C[dim];
          //cuPrintf("F_ind else val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
        }
      }

      double kek=d_F[F_ind];
      if (CUPRINTF){
	cuPrintf("F_ind %d d_F[F_ind] %d\n", F_ind, kek);
      }
      tmp_sum += d_F[F_ind];
    }





    // store this element of d_C
    if (CUPRINTF){
      cuPrintf("C_ind %d C_element_number %d\n",C_ind, C_element_number);
    }
    d_C[C_ind] = tmp_sum;
  }
}


double get_element(ct* h_ct, size_t* global_index, char* str=""){
  if ( COUT ) std::cout << "get_element: " << str << " cur_ind ";
  size_t cur_ind=0;
  for (size_t dim=0; dim<h_ct->ndims; dim++){
    if ( COUT ) std::cout << global_index[dim] << " ";
    if(h_ct->strides[dim] != 0 )
      cur_ind += h_ct->strides[dim] * global_index[dim];
  }
  if ( COUT ) std::cout << " index " << cur_ind << " val " << h_ct->data[cur_ind]
			<< std::endl;
  return h_ct->data[cur_ind];
}


void set_element(ct* h_ct, size_t* global_index, double val, char* str=""){
  if ( COUT ) std::cout << "set_element: " << str << " cur_ind ";
  size_t cur_ind=0;
  for (size_t dim=0; dim<h_ct->ndims; dim++){
    if ( COUT ) std::cout << global_index[dim] << " ";
    if(h_ct->strides[dim] != 0 )
      cur_ind += h_ct->strides[dim] * global_index[dim];
  }
  if ( COUT ) std::cout << " index " << cur_ind << " prev val " << h_ct->data[cur_ind]
			<< " new val " << val
			<< std::endl;
  h_ct->data[cur_ind] = val;
}




void increment_cur_index(size_t ndims, size_t* h_full_cardinalities, size_t* global_index){
  for (size_t dim=0; dim<ndims; dim++){
    // if we have NOT reached limit of this dimension
    if( global_index[dim] != (h_full_cardinalities[dim]-1) && h_full_cardinalities[dim] != 0 ){
      // increment this dimension
      global_index[dim]++;
      break;
    }else{
      // we have reached limit of this dimension

      // if next dimension is at limit as well, skip this dimension, operation will take place in next dimension
      if( dim != (ndims-1) &&
          (global_index[dim+1] == (h_full_cardinalities[dim+1]-1) || h_full_cardinalities[dim+1] == 0 ) ){
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
  if ( COUT ) std::cout << "mex: found " << nrhs << " number of arguments " << std::endl;
  if (nrhs!=6){
    std::cout << "mex: cudatensor3 requires 5 arguments. A, dimensions of A, B, dimensions of B, dimensions of C, optype " << std::endl;
    return;
  }

  const mxArray* m_A_data = prhs[0];
  const mxArray* m_A_card = prhs[1];

  const mxArray* m_B_data = prhs[2];
  const mxArray* m_B_card = prhs[3];

  const mxArray* m_C_card = prhs[4];

  const mxArray* optype = prhs[5];

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


  size_t h_zero_cardinality_dim_tuple_size_C = zero_cardinality_dims.size();


  size_t h_zero_cardinality_dim_tuples_C_element_number; // set by gen_range_permutation

  size_t* h_zero_cardinality_dim_tuples_C =
    gen_range_permutation(zero_cardinality_dims,
                          &(h_zero_cardinality_dim_tuples_C_element_number));






  if ( ((double*)mxGetData(optype))[0] == 0 ) {

    // prepare device memory for tensors  /////////////////////////////////////////////////////

    dev_ptrs dp = prepareDeviceParameters(h_full_cardinalities, ndims, &h_A, &h_B, &h_C, &h_F,
                                          h_zero_cardinality_dim_tuple_size_C,
                                          h_zero_cardinality_dim_tuples_C_element_number,
                                          h_zero_cardinality_dim_tuples_C);


    ///////////////////////////////////////////////////////////////////////////////////////////


    // run kernels //////////////////////////////////////////////////////////////////////////////


    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    if (CUPRINTF) cudaPrintfInit();

    if ( COUT ) std::cout << " Running kernels " << std::endl << std::endl;

    //pairmul<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(d_pairmul, C_elnum*2, d_pairmul_result);

    // generate the full output
    genFullResult<<<NUM_BLOCKS,THREADS_FOR_BLOCK>>>(dp.d_full_cardinalities, ndims, dp.d_strides_A, dp.d_strides_B, dp.d_strides_F, dp.d_data_A, dp.d_data_B, dp.d_data_F, h_F.element_number);

    // test full result
    cutilSafeCall(cudaMemcpy(h_F.data, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
    if ( PRINT_CT ) print_ct("genFullResult", &h_F,true);

    // if no contraction is required, result is already stored in F, return that
    bool got_zeros = false;
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


    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");
    cudaThreadSynchronize();
    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    cutilCheckError(cutDeleteTimer(timer));

    if (CUPRINTF){
      cudaPrintfDisplay(stdout, true);
      cudaPrintfEnd();
    }


    if ( got_zeros ){
      cutilSafeCall(cudaMemcpy(m_C, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
      cutilSafeCall(cudaMemcpy(h_C.data, dp.d_data_C, h_C.mem_size, cudaMemcpyDeviceToHost));
      if ( PRINT_CT ) print_ct("result on C side (after)", &h_C,true);
    }else{
      cutilSafeCall(cudaMemcpy(m_C, dp.d_data_F, h_F.mem_size, cudaMemcpyDeviceToHost));
    }

    if ( COUT ) {
      std::cout << "plhs elnum " << mxGetNumberOfElements(plhs[0]) << std::endl;
      std::cout << "C_elnum " << C_elnum << std::endl;
    }

    cudaThreadExit();
    ///////////////////////////////////////////////////////////////////////////////////////////




  }else{  // operate on CPU



    // generate full tensor
    size_t global_index[ndims];
    for ( size_t i=0; i<ndims; i++)
      global_index[i] = 0;

    for ( size_t F_ind = 0; F_ind < h_F.element_number; F_ind++){
      set_element(&h_F, global_index,
                  get_element(&h_A,global_index) * get_element(&h_B, global_index) );

      increment_cur_index(ndims, h_full_cardinalities, global_index);
    }

    if ( PRINT_CT ) print_ct("C: generate full tensor", &h_F,true);



    // if no contraction is required, result is already stored in F, return that
    bool got_zeros = false;
    for ( size_t dim=0; dim<ndims; dim++){
      if ( h_C.cardinalities[dim] == 0 && h_F.cardinalities[dim] != 0){
	if ( COUT ) std::cout << " GOT ZEROS found zero on h_C dimension " << dim << std::endl;
        got_zeros = true;
        break;
      }
    }

    if ( got_zeros ){


      // contract on necessary dimensions
      size_t C_index[ndims];
      for ( size_t i=0; i<ndims; i++)
        C_index[i] = 0;

      for ( size_t C_ind = 0; C_ind < h_C.element_number; C_ind++){
	if ( COUT ) std::cout << "C_ind " <<  C_ind << std::endl;

        // calculate contraction value for this index of output tensor C

        // d_zero_cardinality_dim_tuples_C contains tuples of size zero_cardinality_dim_tuple_size_C
        // these correspond to the set of all possible indices over zero cardinality indices of tensor C

	if ( COUT ) std:: cout << "h_zero_cardinality_dim_tuples_C_element_number "
			       << h_zero_cardinality_dim_tuples_C_element_number
			       << std::endl;

        for ( size_t iter=0;
              iter < h_zero_cardinality_dim_tuples_C_element_number; ){
          //std::cout << " iter  " << iter << std::endl;

          size_t F_ind = 0;
          for ( size_t dim=0 ; dim<ndims; dim++){
            //std::cout << " iter:dim " << dim << std::endl;
            //std::cout << " h_F.strides[dim] " << h_F.strides[dim] << std::endl;
            if ( h_F.strides[dim] == 0 ){
              continue;
            }

            if ( h_C.strides[dim] == 0 ){
              F_ind += h_F.strides[dim] * h_zero_cardinality_dim_tuples_C[iter];

              // std::cout << "F_ind val " << h_F.strides[dim] * h_zero_cardinality_dim_tuples_C[iter]
              //              << " h_F.strides[dim] " << h_F.strides[dim]
              //              << " h_zero_cardinality_dim_tuples_C[iter] " << h_zero_cardinality_dim_tuples_C[iter]
              //              << " stride " << h_F.strides[dim]
              //              << " inds " << h_zero_cardinality_dim_tuples_C[iter]
              //              << std::endl;

              iter++;
            }else{
              F_ind += h_F.strides[dim] * C_index[dim];
              // size_t tmp = h_F.strides[dim] * C_index[dim];
              // size_t tmp1 = h_F.strides[dim];
              // size_t tmp2 = C_index[dim];
              // std::cout << "F_ind else val " << h_F.strides[dim] * C_index[dim]
              //              << " h_F.strides[dim] " << h_F.strides[dim]
              //              << " C_index[dim] " << C_index[dim]
              //              << " stride " << h_F.strides[dim]
              //              << " inds " << C_index[dim]
              //              << std::endl;
            }

            //std::cout << " F_ind " <<  F_ind << std::endl;
          }

          //std::cout << "F_ind " << F_ind << " d_F[F_ind] " << h_F.data[F_ind] << std::endl;
          h_C.data[C_ind] += h_F.data[F_ind];

        }

        // store this element of d_C
	if ( COUT ) std::cout << "C_ind " << C_ind
			      << std::endl;

	get_element(&h_C, C_index);
        increment_cur_index(ndims, h_C.cardinalities, C_index);
	
      }



      memcpy(m_C, h_C.data, h_C.mem_size);
      if ( COUT ) {
	std::cout << "plhs elnum " << mxGetNumberOfElements(plhs[0]) << std::endl;
	std::cout << "C_elnum " << C_elnum << std::endl;
      }
    }else{
      memcpy(m_C, h_F.data, h_F.mem_size);
    }
    if ( PRINT_CT ) print_ct("C: contraction result", &h_C, true);
  }

}
