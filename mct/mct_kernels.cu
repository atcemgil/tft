/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */


#include "settings.h"

#if CUPRINTF == true
#include "cuPrintf.cu"
#endif


// generates pair-wise multiplication result
__global__ void hadamard_mul(double* d_A, double* d_B, double* d_C, size_t C_element_number){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = d_A[tid] * d_B[tid];
  }
}

// generates pair-wise division result
__global__ void hadamard_div(double* d_A, double* d_B, double* d_C, size_t C_element_number){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = d_A[tid] / d_B[tid];
  }
}


// generates the full result tensor
__global__ void genFullResult(size_t* d_total_cards, size_t ndims,
                              size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_F,
                              double* d_A, double* d_B, double* d_F, size_t F_element_number,
			      size_t use_multiplication){

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t d_inds_F;// = (size_t*) malloc(sizeof(size_t)*ndims);

  if (tid < F_element_number){

    // for each element of the full result tensor
    //      multiply corresponding elements of input tensors A and B
    
#if CUPRINTF == true
    cuPrintf("tid %d \n",tid);
#endif

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

    if (use_multiplication == 1)
      d_F[F_ind] = d_A[A_ind] * d_B[B_ind];
    else
      d_F[F_ind] = d_A[A_ind] / d_B[B_ind];

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

#if CUPRINTF == true
      cuPrintf("zero_cardinality_dim_tuples_C_element_number %d\n",zero_cardinality_dim_tuples_C_element_number);
      cuPrintf("zero_cardinality_dim_tuple_size_C %d\n",zero_cardinality_dim_tuple_size_C);
#endif

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
#if CUPRINTF == true
      cuPrintf("F_ind %d d_F[F_ind] %d\n", F_ind, kek);
#endif
      tmp_sum += d_F[F_ind];
    }





    // store this element of d_C
#if CUPRINTF == true
      cuPrintf("C_ind %d C_element_number %d\n",C_ind, C_element_number);
#endif
    d_C[C_ind] = tmp_sum;
  }
}
