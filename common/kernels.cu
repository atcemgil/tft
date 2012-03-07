/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */


#include "settings.h"

#if CUPRINTF == true
#include "cuPrintf.cu"
#endif


// generates pair-wise multiplication result
__global__ void hadamard_mul(double* d_A, double* d_B, double* d_C, size_t C_element_number, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){

    d_C[tid] = pow(d_A[tid], to_power_A) * pow(d_B[tid], to_power_B);

#if CUPRINTF == true
    double result = pow(d_A[tid], to_power_A) * pow(d_B[tid], to_power_B);
    cuPrintf("hadamard_mul result %f \n", result);
#endif

  }
}

// generates pair-wise division result
__global__ void hadamard_div(double* d_A, double* d_B, double* d_C, size_t C_element_number, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = pow(d_A[tid], to_power_A) / pow(d_B[tid], to_power_B);

#if CUPRINTF == true
    double result = pow(d_A[tid], to_power_A) / pow(d_B[tid], to_power_B);
    cuPrintf("hadamard_div result %f \n", result);
#endif

  }
}

// generates pair-wise summation result
__global__ void hadamard_sum(double* d_A, double* d_B, double* d_C, size_t C_element_number, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = pow(d_A[tid], to_power_A) + pow(d_B[tid], to_power_B);

#if CUPRINTF == true
    double result = pow(d_A[tid], to_power_A) + pow(d_B[tid], to_power_B);
    cuPrintf("hadamard_sum result %f \n", result);
#endif

  }
}


// generates the full result tensor
__global__ void genFullResult(size_t* d_total_cards, size_t ndims,
                              size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_F,
                              double* d_A, double* d_B, double* d_F, 
			      size_t F_element_number, size_t A_element_number, size_t B_element_number,
			      size_t use_multiplication,
			      int to_power_A, int to_power_B){

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t d_inds_F;// = (size_t*) malloc(sizeof(size_t)*ndims);

  if (tid < F_element_number){

    // for each element of the full result tensor
    //      multiply corresponding elements of input tensors A and B
    
#if CUPRINTF == true
    cuPrintf("tid %d element numbers F %d A %d B %d \n",tid,F_element_number, A_element_number, B_element_number);
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


#if CUPRINTF == true 
      size_t tmp= d_strides_F[dim];
      cuPrintf("F_ind %d d_strides_F %d d_inds_F %d\n", F_ind, F_ind, tmp);
#endif

      if(dim == 0) break;
    }


    if ( A_ind >= A_element_number ){
#if CUPRINTF == true 
      cuPrintf("A preventing index overflow index %d max %d\n",A_ind, A_element_number-1);
#endif
      A_ind = A_element_number-1;
    }

    if ( B_ind >= B_element_number ){
#if CUPRINTF == true 
      cuPrintf("B preventing index overflow index %d max %d\n",B_ind, B_element_number-1);
#endif
      B_ind = B_element_number-1;
    }


    if (use_multiplication == 1)
      d_F[F_ind] = pow(d_A[A_ind], to_power_A) * pow(d_B[B_ind], to_power_B);
    else
      d_F[F_ind] = pow(d_A[A_ind], to_power_A) / pow(d_B[B_ind], to_power_B);


#if CUPRINTF == true 
    double tmpval = 0;

    if (use_multiplication == 1)
      tmpval = pow(d_A[A_ind], to_power_A) * pow(d_B[B_ind], to_power_B);
    else
      tmpval = pow(d_A[A_ind], to_power_A) / pow(d_B[B_ind], to_power_B);

    double Aval = d_A[A_ind];
    double Bval = d_B[B_ind];

    cuPrintf("tidABC %d: d_F[%d] = d_A[%d] * d_B[%d] = %f op %f = %f \n", tid, F_ind, A_ind, B_ind, Aval, Bval, tmpval);
#endif

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
      cuPrintf("cont: zero_cardinality_dim_tuples_C_element_number %d\n",zero_cardinality_dim_tuples_C_element_number);
      cuPrintf("cont: zero_cardinality_dim_tuple_size_C %d\n",zero_cardinality_dim_tuple_size_C);
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

#if CUPRINTF == true
           size_t tmp = d_strides_F[dim] * d_zero_cardinality_dim_tuples_C[iter];
           size_t tmp1 = d_strides_F[dim];
           size_t tmp2 = d_zero_cardinality_dim_tuples_C[iter];
          cuPrintf("cont: F_ind val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
#endif

          iter++;
        }else{

          F_ind += d_strides_F[dim] * d_inds_C[dim];

#if CUPRINTF == true
           size_t tmp = d_strides_F[dim] * d_inds_C[dim];
           size_t tmp1 = d_strides_F[dim];
           size_t tmp2 = d_inds_C[dim];
          cuPrintf("cont: F_ind else val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
#endif
        }
      }

#if CUPRINTF == true
      double kek=d_F[F_ind];
      cuPrintf("cont: F_ind %d d_F[F_ind] %f\n", F_ind, kek);
#endif
      tmp_sum += d_F[F_ind];
    }





    // store this element of d_C
#if CUPRINTF == true
    cuPrintf("cont: store C_ind %d C_element_number %d value %f\n",C_ind, C_element_number, tmp_sum);
#endif
    d_C[C_ind] = tmp_sum;
  }
}
