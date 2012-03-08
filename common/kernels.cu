/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */


#include "settings.h"

#include "cuPrintf.cu"



// generates pair-wise multiplication result
__global__ void hadamard_mul(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){

    d_C[tid] = pow(d_A[tid], to_power_A) * pow(d_B[tid], to_power_B);

    if( print ){
      double result = pow(d_A[tid], to_power_A) * pow(d_B[tid], to_power_B);
      cuPrintf("hadamard_mul result %f \n", result);
    }

  }
}

// generates pair-wise division result
__global__ void hadamard_div(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = pow(d_A[tid], to_power_A) / pow(d_B[tid], to_power_B);

    if( print ){
      double result = pow(d_A[tid], to_power_A) / pow(d_B[tid], to_power_B);
      cuPrintf("hadamard_div result %f \n", result);
    }

  }
}

// generates pair-wise summation result
__global__ void hadamard_sum(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = pow(d_A[tid], to_power_A) + pow(d_B[tid], to_power_B);

    if( print ){
      double result = pow(d_A[tid], to_power_A) + pow(d_B[tid], to_power_B);
      cuPrintf("hadamard_sum result %f \n", result);
    }

  }
}


// generates the full result tensor
__global__ void genFullResult(size_t* d_total_cards, size_t ndims,
                              size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_F,
                              double* d_A, double* d_B, double* d_F,
                              size_t F_element_number, size_t A_element_number, size_t B_element_number,
                              size_t use_multiplication,
                              bool print,
                              int to_power_A, int to_power_B){

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t d_inds_F;// = (size_t*) malloc(sizeof(size_t)*ndims);

  if (tid < F_element_number){

    // for each element of the full result tensor
    //      multiply corresponding elements of input tensors A and B

    if( print ){
      cuPrintf("tid %d element numbers F %d A %d B %d \n",tid,F_element_number, A_element_number, B_element_number);
    }

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


      if( print ){
        size_t tmp= d_strides_F[dim];
        cuPrintf("F_ind %d d_strides_F %d d_inds_F %d\n", F_ind, F_ind, tmp);
      }

      if(dim == 0) break;
    }


    if ( A_ind >= A_element_number ){
      if( print ){
        cuPrintf("A preventing index overflow index %d max %d\n",A_ind, A_element_number-1);
      }
      A_ind = A_element_number-1;
    }

    if ( B_ind >= B_element_number ){
      if( print ){
        cuPrintf("B preventing index overflow index %d max %d\n",B_ind, B_element_number-1);
      }
      B_ind = B_element_number-1;
    }


    if (use_multiplication == 1)
      d_F[F_ind] = pow(d_A[A_ind], to_power_A) * pow(d_B[B_ind], to_power_B);
    else
      d_F[F_ind] = pow(d_A[A_ind], to_power_A) / pow(d_B[B_ind], to_power_B);


    if( print ){
      double tmpval = 0;

      if (use_multiplication == 1)
        tmpval = pow(d_A[A_ind], to_power_A) * pow(d_B[B_ind], to_power_B);
      else
        tmpval = pow(d_A[A_ind], to_power_A) / pow(d_B[B_ind], to_power_B);

      double Aval = d_A[A_ind];
      double Bval = d_B[B_ind];

      cuPrintf("tidABC %d: d_F[%d] = d_A[%d] * d_B[%d] = %f op %f = %f \n", tid, F_ind, A_ind, B_ind, Aval, Bval, tmpval);
    }

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
                               size_t zero_cardinality_dim_tuples_C_element_number,
                               bool print) {

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

    if( print ){
      cuPrintf("cont: zero_cardinality_dim_tuples_C_element_number %d\n",zero_cardinality_dim_tuples_C_element_number);
      cuPrintf("cont: zero_cardinality_dim_tuple_size_C %d\n",zero_cardinality_dim_tuple_size_C);
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

          if( print ){
            size_t tmp = d_strides_F[dim] * d_zero_cardinality_dim_tuples_C[iter];
            size_t tmp1 = d_strides_F[dim];
            size_t tmp2 = d_zero_cardinality_dim_tuples_C[iter];
            cuPrintf("cont: F_ind val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
          }

          iter++;
        }else{

          F_ind += d_strides_F[dim] * d_inds_C[dim];

          if( print ){
            size_t tmp = d_strides_F[dim] * d_inds_C[dim];
            size_t tmp1 = d_strides_F[dim];
            size_t tmp2 = d_inds_C[dim];
            cuPrintf("cont: F_ind else val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
          }
        }
      }

      if( print ){
        double kek=d_F[F_ind];
        cuPrintf("cont: F_ind %d d_F[F_ind] %f\n", F_ind, kek);
      }
      tmp_sum += d_F[F_ind];
    }





    // store this element of d_C
    if( print ){
      cuPrintf("cont: store C_ind %d C_element_number %d value %f\n",C_ind, C_element_number, tmp_sum);
    }
    d_C[C_ind] = tmp_sum;
  }
}







// for each element of d_C (tid corresponds to a single iteration)
//    loop over every zero cardinality dimension summing in tmp_sum
//    store tmp_sum as corresponding element of d_C
__global__ void calculate_C_mops(size_t ndims,
                                 size_t operand_num,

                                 size_t** d_strides_operand_pointers, //

                                 size_t* d_strides_output,
                                 size_t* d_cards_F,

                                 size_t** d_cards_operand_pointers, //
                                 double** d_operand_pointers,       //

                                 double* d_output,
                                 //size_t* operand_element_numbers,
                                 size_t output_element_number,
                                 size_t use_multiplication,
                                 bool print
                                 //int* to_power_operands             // -
                                 ){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  size_t d_inds_output[20]; // 20 dimensions limit

  if ( tid < output_element_number ){

    ///// calculate output index for this tid
    size_t output_ind=0;
    for ( size_t dim=ndims-1; ; dim--){
      if (d_strides_output[dim] != 0){
        if ( tid / d_strides_output[dim] > 0 ){
          d_inds_output[dim] = tid / d_strides_output[dim];
          tid -= d_inds_output[dim]*d_strides_output[dim];
        }else{
          d_inds_output[dim] = 0;
        }
      }

      output_ind += d_strides_output[dim] * d_inds_output[dim];

      // size_t tmp= d_strides_C[dim];
      // size_t tmp1= d_inds_C[dim];
      // cuPrintf("dim %d C_ind %d d_strides_C %d d_inds_C %d\n",dim, C_ind, tmp, tmp1);

      if(dim == 0) break;
    }

    if ( print ){
      // for(size_t i=0; i<ndims; i++){
      //   size_t tmp=d_inds_output[i];
      //   cuPrintf("d_inds_output dim %d : %d \n", i, tmp);
      // }
      cuPrintf("OUTPUT IND %d\n",output_ind);
    }

    /////////////////////////////////////////////

    // new value for d_output[output_ind]
    double val=0;

    double tmp_vals[2] = {1, 1};
    size_t tmp_val_ind = 0;


    // for each operand
    for( size_t operand=0; operand<operand_num; operand++){

      ///// find operand index(s) for current output_ind

      size_t operand_ind = 0;
      size_t tmp_output_ind = output_ind;
      size_t tmp_digit;

      for ( size_t dim=ndims-1; ; dim--){
        if (d_strides_output[dim] != 0){
          if ( tmp_output_ind / d_strides_output[dim] > 0 ){
            tmp_digit = tmp_output_ind / d_strides_output[dim];
            tmp_output_ind -= tmp_digit*d_strides_output[dim];
          }else{
            tmp_digit = 0;
          }
        }else{
	  tmp_digit = 0;
	}

        operand_ind += d_strides_operand_pointers[operand][dim] * tmp_digit;

        if( print ){
	  size_t tmp = d_strides_operand_pointers[operand][dim];
	  cuPrintf("d_strides_operand_pointers[%d][%d] %d tmp_digit %d -> operand_ind %d\n",operand, dim, tmp, tmp_digit, operand_ind); 
        }

        if(dim == 0) break;
      }


      // if ( operand_ind >= operand_element_numbers[operand] ){
      //   operand_ind = operand_element_numbers[operand]-1;
      //   if( print ){
      //     cuPrintf("preventing operand index overflow index %d max %d\n",operand_ind, operand_element_numbers[operand]-1);
      //   }
      // }
      /////////////////////////////////////////////


      ///// increment val for this operand


      // cardinalities of contraction indices
      //size_t d_contraction_cards[20]; // 20 dimensions limit
      // number of indices to contract
      size_t contraction_index_num = 0;

      // contains contraction indices of the operand
      size_t d_contraction_ind_operand[20] = {0}; // 20 dimension limit 
      size_t dcio_i=0;
      for( size_t dim=0; dim<ndims; dim++){
        if ( d_strides_output[dim] == 0 && d_cards_operand_pointers[operand][dim] != 0 ){
          //d_contraction_cards[contraction_index_num] = d_cards_F[dim];
          contraction_index_num++;

	  // store which index of the operand we contract
	  d_contraction_ind_operand[dcio_i]=dim;
	  dcio_i++;
        }

	if(print){
	  size_t tmp=d_strides_output[dim];
	  size_t tmp1= d_cards_operand_pointers[operand][dim];
	  cuPrintf("operand %d d_strides_output[%d] %d d_cards_operand_pointers[%d][%d] %d contraction_index_num %d\n", operand, dim, tmp, operand, dim, tmp1, contraction_index_num);
	}
      }

      // if(print){
      // 	if ( blockIdx.x * blockDim.x + threadIdx.x == 0) {
      // 	  size_t elnum=1;
      // 	  for(int d=0; d<ndims; d++)
      // 	    if (d_cards_operand_pointers[operand][d] != 0)
      // 		elnum *= d_cards_operand_pointers[operand][d];

      // 	  cuPrintf("operand %d elnum %d\n" , operand, elnum);
      // 	  for (int i=0; i<elnum; i++){
      // 	    double tmp1=d_operand_pointers[operand][ i ];
      // 	    cuPrintf("operand %d d_operand_pointers[%d][%d] = %f \n",operand, operand, i, tmp1);
      // 	  }
      // 	}
      // }


      if (print){
      	// print d_output
      	if ( blockIdx.x * blockDim.x + threadIdx.x == 0){
      	  for(int i=0; i<60; i++){
      	    double tmp=d_output[i];
      	    cuPrintf("d_output[%d] = %f\n", i, tmp);
      	  }
      	}
      }


      // // calculate displacement on d_operands due to previous operands
      // size_t prev_operand_element_num=0;
      // for( size_t prev_operands=0; prev_operands<operand; prev_operands++){
      //   prev_operand_element_num += operand_element_numbers[prev_operands];
      // }

      // if V_output == V_operand
      //   operand and output indices are the same only multiply
      if( contraction_index_num == 0 ){

        //val *= pow(d_operand_pointers[opreand][ operand_ind ], to_power_operands[operand]);
	if (val == 0){
	  val = d_operand_pointers[operand][ operand_ind ];
	}else{
	  val *= d_operand_pointers[operand][ operand_ind ];
	}


	d_output[output_ind] += val;
	cuPrintf("d_output ASSIGN output_ind %d = val %d \n",output_ind, val);


	if(print && blockIdx.x * blockDim.x + threadIdx.x == 0){
	  cuPrintf("contraction_index_num=0 val %f d_operand_pointers[%d][%d] = %f ", val, operand, operand_ind);
	}
	
      }else{
        // operand and output indices are not the same, must multiply and contract

	double prev_val;
	if( val == 0 ){
	  prev_val = 1;
	}else{
	  prev_val = val;
	}

        // val += prev_val * operand[ base_index + stride_contraction_index ]

        // for each combination of the contraction indices, perform val += operation
        bool not_done=true;
        size_t d_contraction_ind[20] = {0}; // 20 dimensions limit

        do{

          // d_contraction_cards contains cardinalities of indices to be contracted on operand
          // careful of indices which have zero cardinality both in output and operand -> NOT REQUIRED ONLY CALCULATING STRIDES!


          // use this configuration of d_contraction_indices and increment val

          size_t contraction_stride = 0;
	  size_t i=0;
          for( size_t d=0; d<ndims; d++){
	    if ( d_strides_output[d] == 0 && d_cards_operand_pointers[operand][d] != 0 ){
	      contraction_stride += d_strides_operand_pointers[operand][d] * d_contraction_ind[i];
	      i++;
	      cuPrintf("contraction_stride += %d * %d = %d\n", d_strides_operand_pointers[operand][d] , d_contraction_ind[i], contraction_stride);
	    }
          }

          //val += prev_val * pow(d_operand_pointers[operand][operand_ind + contraction_stride ], to_power_operands[operand]);
          tmp_vals[tmp_val_ind] *= prev_val * d_operand_pointers[operand][operand_ind + contraction_stride ];

	  if( tmp_val_ind == 0 ) tmp_val_ind=1;
	  else tmp_val_ind=0;
	  

	  if ( print 
	       //&& blockIdx.x * blockDim.x + threadIdx.x == 0 
	       ){
	    double tmp = d_operand_pointers[operand][operand_ind + contraction_stride ];
	    cuPrintf("val increment output_ind %d = val %f prev_val %f d_operand_pointers[%d][%d +%d] %f\n",output_ind, val, prev_val, operand, operand_ind, contraction_stride, tmp);
	    for(int i=0; i<contraction_index_num; i++){
	      size_t tmp = d_contraction_ind[i];
	      cuPrintf("cin: operand %d cin[%i] = %d \n", operand, i, tmp);
	    }
	  }


          // increment d_contraction_ind
          for (size_t dim=0; dim<contraction_index_num; dim++){
            // if we have NOT reached limit of this dimension
            if( d_contraction_ind[dim] != (d_cards_operand_pointers[operand][d_contraction_ind_operand[dim]]-1) && d_cards_operand_pointers[operand][d_contraction_ind_operand[dim]] != 0 ){
              // increment this dimension
              d_contraction_ind[dim]++;
              break;
            }else{
              // we have reached limit of this dimension

              // if next dimension is at limit as well, skip this dimension, operation will take place in next dimension
              if( dim != (contraction_index_num-1) &&
                  (d_contraction_ind[dim+1] == (d_cards_operand_pointers[operand][d_contraction_ind_operand[dim+1]]-1) || d_cards_operand_pointers[operand][d_contraction_ind_operand[dim+1]] == 0 ) ){
                //std::cout << "skip" << std::endl;
                continue;
              }else{

                // if this is the last dimension (and it is full) no increment is possible increment error
                if (dim == contraction_index_num-1){
                  //h_ct->increment_error = 1;
                  not_done = false;
                  break;
                }

                // make this and all previous dimensions zero
                for (int dim_prev=dim; dim_prev>=0 ; dim_prev--){
                  d_contraction_ind[dim_prev] = 0;
                }
                // increment next dimension
                d_contraction_ind[dim+1]++;
                break;
              }
            }
          }
        }while( not_done );
      }
    }
    d_output[output_ind] += tmp_vals[0] + tmp_vals[1];
    cuPrintf("d_output ASSIGN output_ind %d = val %f \n",output_ind, d_output[output_ind]);
  }
}


__global__ void printData(double* data, size_t count, size_t id){
  cuPrintf("printData id %d", id);
  for(int i=0; i<6; i++){
    double tmp=data[i];
    cuPrintf("data[%d] = %e\n", i, tmp);
  }
}

