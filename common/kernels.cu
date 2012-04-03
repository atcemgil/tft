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

    // if( print ){
    //   double result = pow(d_A[tid], to_power_A) * pow(d_B[tid], to_power_B);
    //   cuPrintf("hadamard_mul result %f \n", result);
    // }

  }
}

// generates pair-wise division result
__global__ void hadamard_div(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = pow(d_A[tid], to_power_A) / pow(d_B[tid], to_power_B);

    // if( print ){
    //   double result = pow(d_A[tid], to_power_A) / pow(d_B[tid], to_power_B);
    //   cuPrintf("hadamard_div result %f \n", result);
    // }

  }
}

// generates pair-wise summation result
__global__ void hadamard_sum(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A, int to_power_B){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < C_element_number){
    d_C[tid] = pow(d_A[tid], to_power_A) + pow(d_B[tid], to_power_B);

    // if( print ){
    //   double result = pow(d_A[tid], to_power_A) + pow(d_B[tid], to_power_B);
    //   cuPrintf("hadamard_sum result %f \n", result);
    // }

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

    // if( print ){
    //   cuPrintf("tid %d element numbers F %d A %d B %d \n",tid,F_element_number, A_element_number, B_element_number);
    // }

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


      // if( print ){
      //   size_t tmp= d_strides_F[dim];
      //   cuPrintf("F_ind %d d_strides_F %d d_inds_F %d\n", F_ind, F_ind, tmp);
      // }

      if(dim == 0) break;
    }


    if ( A_ind >= A_element_number ){
      // if( print ){
      //   cuPrintf("A preventing index overflow index %d max %d\n",A_ind, A_element_number-1);
      // }
      A_ind = A_element_number-1;
    }

    if ( B_ind >= B_element_number ){
      // if( print ){
      //   cuPrintf("B preventing index overflow index %d max %d\n",B_ind, B_element_number-1);
      // }
      B_ind = B_element_number-1;
    }


    if (use_multiplication == 1)
      d_F[F_ind] = pow(d_A[A_ind], to_power_A) * pow(d_B[B_ind], to_power_B);
    else
      d_F[F_ind] = pow(d_A[A_ind], to_power_A) / pow(d_B[B_ind], to_power_B);


    // if( print ){
    //   double tmpval = 0;

    //   if (use_multiplication == 1)
    //     tmpval = pow(d_A[A_ind], to_power_A) * pow(d_B[B_ind], to_power_B);
    //   else
    //     tmpval = pow(d_A[A_ind], to_power_A) / pow(d_B[B_ind], to_power_B);

    //   double Aval = d_A[A_ind];
    //   double Bval = d_B[B_ind];

    //   cuPrintf("tidABC %d: d_F[%d] = d_A[%d] * d_B[%d] = %f op %f = %f \n", tid, F_ind, A_ind, B_ind, Aval, Bval, tmpval);
    // }

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


  if ( tid < C_element_number ){
    size_t d_inds_C[20]; // 20 dimensions limit

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
      // //cuPrintf("dim %d C_ind %d d_strides_C %d d_inds_C %d\n",dim, C_ind, tmp, tmp1);


      if(dim == 0) break;
    }

    // for(size_t i=0; i<ndims; i++){
    //  size_t tmp=d_inds_C[i];
    //  //cuPrintf("d_inds_C %d\n",tmp);
    //   }


    // calculate contraction value for this index of output tensor C
    double tmp_sum=0;

    // d_zero_cardinality_dim_tuples_C contains tuples of size zero_cardinality_dim_tuple_size_C
    // these correspond to the set of all possible indices over zero cardinality indices of tensor C

    // if( print ){
    //   //cuPrintf("cont: zero_cardinality_dim_tuples_C_element_number %d\n",zero_cardinality_dim_tuples_C_element_number);
    //   //cuPrintf("cont: zero_cardinality_dim_tuple_size_C %d\n",zero_cardinality_dim_tuple_size_C);
    // }

    for ( size_t iter=0;
          iter < zero_cardinality_dim_tuples_C_element_number; ){

      size_t F_ind = 0;
      for ( size_t dim=0 ; dim<ndims; dim++){
        if ( d_strides_F[dim] == 0 ){
          continue;
        }

        if ( d_strides_C[dim] == 0 ){
          F_ind += d_strides_F[dim] * d_zero_cardinality_dim_tuples_C[iter];

          // if( print ){
          //   size_t tmp = d_strides_F[dim] * d_zero_cardinality_dim_tuples_C[iter];
          //   size_t tmp1 = d_strides_F[dim];
          //   size_t tmp2 = d_zero_cardinality_dim_tuples_C[iter];
          //   cuPrintf("cont: F_ind val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
          // }

          iter++;
        }else{

          F_ind += d_strides_F[dim] * d_inds_C[dim];

          // if( print ){
          //   size_t tmp = d_strides_F[dim] * d_inds_C[dim];
          //   size_t tmp1 = d_strides_F[dim];
          //   size_t tmp2 = d_inds_C[dim];
          //   cuPrintf("cont: F_ind else val %d, stride %d, inds %d\n",tmp, tmp1, tmp2 );
          // }
        }
      }

      // if( print ){
      //   double kek=d_F[F_ind];
      //   cuPrintf("cont: F_ind %d d_F[F_ind] %f\n", F_ind, kek);
      // }
      tmp_sum += d_F[F_ind];
    }





    // store this element of d_C
    if( print ){
      //cuPrintf("cont: store C_ind %d C_element_number %d value %f\n",C_ind, C_element_number, tmp_sum);
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
                                 bool print,
                                 size_t opnum,
                                 int* to_power_operands
                                 ){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;


  if ( tid < output_element_number ){
    ndims=3;
    operand_num=2;

    
    size_t d_inds_output[7] = {0}; // 20 dimensions limit

    __shared__ int d_strides_operand_pointers_shr[2][3];
    __shared__ int d_strides_output_shr[3];
    __shared__ int d_cards_F_shr[3];
    //__shared__ int d_cards_operand_pointers_shr[2][3];
    //__shared__ int to_power_operands_shr[2];

    for( int o=0; o<operand_num; o++ ){
      for ( int d=0; d<ndims; d++){
	d_strides_operand_pointers_shr[o][d] = d_strides_operand_pointers[o][d];
	//d_cards_operand_pointers_shr[o][d] = d_cards_operand_pointers_shr[o][d];
      }
    }

    for( int d=0; d<ndims; d++){
      d_strides_output_shr[d] = d_strides_output[d];
      d_cards_F_shr[d] = d_cards_F[d];
    }

    // cuPrintf("selam ndims %d opnum %d\n",ndims, opnum);

    // for( size_t i=0; i<ndims; i++ ){
    //   cuPrintf("calculate_C_mops d_strides_output[%d] = %d\n", i, d_strides_output[i]);
    // }


    size_t global_index[7] = {0}; // 20 dimensions limit
    size_t contract_dims[7] = {0}; // 20 dimensions limit
    size_t contract_dim_num=0;
    ///// calculate output index for this tid
    size_t output_ind=0;
    for ( int dim=ndims-1; ; dim--){
      if (d_strides_output_shr[dim] != 0){
        if ( tid / d_strides_output_shr[dim] > 0 ){
          d_inds_output[dim] = tid / d_strides_output_shr[dim];
          tid -= d_inds_output[dim]*d_strides_output_shr[dim];
        }else{
          d_inds_output[dim] = 0;
        }


        // this index is fixed, it will not iterated over
        global_index[dim] = d_inds_output[dim];

      }else{
	// global_index[dim] = 0 by init


	// if any of operands have data in this dimension we need to contract
	for( int operand=0; operand<operand_num; operand++){

	  if ( d_strides_operand_pointers_shr[operand][dim] != 0 ){
	    //cuPrintf(" nedir: operand %d operand stride on dim %d : %d, opnum %d\n", operand, dim, d_strides_operand_pointers_shr[operand][dim], opnum);
	    //this dimension is not fixed, will iterate over for contraction
	    contract_dims[contract_dim_num] = dim;
	    contract_dim_num++;
	    
	    break; // done for this dim
	  }
	}
	
      }

      output_ind += d_strides_output_shr[dim] * d_inds_output[dim];

      if(dim == 0) break;
    }

    // if ( print ){
    //   for(size_t i=0; i<ndims; i++){
    //     size_t tmp=d_inds_output[i];
    //     cuPrintf("d_inds_output dim %d : %d \n", i, tmp);
    //   }
    //   cuPrintf("OUTPUT IND %d\n",output_ind);

    //   for(size_t i=0; i<contract_dim_num; i++){      
    // 	cuPrintf("contract_dims %d opnum %d\n",contract_dims[i], opnum);
    //   }
    // }

    d_output[output_ind]=0;

    /////////////////////////////////////////////


    // for all F indices find one value from each operand, multiply them and increment value in output index with it

    bool not_done = true;


    //size_t counter=0;


    do{
      // cuPrintf("counter %d\n", counter);
      // cuPrintf("global_index %d %d %d %d %d %d %d\n", 
      // 	       global_index[0], 
      // 	       global_index[1], 
      // 	       global_index[2], 
      // 	       global_index[3], 
      // 	       global_index[4], 
      // 	       global_index[5], 
      // 	       global_index[6]
      // 	       );

      // for each global index to be contracted, find multiplication of operands and sum them
      double val=1;

      for( int operand=0; operand<operand_num; operand++){


        // find operand index corresponding to current global index
        size_t op_inds=0;
        for( int dim=0; dim<ndims; dim++){
          if (d_strides_operand_pointers_shr[operand][dim] != 0){
            op_inds += global_index[dim] * d_strides_operand_pointers_shr[operand][dim];
          }
        }

	if( to_power_operands == NULL ){
	  val *= d_operand_pointers[operand][ op_inds ];
	}else{
	  val *= pow(d_operand_pointers[operand][ op_inds ], to_power_operands[operand]);
	}
        ////cuPrintf("val increment operand %d op_inds %d d_operand_pointers %f new val %f\n", operand, op_inds, d_operand_pointers[operand][ op_inds ], val);
      }

      d_output[output_ind] += val;

      //double tmp = d_output[output_ind];
      //cuPrintf("d_output increment output_ind %d val %f new d_output %f\n", output_ind, val, tmp);










      // increment global_index for next loop OR end iteration if done
      for ( int dim=0; dim<ndims; dim++){

        // iterate only over indices which must be contracted -> dim \in contract_dims[] and dim \in any operands
        bool contract_over=false;
        for( int cd=0; cd<contract_dim_num; cd++){
          if( contract_dims[cd] == dim ){
            contract_over = true;
            break;
          }
        }
        if( ! contract_over ){
          // this dimension is fixed skip it, already set to fixed value above
          if( dim == ndims-1 ){
            // if this is also the last dimension we are done
            not_done = false;
          }

	  //cuPrintf("skip index %d opnum %d\n", dim, opnum);
          continue;
        }





        // if we have NOT reached limit of this dimension
        if( global_index[dim] != (d_cards_F_shr[dim]-1) ){
          // increment this dimension
          //cuPrintf("INCREMENT %d %d\n",global_index[dim], d_cards_F_shr[dim]-1);

          global_index[dim]++;

	  // if(counter++ == 10000) {
	  //   cuPrintf("INCREMENT %d %d\n",global_index[dim], d_cards_F_shr[dim]-1);
	  //   // cuPrintf("global_index %d %d %d %d %d %d %d %d %d %d\n", 
	  //   // 	     global_index[0], 
	  //   // 	     global_index[1], 
	  //   // 	     global_index[2], 
	  //   // 	     global_index[3], 
	  //   // 	     global_index[4], 
	  //   // 	     global_index[5], 
	  //   // 	     global_index[6], 
	  //   // 	     global_index[7], 
	  //   // 	     global_index[8], 
	  //   // 	     global_index[9]);
	    
	  //   cuPrintf("counter limited opnum %d\n", opnum);
	  //   not_done=false;
	  //   return;
	  // }

          break;
        }else{



          // we have reached limit of this dimension

          // if next dimension is at limit as well, skip this dimension, operation will take place in next dimension

          if( dim != (ndims-1) &&
              (global_index[dim+1] == (d_cards_F_shr[dim+1]-1) || d_strides_output_shr[dim+1] == 0 ) ){
            ////cuPrintf("SKIP\n");
            continue;
          }else{

            // if this is the last dimension (and it is full) no increment is possible increment error
            if (dim == ndims-1){
              ////cuPrintf("NOT DONE -> FALSE\n");
              not_done = false;
              break;
            }

	    // if there are no other dimensions to increment we are done
	    bool found_dim_to_increment = false;
	    for( int nd=dim+1; nd<ndims; nd++){
	      bool found=false;
	      //cuPrintf("dim %d\n", nd); 
	      for( int cd=0; cd<contract_dim_num; cd++){
		if( contract_dims[cd] == nd ){
		  //cuPrintf("contract_dims[%d] %d\n" , cd, contract_dims[cd]);
		  found = true;
		  break;
		}
	      }
	      if( found ){
		found_dim_to_increment = true;
		break;
	      }
	    }
	    if( found_dim_to_increment == false ){
	      not_done = false;
	      //cuPrintf("found_dim_to_increment BREAK\n");
	      break;
	    }else{
	      //cuPrintf("found_dim_to_increment NOT BREAK\n");
	    }


            // make this and all previous dimensions zero
            for (int dim_prev=dim; dim_prev>=0 ; dim_prev--){

              bool is_prev_dim_contracted=false;
              for( int cd=0; cd<contract_dim_num; cd++){
                if( contract_dims[cd] == dim_prev ){
                  is_prev_dim_contracted = true;
                  break;
                }
              }

	      // non-contracted dimensions are fixed to index values
	      if( is_prev_dim_contracted ){
		// this dimension is contracted, restart it
		global_index[dim_prev] = 0;
	      }
            }


            // increment next dimension non-zero, contracted dimension
	    for( int nd=dim+1; nd<ndims; nd++ ){
	      if( d_strides_output_shr[nd] != 0 ){

		bool is_next_dim_contracted = false;
		for( int cd=0; cd<contract_dim_num; cd++){
		  if( contract_dims[cd] == nd ){
		    is_next_dim_contracted = true;
		    break;
		  }
		}
		
		if( is_next_dim_contracted ){
		  global_index[dim+1]++;
		  break;
		}
	      }
	    }

            break;
          }
        }
      }
    }while(not_done);
  }
}




__global__ void printData(double* data, size_t count, size_t id){
  //cuPrintf("printData id %d", id);
  for(int i=0; i<count; i++){
    double tmp=data[i];
    cuPrintf("data[%d] = %e\n", i, tmp);
  }
}

__global__ void printData(size_t* data, size_t count, size_t id){
  //cuPrintf("printData id %d", id);
  for(int i=0; i<count; i++){
    size_t tmp=data[i];
    cuPrintf("data[%d] = %d\n", i, tmp);
  }
}

