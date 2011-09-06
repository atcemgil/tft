/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */

#include "mct_tensorop_utils.cuh"

double get_element(const ct* h_ct, size_t* global_index, const char* str=""){
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


void set_element(ct* h_ct, size_t* global_index, double val, const char* str=""){
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




void increment_cur_index(size_t ndims, const size_t* h_full_cardinalities, size_t* global_index){
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



void mct_tensorop_c(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, const size_t* h_full_cardinalities,size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C){

    if ( isHadamard ){

      for( size_t i=0; i<h_C.element_number; i++)
        h_C.data[i] = h_A.data[i] * h_B.data[i];

      memcpy(m_C, h_C.data, h_C.mem_size);

    }else{

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
      }else{
        memcpy(m_C, h_F.data, h_F.mem_size);
      }

    }

    if ( PRINT_CT ) print_ct("C: contraction result", &h_C, true);

}
