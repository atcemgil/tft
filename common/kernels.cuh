/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#ifndef MCT_KERNELS_H
#define MCT_KERNELS_H

__global__ void hadamard_mul(double* d_A, double* d_B, double* d_C, size_t C_element_number);
__global__ void hadamard_div(double* d_A, double* d_B, double* d_C, size_t C_element_number);
__global__ void genFullResult(size_t* d_total_cards, size_t ndims,
                              size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_F,
                              double* d_A, double* d_B, double* d_F,
			      size_t F_element_number, size_t A_element_number, size_t B_element_number,
			      size_t use_multiplication);
__global__ void contractFintoC(size_t ndims,
                               size_t* d_strides_F, size_t* d_strides_C,
                               double* d_F, double* d_C,
                               size_t C_element_number,
                               size_t* d_zero_cardinality_dim_tuples_C,
                               size_t zero_cardinality_dim_tuple_size_C,
                               size_t zero_cardinality_dim_tuples_C_element_number);

#endif 
