/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#ifndef MCT_KERNELS_H
#define MCT_KERNELS_H

__global__ void hadamard_mul(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A=1, int to_power_B=1);
__global__ void hadamard_div(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A=1, int to_power_B=1);
__global__ void hadamard_sum(double* d_A, double* d_B, double* d_C, size_t C_element_number, bool print, int to_power_A=1, int to_power_B=1);
__global__ void genFullResult(size_t* d_total_cards, size_t ndims,
                              size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_F,
                              double* d_A, double* d_B, double* d_F,
			      size_t F_element_number, size_t A_element_number, size_t B_element_number,
			      size_t use_multiplication,
			      bool print,
			      int to_power_A=1, int to_power_B=1);
__global__ void contractFintoC(size_t ndims,
                               size_t* d_strides_F, size_t* d_strides_C,
                               double* d_F, double* d_C,
                               size_t C_element_number,
                               size_t* d_zero_cardinality_dim_tuples_C,
                               size_t zero_cardinality_dim_tuple_size_C,
                               size_t zero_cardinality_dim_tuples_C_element_number,
			       bool print);

__global__ void calculate_C(size_t ndims,
                            size_t* d_strides_F, size_t* d_strides_A, size_t* d_strides_B, size_t* d_strides_C,
                            double* d_A, double* d_B, double* d_C,
                            size_t A_element_number, size_t B_element_number, size_t C_element_number,
                            size_t* d_zero_cardinality_dim_tuples_C,
                            size_t zero_cardinality_dim_tuple_size_C,
                            size_t zero_cardinality_dim_tuples_C_element_number,
                            size_t use_multiplication,
                            bool print,
                            int to_power_A=1, int to_power_B=1); // pow(double, size_t) does not appear to exist in cuda:math.h


__global__ void calculate_C_keys(size_t ndims,
                                 size_t operand_num,
                                 size_t* d_strides_operands, //  dim strided: stride_A_1, stride_A_2 .. stride_A_dim, stride_B_1, stride_B_2, ...
                                 size_t* d_indices_operands, //  dim strided
                                 size_t* d_strides_output,
                                 size_t* d_indices_output,
                                 size_t* d_cards_F,
                                 size_t* d_cards_operands,   //  dim strided
                                 double* d_operands,         //  dim strided
                                 double* d_output,
                                 size_t* operand_element_numbers,
                                 size_t output_element_number,
                                 size_t use_multiplication,
                                 bool print,
                                 int to_power_A, int to_power_B);

#endif 
