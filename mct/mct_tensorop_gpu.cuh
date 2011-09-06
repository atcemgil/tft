#ifndef MCT_TENSOOP_GPU_H
#define MCT_TENSOOP_GPU_H

dev_ptrs prepareDeviceParameters(const size_t* h_full_cardinalities, size_t ndims, const ct* h_A, const ct* h_B, ct* h_C, ct* h_F, size_t zero_cardinality_dims_elnum, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, bool isHadamard);

void mct_tensorop_gpu(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, const size_t* h_full_cardinalities,size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, size_t h_zero_cardinality_dim_tuple_size_C, size_t use_multiplication);

#endif