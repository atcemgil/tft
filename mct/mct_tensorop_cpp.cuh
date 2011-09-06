#ifndef MCT_TENSOROP_CPP_H
#define MCT_TENSOROP_CPP_H

double get_element(const ct* h_ct, size_t* global_index, const char* str="");
void set_element(ct* h_ct, size_t* global_index, double val, const char* str="");
void increment_cur_index(size_t ndims, const size_t* h_full_cardinalities, size_t* global_index);
void mct_tensorop_cpp(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, const size_t* h_full_cardinalities,size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C);

#endif