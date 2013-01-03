

#ifndef MCT_TENSOROP_CPP_H
#define MCT_TENSOROP_CPP_H

#include "enums.cuh"
#include "tensor.h"

double get_element(const ct* h_ct, size_t* global_index, const char* str="");
void set_element(ct* h_ct, size_t* global_index, double val, const char* str="");
void increment_cur_index(size_t ndims, size_t* obj_cardinalities, size_t* global_index);
void tensorop_seq(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C);

#include <string>

// returns false on error (specified input object not found in object map)
// result_in_F is true if result is not copied to the the C 
bool tensorop_seq_keys( operation_type op_type,
			size_t ndims,
			bool* result_in_F,
			std::string A, std::string B, std::string C, 
			std::string F="F", 
			int to_power_A=1, int to_power_B=1
			);

#endif
