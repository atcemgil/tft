#ifndef MCT_TENSOROP_GPU_H
#define MCT_TENSOROP_GPU_H

dev_ptrs prepareDeviceParameters(const size_t* h_full_cardinalities, size_t ndims, const ct* h_A, const ct* h_B, ct* h_C, ct* h_F, size_t zero_cardinality_dims_elnum, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, bool isHadamard);

void mct_tensorop_gpu(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, const size_t* h_full_cardinalities,size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, size_t h_zero_cardinality_dim_tuple_size_C, size_t use_multiplication);


#include <string>

void addToTransferList(std::string key, ct* obj);

void register_ct(std::string key, ct* obj);
#define NOSWAP 999999
bool mct_tensorop_gpu_keys(bool isHadamard, 
			   size_t use_multiplication,
			   size_t ndims,
			   std::string A, std::string B, std::string C, std::string F="F",
			   size_t swap_A_first=NOSWAP, size_t swap_A_second=NOSWAP,
                           size_t swap_B_first=NOSWAP, size_t swap_B_second=NOSWAP);
void transferToDevice(const size_t* h_full_cardinalities, size_t ndims);
void resetDevice();
void transferFromDevice(double* matlab_storage, std::string d_storage_key);


#endif
