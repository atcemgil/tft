/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#ifndef MCT_TENSOROP_GPU_H
#define MCT_TENSOROP_GPU_H

#include "enums.cuh"

dev_ptrs prepareDeviceParameters(size_t ndims, const ct* h_A, const ct* h_B, ct* h_C, ct* h_F, size_t zero_cardinality_dims_elnum, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, bool isHadamard);

void tensorop_par(bool isHadamard, const ct& h_A, const ct& h_B, ct& h_C, double* m_C, ct& h_F, size_t ndims, size_t h_zero_cardinality_dim_tuples_C_element_number, const size_t* h_zero_cardinality_dim_tuples_C, size_t h_zero_cardinality_dim_tuple_size_C, size_t use_multiplication);




#include <string>
#include <map>


// extern std::map<std::string,size_t*> d_obj_strides;
// extern std::map<std::string,size_t*> d_obj_cards;
// extern std::map<std::string,double*> d_obj_data;

void addToTransferList(std::string key, ct* obj);

bool tensorop_par_keys( operation_type op_type,
                        size_t ndims,
                        bool* result_in_F,
                        std::string A, std::string B, std::string C,
                        std::string F="F",
                        int to_power_A=1, int to_power_B=1
                        );

void transferToDevice(size_t full_ndims);
void resetDevice();
void transferFromDevice(double* matlab_storage, std::string d_storage_key);


std::map<std::string,size_t*> get_d_obj_strides();
std::map<std::string,size_t*> get_d_obj_cards();
std::map<std::string,double*> get_d_obj_data();


#endif
