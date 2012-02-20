/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#ifndef MCT_TENSOROP_UTILS_H
#define MCT_TENSOROP_UTILS_H

#include "settings.h"
#include "tensor.h"
#include "mex.h"
#include <vector>
#include <iostream>

#include "tensorop_seq.cuh"
#include "tensorop_par.cuh"

#define REGISTER_CT(obj) register_ct(#obj, &obj)

void print_ct(const char* txt, ct* ct, bool printdata=false);
//void prepareHostTensor(ct* h_ct, const mxArray* m_data, const mxArray* tensor_card, const char* txt=NULL);
size_t* gen_range_permutation(std::vector<size_t> permutation_list, size_t* elnum);
void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims, 
			      const char* txt=NULL, bool rand=false,
			      bool init_to_one=false);

void free_ct(ct* c);

#include <string>
#include <map>

extern std::map<std::string,ct*> h_objs; // defined in mct_tensorop_utils.cu
extern size_t* h_full_cardinalities;     // defined in mct_tensorop_utils.cu

void register_ct(std::string key, ct* obj);
void clear_ct();
void read_config();
void print_config();


// using this construct avoids unnecessary memory duplication
// example:
// operation occurs C = A * B
// if there is no contraction in this operation and this is not a hadamard operation: 
// then result will be stored in F not C
// in these cases using the operate function will take care of using the correct memory location
// however this only works in one level
// example:
// operation occurs C1 = A * B
// assume result is stored in F
// operation occurs C2 = Z * Y
// assume result is stored in F
// results of the first operation are lost!
struct operation{
  std::string A;
  std::string B;
  std::string C;
  std::string F;
  int to_power_A;
  int to_power_B;
  bool isHadamard;
  operation_type op_type;
  size_t ndims;
  bool result_in_F;
  // pointer to one of the following functions: mct_tensorop_gpu_keys, mct_tensorop_cpp_keys
  bool (*operate) (operation_type, size_t, bool*, std::string, std::string, std::string, std::string, int, int);
};

#include <vector>
void print_oc_element(operation* oc);
void print_oc(std::vector<operation>* operation_chain);
void operate(std::vector<operation>* operation_chain);

bool check_input_keys(std::string A, std::string B, std::string C, std::string F);

void print_model_elements(std::vector<m_tensor>* model_elements, m_tensor* x_tensor);
void print_model_elements_text(std::vector<m_tensor>* model_elements, char* text);
void assign_m_tensor_cards_numeric(m_tensor* m_t, mxChar* V_char, double* V_numeric, size_t ndims);

void oc_push_back(std::vector<operation>* operation_chain, operation_type op_type, size_t ndims, std::string A, std::string B, std::string C, bool is_parallel, std::string F="F", int to_power_A=1, int to_power_B=1);

int get_latent_tensor_num(bool* R, size_t v, size_t max_alpha, size_t max_v);

inline bool is_hadamard(operation_type op_type){
  if ( op_type == HADAMARD_DIV || op_type == HADAMARD_MUL || op_type == HADAMARD_SUM){ return true; }
  else{ return false; }
}

inline bool is_multiplication(operation_type op_type){
  if ( op_type == HADAMARD_MUL || op_type == GMULT){ return true; }
  else{ return false; }
}

inline bool is_addition(operation_type op_type){
  if ( op_type == HADAMARD_SUM ){ return true;
  }else{ return false; }
}

inline bool is_division(operation_type op_type){
  if ( op_type == HADAMARD_DIV ){  return true; }
  else{  return false; }
}


#endif
