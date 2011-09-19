/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */

#ifndef MCT_TENSOROP_UTILS_H
#define MCT_TENSOROP_UTILS_H

#include "settings.h"
#include "tensor.h"
#include "mex.h"
#include <vector>
#include <iostream>

void print_ct(const char* txt, ct* ct, bool printdata=false);
void prepareHostTensor(ct* h_ct, const mxArray* m_data, const mxArray* tensor_card, const char* txt=NULL);
size_t* gen_range_permutation(std::vector<size_t> permutation_list, size_t* elnum);
void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims, const char* txt
=NULL, bool rand=false);

#include <string>
#include <map>

extern std::map<std::string,ct*> h_objs; // defined extern elsewhere
extern size_t* h_full_cardinalities;     // defined extern elsewhere

void register_ct(std::string key, ct* obj);
void clear_ct();





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
  bool isHadamard;
  bool use_multiplication;
  size_t ndims;
  bool result_in_F;
  // pointer to one of the following functions: mct_tensorop_gpu_keys, mct_tensorop_cpp_keys
  bool (*operate) (bool, bool, size_t, bool*, std::string, std::string, std::string, std::string);
};

#include <vector>
void operate(std::vector<operation>* operation_chain);

bool check_input_keys(std::string A, std::string B, std::string C, std::string F);


#endif
