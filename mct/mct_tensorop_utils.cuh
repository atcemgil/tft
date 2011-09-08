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
void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims, const char* txt=NULL, bool rand=false);

#endif
