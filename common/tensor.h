/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <cstring>

// cuda tensor object
struct ct{
  size_t ndims;

  // defines size of each dimension for this tensor
  // must be allocated dynamically as an array of type size_t
  // size of the array must be equal to config.ndims
  size_t* cardinalities;

  // holds cumulative sum of cardinalities vector
  size_t* strides;

  // size of the corresponding data
  size_t mem_size;

  // number of elements in the data
  size_t element_number;

  // points to the values of this tensor
  double* data;
};

// data pointers required to perform the operations
// used as a convenient storage on the host side
struct dev_ptrs{

  size_t* d_full_cardinalities;

  size_t* d_strides_A;
  size_t* d_strides_B;
  size_t* d_strides_F;
  size_t* d_strides_C;

  double* d_data_A;
  double* d_data_B;
  double* d_data_C;
  double* d_data_F;

  // holds index combinations of dimensions with zero cardinality on result tensor
  // contraction operation loops over this array
  //
  // structure: assume we have the following cardinalities
  //            [ 2 2 0 2 2 0 2 ]
  //            we have 2 indices to loop through on contraction (2 and 5)
  //            therefore we need a vector such as the following:
  //            (assuming missing indices have cardinality 2 in other tensors)
  //            [ 0 0   0 1    1 0    1 1 ]
  //            in order to be able to sum through all elements in the full tensor
  // 
  // Using this structure kernels can loop through the full tensor to locate elements to contract
  // with the number of zero cardinality dimensions available.
  size_t* d_zero_cardinality_dim_tuples_C;

  // each tuple contains this many elements
  size_t zero_cardinality_dim_tuple_size_C;

  // total size of d_zero_cardinality_dim_tuples_C
  size_t zero_cardinality_dim_tuples_C_element_number;
};


#include <vector>
#include "mex.h"
struct m_tensor{
  char* cards_char; // ['i', 'j'] 
  std::vector<size_t> cards_numeric; // [2, 3, 0]
  double* data;

  // used by gctf
  bool is_updateable;
};

#endif
