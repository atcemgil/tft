
// cuda tensor operation configuration object
struct ct_config{
  // defines how many dimensions are there
  size_t ndims;

  // defines the maximum possible size of each dimension
  //   for all tensors using this configuration
  // must be allocated dynamically as an array of type size_t
  // size of the array must be equal to ndims
  size_t* cardinalities;

  // total size of the related objects
  // maximum of cardinality of input objects
  // cardinality for an object is found by multiplying object's cardinalities of each dimension
  size_t total_cardinality;

  // number of elements in the data
  size_t element_number;

  // index of the dimension to contract over
  //size_t contract_dim;
};


// cuda tensor object
struct ct{
  // related configuration object
  ct_config* config;

  // defines size of each dimension for this tensor
  // must be allocated dynamically as an array of type size_t
  // size of the array must be equal to config.ndims
  size_t* cardinalities;

  // holds cumulative sum of cardinalities vector
  size_t* strides;

  // size of the corresponding data
  size_t mem_size;

  // points to the values of this tensor
  double* data;
};








///////////////////////////////////////////////////////////////////////////////
// STORAGE

// non zero value indicates incrementation error
// 1 : increment beyond capacity
//  int increment_error;


// current index value, used to reference double* data
// has same dimension as the cardinalities (config->ndims)
//size_t* cur_ind;


// zero cardinality indices are incremented in this index
// this is not used in any pointer operation
//size_t* cur_anti_ind;


/*
// compact structure carying pointers to elements of a cudatensor on the device
struct dev_ct_ptrs{
  ct* ct;
  ct_config* ctc;
  size_t* cardinalities;
  float* data;
};
*/

/*
// proxy to current index array
inline size_t get_ct_ind(ct* h_ct, size_t dim){
  if(h_ct->cardinalities[dim] == 0){
    return h_ct->cur_anti_ind[dim];
  }else{
    return h_ct->cur_ind[dim];
  }
}
inline void set_ct_ind(ct* h_ct, size_t dim, size_t val){
  if(h_ct->cardinalities[dim] == 0){
    h_ct->cur_anti_ind[dim] = val;
  }else{
    h_ct->cur_ind[dim] = val;
  }
}
inline void inc_ct_ind(ct* h_ct, size_t dim){
  if(h_ct->cardinalities[dim] == 0){
    h_ct->cur_anti_ind[dim]++;
  }else{
    h_ct->cur_ind[dim]++;
  }
}
*/
