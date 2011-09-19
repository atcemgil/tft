/*
 * author: ck
 * created: 05.08.2011
 * advisor: atc
 */

#include "mct_tensorop_utils.cuh"
#include <stdlib.h>

// real definitions for global (extern) variables
std::map<std::string,ct*> h_objs;
size_t* h_full_cardinalities;

void register_ct(std::string key, ct* obj){
  h_objs[key] = obj;
}

void reset_ct(){
  h_objs.clear();
}


void print_ct(const char* txt, ct* ct, bool printdata){ //bool print_config=false,

  std::cout << txt << std::endl;

  //if (print_config) print_ct_config(txt, ct->config);

  std::cout << "Mem size " << ct->mem_size << std::endl;

  std::cout << "Element number " << ct->element_number << std::endl;

  std::cout << "Cardinalities for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->ndims; i++){
    std::cout << ct->cardinalities[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Strides for each dimension of this object "<< std::endl;
  for (size_t i=0; i< ct->ndims; i++){
    std::cout << ct->strides[i] << " ";
  }
  std::cout << std::endl;

  if (printdata){
    std::cout << "Data" << std::endl;
    for (size_t i=0; i< ct->element_number; i++){
      std::cout << ct->data[i] << " ";
    }
  }
  std::cout << std::endl << std::endl << std::endl;
}



void prepareHostTensor(ct* h_ct, const mxArray* m_data, const mxArray* tensor_card, const char* txt){
  h_ct->ndims = mxGetNumberOfElements(tensor_card); // assumes both objects of same size
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * mxGetNumberOfElements(tensor_card) );

  // assign cardinalities for the tensor objects and init cur_ind values
  size_t cum_sum=1;
  for (size_t i=0; i<h_ct->ndims; i++){
    h_ct->cardinalities[i] = ((double *)mxGetData(tensor_card))[i];
    //std::cout << "TC dim "<< i << " cardinality assignment: "
    //          << h_ct->cardinalities[i]
    //          << " <- " << ((double *)mxGetData(tensor_card))[i] << std::endl;

    if ( h_ct->cardinalities[i] == 0){
      h_ct->strides[i]=0;
    }else{
      h_ct->strides[i]=cum_sum;
      cum_sum *= h_ct->cardinalities[i];
    }
  }

  // assign h_ct host data
  size_t elnum = (size_t) mxGetNumberOfElements(m_data);
  if ( txt != NULL && COUT ){
    std::cout << txt << std::endl;
  }
  if ( COUT ) std::cout << "prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;

  h_ct->data = (double*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (double*)mxGetData(m_data), h_ct->mem_size);

  if ( PRINT_CT ) print_ct("prepareDeviceTensor h_ct",h_ct,true);
}



void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims, const char* txt, bool rand){
  h_ct->ndims = ndims;
  h_ct->cardinalities = (size_t*) malloc( sizeof(size_t) * h_ct->ndims );
  h_ct->strides = (size_t*) malloc( sizeof(size_t) * h_ct->ndims );

  size_t elnum=1;
  // assign cardinalities for the tensor objects and init cur_ind values
  size_t cum_sum=1;
  for (size_t i=0; i<h_ct->ndims; i++){
    if ( tensor_card[i] != 0 ){
      elnum *= tensor_card[i];
    }

    h_ct->cardinalities[i] = tensor_card[i];
    // std::cout << "TC dim "<< i << " cardinality assignment: "
    //           << h_ct->cardinalities[i] << " <- " << tensor_card[i] << std::endl;

    if ( h_ct->cardinalities[i] == 0){
      h_ct->strides[i]=0;
    }else{
      h_ct->strides[i]=cum_sum;
      cum_sum *= h_ct->cardinalities[i];
    }
  }

  // assign h_ct host data
  if ( txt != NULL && COUT == true){
    std::cout << txt << std::endl;
  }
  if ( COUT ) std::cout << "prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;

  if (data == NULL){
    //h_ct->data = (double*)calloc( h_ct->element_number, sizeof(double) );
    h_ct->data = (double*) malloc(h_ct->mem_size);
    for (size_t i=0; i<h_ct->element_number; i++ ) 
      if (rand)
	h_ct->data[i]= ((double) std::rand()) / (double)(RAND_MAX+1.0);
      else
	h_ct->data[i]=(double)0;

  }else{
    h_ct->data = (double*)malloc( h_ct->mem_size );
    memcpy(h_ct->data, data, h_ct->mem_size);
  }

  if ( PRINT_CT ) print_ct("prepareDeviceTensor h_ct",h_ct,true);

}










// Recursive function which generates all permutations of a given list
void gen_range_permutation_helper(std::vector<size_t> iter_dims, std::vector<size_t> cur_perm, std::vector<size_t>* acc){
  if ( iter_dims.size() == 0 ){
    if (COUT){
      std::cout << "final cur_perm" <<  std::endl;
      for ( size_t j=0; j<cur_perm.size(); j++){
        std::cout << cur_perm.at(j) << std::endl;
      }
    }
    acc->insert(acc->end(), cur_perm.begin(), cur_perm.end());
  }else{
    // pick one dimension from iter_dims and iterate for each element in its cardinality
    size_t one_dim = iter_dims.front();
    // remove that dim for the remaining recursions
    iter_dims.erase(iter_dims.begin());

    for ( size_t i=0; i<one_dim ; i++){
      std::vector<size_t> tmp_vec (cur_perm.begin(), cur_perm.end());
      tmp_vec.push_back(i);
      if ( COUT ){
        std::cout << " tmp_vec " << std::endl;
        for ( size_t j=0; j<tmp_vec.size(); j++){
          std::cout << tmp_vec.at(j) << std::endl;
        }
      }
      gen_range_permutation_helper( iter_dims, tmp_vec, acc );
    }
  }
}

// generates range-permutation of numbers given in permutation_list
// Example
// (2 3) -> ( 0,0,   0,1,  0,2,   1,0,   1,1,  1,2 )
size_t* gen_range_permutation(std::vector<size_t> permutation_list, size_t* elnum){

  std::vector<size_t> empty_list;
  std::vector<size_t> acc;

  gen_range_permutation_helper(permutation_list, empty_list, &acc);

  size_t* acc_array = (size_t*) malloc(sizeof(size_t)*acc.size());
  std::copy(acc.begin(), acc.end(), acc_array);
  (*elnum) = acc.size();

  if ( COUT ){
    std::cout << "gen_range_permutation \n acc:" << std::endl;
    for ( size_t i=0; i<acc.size(); i++){
      std::cout << acc.at(i) << std::endl;
    }
    std::cout << "elnum " << *elnum << std::endl;
  }

  return acc_array;
}


bool check_input_keys(std::string A, std::string B, std::string C, std::string F){
  // check requested objects are registered
  if ( h_objs.find(A) == h_objs.end() ||
       h_objs.find(B) == h_objs.end() ||
       h_objs.find(C) == h_objs.end() ||
       h_objs.find(F) == h_objs.end() ){
    std::cout << "mct_tensorop_gpu_keys: all of requested keys should be registered "
              << " requested keys: A " << A << " B " << B << " C " << C << " F " << F
              << std::endl;
    return false;
  }

  return true;
}



void operate(std::vector<operation>* operation_chain){
  std::vector<operation>::iterator it;

  for ( it=operation_chain->begin() ; it < operation_chain->end(); it++ ){

    std::string A = it->A;
    std::string B = it->B;
    
    if ( it != operation_chain->begin() && // if this is not the first operation and
	 (it-1)->result_in_F){ // previous operation's result was stored in F instead of C

      //we must use F instead of C in next operation for any input named 
      //same as previous operation's output object (C)

      // example:
      // in previous operation we had
      // E * G = H
      // but H was not stored in object pointed by C
      // it was stored in object pointed by F to avoid copying

      // thus if we are using H in current operation's input (A or B)
      // we must not use the object pointed by C (H) in previous operation
      // but instead use the object pointed by F in previous operation
      
      if ( it->A.compare( (it-1)->C ) == 0)
	A = (it-1)->F;

      if ( it->B.compare( (it-1)->C ) == 0)
	B = (it-1)->F;

    }

    if (COUT_operate)
      std::cout << "util: operate A " << A << " B " << B << " C " << it->C << " F " << it->F 
		<< " chain size " << operation_chain->size() 
		<< std::endl;
    
    (it->operate) (it->isHadamard, 
		   it->use_multiplication, 
		   it->ndims, 
		   &(it->result_in_F), 
		   A, B, it->C, 
		   it->F);

    if (COUT_operate)
      std::cout << " util result: result_in_F " << it->result_in_F
		<< std::endl;

  }

}

