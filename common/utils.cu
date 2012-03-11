/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#include "utils.cuh"
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <fstream>

// defined in settings.h
bool COUT = false;
bool PRINT_CT = false;
bool PRINT_CHAIN = false;
bool COUT_operate = false;

bool COUT_get_set = false;
bool CUPRINTF = false;
bool COUT_contract = false;
bool COUT_cpp_contract = false;
bool COUT_cpp_contract2 = false;

size_t NUM_BLOCKS = 53000;
size_t THREADS_FOR_BLOCK =  512;


// real definitions for global (extern) variables
std::map<std::string,ct*> h_objs;
size_t* h_full_cardinalities;
size_t* h_full_cardinalities2;

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
  if ( txt != NULL && (PRINT_CT || COUT) ){
    std::cout << txt << std::endl;
  }
  if ( COUT ) std::cout << "prepareDeviceTensor elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;

  h_ct->data = (double*)malloc( h_ct->mem_size );
  memcpy(h_ct->data, (double*)mxGetData(m_data), h_ct->mem_size);

  if ( PRINT_CT ) print_ct("prepareDeviceTensor h_ct",h_ct,true);
}



void prepareHostTensorFromCpp(ct* h_ct, double* data, size_t* tensor_card, size_t ndims, const char* txt, bool rand, bool init_to_one, bool init_data){
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
  if ( txt != NULL && ( COUT || PRINT_CT) ){
    std::cout << txt << std::endl;
  }
  if ( COUT ) std::cout << "prepareDeviceTensorFromCpp elnum " << elnum << std::endl;
  h_ct->mem_size= sizeof(double) * elnum;
  h_ct->element_number = elnum;



  if ( init_data ){
    if (data == NULL){
      //std::cout << " prepareHostTensorFromCpp " << txt << " data = NULL" << std::endl;

      //h_ct->data = (double*)calloc( h_ct->element_number, sizeof(double) );
      h_ct->data = (double*) malloc(h_ct->mem_size);
      for (size_t i=0; i<h_ct->element_number; i++ )
        if (rand)
          h_ct->data[i]= ((double) std::rand()) / (double)(RAND_MAX+1.0);
        else{
          if ( init_to_one )
            h_ct->data[i]=(double)1;
          else
            h_ct->data[i]=(double)0;
        }

    }else{
      //std::cout << " prepareHostTensorFromCpp " << txt << " data = NULL DEGIL" << std::endl;
      h_ct->data = (double*)malloc( h_ct->mem_size );
      memcpy(h_ct->data, data, h_ct->mem_size);
    }
  }else{
    h_ct->data = NULL;
  }


  if ( PRINT_CT ){
    if( init_data ) {
      print_ct("prepareDeviceTensorFromCpp h_ct",h_ct,true);
    }else{
      print_ct("prepareDeviceTensorFromCpp h_ct",h_ct,false);
    }
  }

}










// Recursive function which generates all permutations of a given list
void gen_range_permutation_helper(std::vector<size_t> iter_dims, std::vector<size_t> cur_perm, std::vector<size_t>* acc){
  if ( iter_dims.size() == 0 ){
    if (COUT_contract){
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
      if ( COUT_contract ){
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

  if ( COUT_contract ){
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

  size_t opnum=0;
  for ( it=operation_chain->begin() ; it < operation_chain->end(); it++ ){
    if(COUT) {
      std::cout << "operation chain number " << opnum << " " << std::endl;
      opnum++;
    }


    std::string A = it->A;
    std::string B = it->B;

    // if ( it != operation_chain->begin() && // if this is not the first operation and
    //      (it-1)->result_in_F){ // previous operation's result was stored in F instead of C

    // //   //we must use F instead of C in next operation for any input named
    // //   //same as previous operation's output object (C)

    // //   // example:
    // //   // in previous operation we had
    // //   // E * G = H
    // //   // but H was not stored in object pointed by C
    // //   // it was stored in object pointed by F to avoid copying

    // //   // thus if we are using H in current operation's input (A or B)
    // //   // we must not use the object pointed by C (H) in previous operation
    // //   // but instead use the object pointed by F in previous operation

    //   if ( it->A.compare( (it-1)->C ) == 0)
    //  A = (it-1)->F;

    //   if ( it->B.compare( (it-1)->C ) == 0)
    //  B = (it-1)->F;

    // }

    if (COUT_operate)
      print_oc_element(&(*it));


    (it->operate) (it->op_type,
                   it->ndims,
                   &(it->result_in_F),
                   A, B, it->C,
                   it->F,
                   it->to_power_A, it->to_power_B);

    if (COUT_operate)
      std::cout << " operate end: result_in_F " << it->result_in_F
                << std::endl;

  }

}

void print_oc_element(operation* oc){
  std::cout << (*oc).A;
  //if ( (*oc).op_type ) std::cout << " . ";
  //else std::cout << " / " ;

  if ( is_multiplication(oc->op_type) ) std::cout << " * " ;
  else if ( is_division(oc->op_type) )  std::cout << " / " ;
  else if ( is_addition(oc->op_type) )  std::cout << " + " ;

  std::cout << (*oc).B << " -> " << (*oc).C << " (F: " << (*oc).F << ")"
            << " \t\t  ndims "<< (*oc).ndims << " result_in_F " << (*oc).result_in_F
            << " isHadamard " << (*oc).isHadamard << " operation_type " << (*oc).op_type
            << " \t to_power_A " << oc->to_power_A << "\t to_power_B " << oc->to_power_B << std::endl;
}

void print_oc(std::vector<operation>* operation_chain){
  std::cout << "printing operation chain" << std::endl;

  for (int o=0; o<(*operation_chain).size(); o++){
    std::cout << "operation " << o << ": \t";
    print_oc_element( &((*operation_chain)[o]) );
  }
}

void print_m_tensor(m_tensor* m_t){
  std::cout << "cards_char element number " << strlen(m_t->cards_char) << std::endl;
  for (size_t m_i=0; m_i<strlen(m_t->cards_char); m_i++){
    std::cout << (char) (*m_t).cards_char[m_i]  <<" ";
  }
  std::cout << std::endl << "cards_numeric number " << (*m_t).cards_numeric.size() << std::endl;
  for (size_t m_i=0; m_i<(*m_t).cards_numeric.size(); m_i++){
    std::cout << (size_t) (*m_t).cards_numeric[m_i] << " ";
  }

  std::cout << std::endl << "data pointer " << m_t->data;
  std::cout << std::endl << "is_updateable " << m_t->is_updateable;
  std::cout << std::endl;
}

void print_model_elements(std::vector<m_tensor>* model_elements, m_tensor* x_tensor){
  std::cout << "printing model elements" << std::endl;
  for (size_t m=0; m<(*model_elements).size(); m++){
    print_m_tensor( &(*model_elements)[m] );
    std::cout << std::endl;
  }

  std::cout << "printing X tensor" << std::endl;
  print_m_tensor( x_tensor );
  std::cout << std::endl;
}

void print_model_elements_text(std::vector<m_tensor>* model_elements, char* text){
  std::cout << text << std::endl;
  for (size_t m=0; m<(*model_elements).size(); m++){
    print_m_tensor( &(*model_elements)[m] );
    std::cout << std::endl;
  }
}

void assign_m_tensor_cards_numeric(m_tensor* m_t, mxChar* V_char, double* V_numeric, size_t ndims){
  for (size_t i=0; i<ndims; i++){
    bool found=false;
    for (size_t m_i=0; m_i< strlen(m_t->cards_char); m_i++){
      //std::cout << "cards_char["<<m_i<<"] " << (char) m_t->cards_char[m_i]
      //<< " V_char["<<i<<"] " << (char)V_char[i] << std::endl;
      if ( (char)m_t->cards_char[m_i] == (char)V_char[i] ){
        m_t->cards_numeric.push_back(V_numeric[i]);
        //std::cout << " insert V_numeric[" << i << "] " << V_numeric[i] << std::endl;
        found=true;
        break; // goto next dimension
      }
    }
    if (!found){
      m_t->cards_numeric.push_back(0);
      //std::cout << " insert 0 " << std::endl;
    }
  }
}

void read_config(){
  std::ifstream is;
  is.open ("config");
  if (is.bad()){
    std::cout << "can not read config file" << std::endl;
  }

  std::string line;
  getline(is,line);
  COUT = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  PRINT_CT = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  PRINT_CHAIN = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  COUT_operate = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  COUT_get_set = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  CUPRINTF = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  COUT_contract = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  COUT_cpp_contract = (line == "1");
  if (is.bad()) return;

  getline(is,line);
  COUT_cpp_contract2 = (line == "1");
  if (is.bad()) return;

  getline(is, line);
  if (is.bad()) return;
  NUM_BLOCKS=atoi(line.c_str());
  if (is.bad()) return;

  getline(is, line);
  if (is.bad()) return;
  THREADS_FOR_BLOCK=atoi(line.c_str());
  if (is.bad()) return;
}

void print_config(){
  std::cout << "COUT" << "\t" << COUT << std::endl;
  std::cout << "PRINT_CT" << "\t" << PRINT_CT << std::endl;
  std::cout << "PRINT_CHAIN" << "\t" << PRINT_CHAIN << std::endl;
  std::cout << "COUT_operate" << "\t" << COUT_operate << std::endl;
  std::cout << "COUT_get_set" << "\t" << COUT_get_set << std::endl;
  std::cout << "CUPRINTF" << "\t" << CUPRINTF << std::endl;
  std::cout << "COUT_contract" << "\t" << COUT_contract << std::endl;
  std::cout << "COUT_cpp_contract" << "\t" << COUT_cpp_contract << std::endl;
  std::cout << "COUT_cpp_contract2" << "\t" << COUT_cpp_contract2 << std::endl;
  std::cout << "NUM_BLOCKS" << "\t" << NUM_BLOCKS << std::endl;
  std::cout << "THREADS_FOR_BLOCK" << "\t" << THREADS_FOR_BLOCK << std::endl;
}

void free_ct(ct* c){
  free(c->cardinalities);
  free(c->strides);
  free(c->data);
}

void oc_push_back(std::vector<operation>* operation_chain, operation_type op_type, size_t ndims, std::string A, std::string B, std::string C, bool is_parallel, std::string F, int to_power_A, int to_power_B){

  operation oc;
  oc.isHadamard = is_hadamard(op_type);
  oc.op_type = op_type;
  oc.ndims = ndims;
  oc.A = A;
  oc.B = B;
  oc.C = C;
  oc.F = F;
  oc.to_power_A = to_power_A;
  oc.to_power_B = to_power_B;
  oc.result_in_F = false;  /// dikkat !!! // untested -> non hadamard , no contraction case

  if (is_parallel){
    oc.operate = &tensorop_par_keys;
  }else{
    oc.operate = &tensorop_seq_keys;
  }

  operation_chain->push_back(oc);



  size_t* full_cardinalities = (size_t*) calloc(ndims, sizeof(size_t)); // defined in mct_tensorop_utils.cuh

  for (size_t dim=0; dim<ndims; dim++){
    size_t max_dim_card = 0;
    if ( max_dim_card < h_objs[A]->cardinalities[dim] )
      max_dim_card = h_objs[A]->cardinalities[dim];
    if ( max_dim_card < h_objs[B]->cardinalities[dim] )
      max_dim_card = h_objs[B]->cardinalities[dim];
    if ( max_dim_card < h_objs[C]->cardinalities[dim] )
      max_dim_card = h_objs[C]->cardinalities[dim];

    full_cardinalities[dim] = max_dim_card;
  }
}

// returns number of used latent tensors for a given observed tensor
int get_latent_tensor_num(bool* R, size_t v, size_t max_alpha, size_t max_v){
  int count=0;
  for (int j=0; j<max_alpha; j++){
    if (R[v + j*max_v]) count++;
  }
  return count;
}





#include "cutil_inline.h"
#include "../common/kernels.cuh"
#include "../common/cuPrintf.cuh"

size_t gen_operation_arguments(std::vector<std::string> ops_str, operands* ops, size_t cur_mem, int* h_to_power){
  size_t operand_elnum = ops_str.size();

  size_t** h_strides_operand_pointers = (size_t**) malloc( operand_elnum * sizeof(size_t*) );
  size_t** h_cards_operand_pointers   = (size_t**) malloc( operand_elnum * sizeof(size_t*) );
  double** h_operand_pointers         = (double**) malloc( operand_elnum * sizeof(double*) );

  for( size_t o=0; o<ops_str.size(); o++){
    h_strides_operand_pointers[o] = get_d_obj_strides()[ops_str[o]];
    h_cards_operand_pointers[o] = get_d_obj_cards()[ops_str[o]];
    h_operand_pointers[o] = get_d_obj_data()[ops_str[o]];
  }

  // copy to device
  cur_mem += sizeof(size_t*)*operand_elnum;
  //std::cout << "   cur_mem increment by " << sizeof(size_t*)*operand_elnum << " new cur_mem " << cur_mem;
  cutilSafeCall(cudaMalloc((void**)&(ops->d_strides_operand_pointers), sizeof(size_t*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops->d_strides_operand_pointers, h_strides_operand_pointers, sizeof(size_t*)*operand_elnum, cudaMemcpyHostToDevice));

  cur_mem += sizeof(size_t*)*operand_elnum;
  //std::cout << "   cur_mem increment by " << sizeof(size_t*)*operand_elnum << " new cur_mem " << cur_mem;
  cutilSafeCall(cudaMalloc((void**)&(ops->d_cards_operand_pointers), sizeof(size_t*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops->d_cards_operand_pointers, h_cards_operand_pointers, sizeof(size_t*)*operand_elnum, cudaMemcpyHostToDevice));

  cur_mem += sizeof(double*)*operand_elnum;
  //std::cout << "   cur_mem increment by " << sizeof(double*)*operand_elnum << " new cur_mem " << cur_mem;
  cutilSafeCall(cudaMalloc((void**)&(ops->d_operand_pointers), sizeof(double*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops->d_operand_pointers, h_operand_pointers, sizeof(double*)*operand_elnum, cudaMemcpyHostToDevice));


  if( h_to_power != NULL ){
    cur_mem += sizeof(int)*operand_elnum;
    //std::cout << "   cur_mem increment by " << sizeof(int)*operand_elnum << " new cur_mem " << cur_mem;
    cutilSafeCall(cudaMalloc((void**)&(ops->d_to_power), sizeof(int)*operand_elnum));
    cutilSafeCall(cudaMemcpy(ops->d_to_power, h_to_power, sizeof(int)*operand_elnum, cudaMemcpyHostToDevice));
  }

  //std::cout << " gen_operation_arguments elnum " << operand_elnum << " curmem " << cur_mem << std::endl;
  return cur_mem;
}

