/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#include <string.h>
#include <sstream>

#include "../common/utils.cuh"




#include "cutil_inline.h"
#include "../common/kernels.cuh"
#include "../common/cuPrintf.cuh"

// sil??
#include "../common/cuPrintf.cu"

struct operands{
  size_t** d_strides_operand_pointers; // pointer to stride list, one for each operand
  size_t** d_cards_operand_pointers;
  double** d_operand_pointers;
};

void gen_operation_arguments(std::vector<std::string> ops_str, operands* ops){
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
  cutilSafeCall(cudaMalloc((void**)&(ops->d_strides_operand_pointers), sizeof(size_t*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops->d_strides_operand_pointers, h_strides_operand_pointers, sizeof(size_t*)*operand_elnum, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(ops->d_cards_operand_pointers), sizeof(size_t*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops->d_cards_operand_pointers, h_cards_operand_pointers, sizeof(size_t*)*operand_elnum, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void**)&(ops->d_operand_pointers), sizeof(double*)*operand_elnum));
  cutilSafeCall(cudaMemcpy(ops->d_operand_pointers, h_operand_pointers, sizeof(double*)*operand_elnum, cudaMemcpyHostToDevice));

}



void pltf(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool is_parallel){


  if ( nrhs < 7 ){
    std::cout << "pltf: Factorization operation requires at least 7 arguments. " << std::endl
              << "Number iterations, example 30" << std::endl
              << "V, index set of factorization operation, example ['i', 'j', 'k']" << std::endl
              << "cardinalities, cardinality for each index provided in V, example [2, 3, 4]" << std::endl
              << "X, index set and data elements for the tensor to be factorized, example ['i','j'], X" << std::endl
              << "Z_alpha, desired factor tensor index set and data list, data may be empty list , example ['i', 'j'], [], ['j', 'k'], Z_2" << std::endl
              << std::endl << "Tensor objects are represented by the indices they use from the index set V." << std::endl
              << "Specified index order must match given data dimension order."
              << std::endl;
    return;
  }


  // prepare model elements  //////////////////////////////////////////////////////

  size_t op_iter_count = ((double *)mxGetData(prhs[0]))[0];

  size_t factor_count = (nrhs - 5)/2;

  m_tensor x_tensor;
  x_tensor.cards_char = (char*) malloc(mxGetNumberOfElements(prhs[3])+1);
  for (size_t i=0; i<=mxGetNumberOfElements(prhs[3]); i++)
    if ( i == mxGetNumberOfElements(prhs[3]) )
      x_tensor.cards_char[i] = '\0';
    else
      x_tensor.cards_char[i] = mxGetChars(prhs[3])[i];


  const mxArray* x_tensor_data = prhs[4];

  std::vector<m_tensor> model_elements;
  for (size_t t=0; t<factor_count*2; t+=2){
    m_tensor tmp_m_tensor;
    tmp_m_tensor.cards_char = (char*) malloc(mxGetNumberOfElements(prhs[5+t])+1);
    for (size_t i=0; i<=mxGetNumberOfElements(prhs[5+t]) ; i++)
      if ( i == mxGetNumberOfElements(prhs[5+t]) )
        tmp_m_tensor.cards_char[i] = '\0';
      else
        tmp_m_tensor.cards_char[i] = (char) mxGetChars(prhs[5+t])[i] ;

    if ( mxGetNumberOfElements(prhs[5+t+1]) == 0 ){
      // tensor init data is not given
      tmp_m_tensor.data = NULL;
    }else{
      // tensor init data is given, save pointer
      tmp_m_tensor.data = (double*) mxGetData(prhs[5+t+1]);
      if (COUT) std::cout << "found factor with init data. Data size " << mxGetNumberOfElements(prhs[5+t+1]) << std::endl;
    }

    model_elements.push_back(tmp_m_tensor);
  }

  if(COUT) std::cout << "found " << model_elements.size() << " model elements" << " will run for " << op_iter_count << " iterations" <<std::endl;

  if( nlhs != model_elements.size() ){
    std::cout << "mct: this factorization requires " << model_elements.size() << " number of output arguments, given: " << nlhs << std::endl;
    // print help;
    return;
  }


  // prepare cards_numeric indices of model elements
  // input arrives like so:
  // A['i','k'], B['k', 'j'], C['i','j'] where V is ['i','k','j'] = [2 3 4]
  // here we convert indices to internal format:
  // A[2, 3, 0], B[0, 3, 4], C[2, 0, 4]
  mxChar* V_char = mxGetChars(prhs[1]);
  size_t ndims = mxGetNumberOfElements(prhs[1]);
  double* V_numeric = (double*) mxGetData(prhs[2]);

  for (size_t m=0; m<model_elements.size(); m++){
    assign_m_tensor_cards_numeric(&(model_elements[m]), V_char, V_numeric, ndims);
  }
  assign_m_tensor_cards_numeric(&x_tensor, V_char, V_numeric, ndims);

  if(COUT) print_model_elements(&model_elements, &x_tensor);

  // now all tensors have correct internal cardinalities.
  // all numeric cardinality arrays (m_tensor.char_numeric) are of same size as V
  // -> ndims


  // more input sanity check may be nice



  // prepare output tensor in matlab  //////////////////////////////////////////////////////

  std::vector<double*> output_data_ptr;


  for (size_t t=0; t<model_elements.size(); t++){
    // size_t non_zero_dims=0;
    // for (size_t i=0; i<ndims; i++) {
    //   if ( model_elements[t].cards_numeric[i] != 0){
    //  non_zero_dims++;
    //   }
    // }

    //mwSize argMatDims[non_zero_dims];
    //size_t j=0;
    mwSize argMatDims[ndims];
    for (size_t i=0; i<ndims; i++) {
      size_t val = model_elements[t].cards_numeric[i];
      if (val == 0) argMatDims[i] = 1; // MATLAB needs to get 1 instead of 0
      else          argMatDims[i] = val;
      // if (val != 0){
      //        argMatDims[j] = val;
      //        j++;
      // }
    }

    //std::cout << " mxCreateNumericArray dimensions argMatDims ";
    //for (size_t i=0; i<ndims; i++) std::cout << " " << argMatDims[i];
    //std::cout << std::endl;

    plhs[t] = mxCreateNumericArray(ndims, argMatDims, mxDOUBLE_CLASS, mxREAL);
    //plhs[t] = mxCreateNumericArray(non_zero_dims, argMatDims, mxDOUBLE_CLASS, mxREAL);

    output_data_ptr.push_back( (double*) mxGetPr(plhs[t]) );
  }


  // prepare host memory for tensors  ///////////////////////////////////////////////////////

  h_full_cardinalities = (size_t*) calloc(ndims, sizeof(size_t)); // defined in mct_tensorop_utils.cuh

  ///// cards_numeric are alligned according to the V cardinalities ///// above //
  for (size_t dim=0; dim<ndims; dim++){ // for each dimension
    size_t max_dim_card = 0;

    for (size_t t=0; t<model_elements.size(); t++){ // for each model
      for (size_t card=0; card<strlen(model_elements[t].cards_char); card++){ // for each dimension of the model
        if (model_elements[t].cards_char[card] == V_char[dim]){ // if this dimension character matches current dimension's
          size_t tensor_dim_card = model_elements[t].cards_numeric[dim]; //see above//
          if ( max_dim_card < tensor_dim_card )
            max_dim_card = tensor_dim_card;
          break; // only one dimension of each model can match with current dimension
        }
      }
    }

    // also check X tensor
    for (size_t card=0; card<strlen(x_tensor.cards_char); card++){ // for each dimension of the X tensor
      if (x_tensor.cards_char[card] == V_char[dim]){ // if this dimension matches current dimension
        size_t tensor_dim_card = x_tensor.cards_numeric[dim] ;
        if ( max_dim_card < tensor_dim_card )
          max_dim_card = tensor_dim_card;
        break; // only one dimension of X tensor can match with current dimension
      }
    }

    h_full_cardinalities[dim] = max_dim_card;
  }


  if(COUT)
    for (int i=0; i<ndims; i++)
      std::cout << "h_full_cardinalities " << i << " " << h_full_cardinalities[i] << std::endl;


  // initialize random seed for random initialization of objects
  //srand((unsigned)time(NULL));
  srand(123);


  std::vector<ct> Z_tensors;
  std::vector<ct> D_tensors;
  for (size_t t=0; t<model_elements.size(); t++){
    ct tmp_ct;
    ct tmp_ct_D1;
    ct tmp_ct_D2;

    size_t Z_card[ndims];
    for (size_t i=0; i<ndims; i++) Z_card[i] = model_elements[t].cards_numeric[i];

    std::stringstream z;
    z << "Host Z" << t;
    prepareHostTensorFromCpp(&tmp_ct, model_elements[t].data, Z_card, ndims, z.str().c_str(), true); // init with given data, if null init with rand

    std::stringstream d1;
    d1 << "Host D1_Z" << t;
    prepareHostTensorFromCpp(&tmp_ct_D1, NULL, Z_card, ndims, d1.str().c_str());
    std::stringstream d2;
    d2 << "Host D2_Z" << t;
    prepareHostTensorFromCpp(&tmp_ct_D2, NULL, Z_card, ndims, d2.str().c_str());


    Z_tensors.push_back(tmp_ct);
    D_tensors.push_back(tmp_ct_D1);
    D_tensors.push_back(tmp_ct_D2);
  }



  ct Xhat, X, A, F, M, Fzeros, Fones;

  mwSize ndims_mwsize = ndims;
  mxArray* m_X_card = mxCreateNumericArray(1, &ndims_mwsize, mxDOUBLE_CLASS, mxREAL);

  size_t X_card[ndims];
  for (size_t i=0; i<ndims; i++){
    X_card[i] = x_tensor.cards_numeric[i];
    mxGetPr(m_X_card)[i] = X_card[i];
  }


  prepareHostTensorFromCpp(&X, mxGetPr(x_tensor_data), X_card, ndims, (const char*) "Host X");

  double M_data[X.mem_size];
  for(size_t i=0; i<X.mem_size; i++) M_data[i]=1;
  prepareHostTensorFromCpp(&M, M_data, X_card, ndims, (const char*) "Host M");

  prepareHostTensorFromCpp(&A, NULL, X_card, ndims, (const char*) "Host A"); // init with 0
  prepareHostTensorFromCpp(&Xhat, NULL, X_card, ndims, (const char*) "Host Xhat");
  prepareHostTensorFromCpp(&F, NULL, h_full_cardinalities, ndims, "Host F");
  prepareHostTensorFromCpp(&Fzeros, NULL, h_full_cardinalities, ndims, (const char*) "Host Fzeros");
  prepareHostTensorFromCpp(&Fones, NULL, h_full_cardinalities, ndims, (const char*) "Host Fones", false, true);


  /*
    if(PRINT_CT){
    print_ct("random Z1 init", &Z1, true);
    print_ct("random Z2 init", &Z2, true);
    print_ct("target X (cpp side)", &X, true);
    print_ct("F (cpp side)", &F, true);
    }
  */

  ///////////////////////////////////////////////////////////////////////////////////////////


  // register & transfer objects to device //////////////////////////////////////////////////

  for (size_t z=0; z<Z_tensors.size(); z++){
    std::stringstream name;
    name << 'Z' << z;
    register_ct( name.str().c_str(), &(Z_tensors[z]) );

    std::stringstream d_name1;
    d_name1 << "D1_Z" << z;
    std::cout << "registering: "<< d_name1.str() << "with ct"<< std::endl;
    print_ct("ct:" ,  &D_tensors[z*2], true);
    register_ct( d_name1.str().c_str(), &D_tensors[z*2]);

    std::stringstream d_name2;
    d_name2 << "D2_Z" << z;
    register_ct( d_name2.str().c_str(), &D_tensors[z*2+1]);
  }

  REGISTER_CT(Xhat); REGISTER_CT(Fzeros); REGISTER_CT(Fones); REGISTER_CT(X); REGISTER_CT(A); REGISTER_CT(M); REGISTER_CT(F);

  if (CUPRINTF == true)
    cudaPrintfInit();

  if (is_parallel)
    transferToDevice(ndims);

  ///////////////////////////////////////////////////////////////////////////////////////////


  // perform PLTF operation //////////////////////////////////////////////////////////////////


  // operation arguments
  operands ops_Z0_ZN_Xhat;
  std::vector<std::string> z_tensors_str;
  for (size_t z=0; z<Z_tensors.size(); z++){
    std::stringstream name;
    name << 'Z' << z;
    z_tensors_str.push_back(name.str());
  }
  gen_operation_arguments(z_tensors_str, &ops_Z0_ZN_Xhat);

  std::vector<operands> ops_A_otherZ_D1_Z;
  std::vector<operands> ops_M_otherZ_D2_Z;
  for( size_t z=0; z<Z_tensors.size(); z++ ){
    std::cout << " process z " << z << std::endl;
    operands ops_A;
    operands ops_M;

    std::vector<std::string> tmp_A;
    std::vector<std::string> tmp_M;
    
    tmp_A.push_back(std::string("A"));
    tmp_M.push_back(std::string("M"));

    for( size_t other_Z=0; other_Z<Z_tensors.size(); other_Z++ ){
      if( other_Z == z ){
	std::cout << "skip other_Z " << other_Z << std::endl; 
	continue;
      }else{
	std::stringstream name;
	name << 'Z' << other_Z;

	tmp_A.push_back(name.str());
	tmp_M.push_back(name.str());
	std::cout << "pushing to tmp_A and tmp_M: " << other_Z  << std::endl;
      }
    }

    std::cout << " ops_A_otherZ_D1_Z" << z << " operands ";
    for(size_t i=0; i<tmp_A.size(); i++)
      std::cout << tmp_A[i] << " ";
    std::cout << std::endl;
    gen_operation_arguments(tmp_A, &ops_A);
    ops_A_otherZ_D1_Z.push_back(ops_A);

    std::cout << " ops_M_otherZ_D2_Z" << z << " operands ";
    for(size_t i=0; i<tmp_A.size(); i++)
      std::cout << tmp_M[i] << " ";
    std::cout << std::endl;
    gen_operation_arguments(tmp_M, &ops_M);
    ops_M_otherZ_D2_Z.push_back(ops_M);
  }


  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  if ( COUT ) std::cout << " tensorop_gpu Running kernels " << std::endl << std::endl;



  for (int iter=0; iter<op_iter_count; iter++){

    if (COUT) std::cout << "iteration number "<< iter << std::endl;
    
    //for all ZN
    for (size_t cur_z=0; cur_z<Z_tensors.size(); cur_z++){
      //size_t z=0;

      std::stringstream d1;
      d1 << "D1_Z" << cur_z;
      std::stringstream d2;
      d2 << "D2_Z" << cur_z;
      std::stringstream zname;
      zname << "Z" << cur_z;

      // Z0 * Z1 * ... * ZN -> Xhat
      std::cout << "Z0 * Z1 * ... * ZN -> Xhat" << std::endl;

      calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                          (size_t) (Z_tensors.size()),

                                                          (size_t**) (ops_Z0_ZN_Xhat.d_strides_operand_pointers),

                                                          (size_t*) (get_d_obj_strides()["Xhat"]),
                                                          (size_t*) (get_d_obj_cards()["F"]),

                                                          (size_t**) (ops_Z0_ZN_Xhat.d_cards_operand_pointers),
                                                          (double**) (ops_Z0_ZN_Xhat.d_operand_pointers),

                                                          (double*) (get_d_obj_data()["Xhat"]),
                                                          //(double*) (get_d_obj_data()["Z0"]),
                                                          (size_t) (Xhat.element_number),
                                                          (size_t) 1,
                                                          CUPRINTF,1);
      // transferFromDevice(h_objs["Xhat"]->data, "Xhat");
      // print_ct("hello 1 ", h_objs["Xhat"], true);



      // // h: X / Xhat -> A
      std::cout << "h: X / Xhat -> A" << std::endl;
      hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(get_d_obj_data()["X"],
                                                      get_d_obj_data()["Xhat"],
                                                      get_d_obj_data()["A"],
                                                      h_objs["A"]->element_number,
                                                      CUPRINTF);

      // transferFromDevice(h_objs["A"]->data, "A");
      // print_ct("hello 2 ", h_objs["A"], true);

      //,
      //to_power_A,
      //to_power_B);


      //   h: M * A -> A
      
      //   A * other_Z  -> D1_Z1      
      std::cout << "A*other_Z -> " << d1.str().c_str() << " cur_z " << cur_z << std::endl;

      for( size_t i=0; i<ndims; i++)
      	std::cout << h_objs[d1.str().c_str()]->strides[i];
      std::cout << std::endl;
      calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                          (size_t) (Z_tensors.size()),

                                                          (size_t**) (ops_A_otherZ_D1_Z[cur_z].d_strides_operand_pointers),

                                                          (size_t*) (get_d_obj_strides()[d1.str().c_str()]),
                                                          (size_t*) (get_d_obj_cards()["F"]),

                                                          (size_t**) (ops_A_otherZ_D1_Z[cur_z].d_cards_operand_pointers),
                                                          (double**) (ops_A_otherZ_D1_Z[cur_z].d_operand_pointers),

                                                          (double*) (get_d_obj_data()[d1.str().c_str()]),
                                                          //(double*) (get_d_obj_data()["Z0"]),
                                                          (size_t) (D_tensors[cur_z*2].element_number),
                                                          (size_t) 1,
                                                          CUPRINTF,2);

      // transferFromDevice(h_objs[d1.str().c_str()]->data, d1.str().c_str());
      // print_ct("hello 3 ", h_objs[d1.str().c_str()], true);

      
      //   M * other_Z -> D2_Z1
      std::cout << "M*other_Z -> " << d2.str().c_str() << std::endl;
      calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                          (size_t) (Z_tensors.size()),

                                                          (size_t**) (ops_M_otherZ_D2_Z[cur_z].d_strides_operand_pointers),

                                                          (size_t*) (get_d_obj_strides()[d2.str().c_str()]),
                                                          (size_t*) (get_d_obj_cards()["F"]),

                                                          (size_t**) (ops_M_otherZ_D2_Z[cur_z].d_cards_operand_pointers),
                                                          (double**) (ops_M_otherZ_D2_Z[cur_z].d_operand_pointers),

                                                          (double*) (get_d_obj_data()[d2.str().c_str()]),
                                                          //(double*) (get_d_obj_data()["Z0"]),
                                                          (size_t) (D_tensors[cur_z*2+1].element_number),
                                                          (size_t) 1,
                                                          CUPRINTF,3);

      // transferFromDevice(h_objs[d2.str().c_str()]->data, d2.str().c_str());
      // print_ct("hello 4 ", h_objs[d2.str().c_str()], true);


      //   h: D1_Z1 / D2_Z1 -> D1_Z1
      hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(get_d_obj_data()[d1.str().c_str()],
      						      get_d_obj_data()[d2.str().c_str()],
      						      get_d_obj_data()[d1.str().c_str()],
      						      h_objs[d1.str().c_str()]->element_number,
      						      CUPRINTF);

      // transferFromDevice(h_objs[d1.str().c_str()]->data, d1.str().c_str());
      // print_ct("hello 5 ", h_objs[d1.str().c_str()], true);

      //   h: Z1 * D1_Z1 -> Z1
      hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(get_d_obj_data()[zname.str().c_str()],
      						      get_d_obj_data()[d1.str().c_str()],
      						      get_d_obj_data()[zname.str().c_str()],
      						      h_objs[zname.str().c_str()]->element_number,
      						      CUPRINTF);
      // transferFromDevice(h_objs[zname.str().c_str()]->data, zname.str().c_str());
      // print_ct("hello 6 ", h_objs[zname.str().c_str()], true);

    }
  }





  // if ( PRINT_CT ) {
  //   transferFromDevice(h_objs["Xhat"]->data, "Xhat");
  //   print_ct("tensorop gpu Xhat ", h_objs["Xhat"], true);
  // }


  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaDeviceSynchronize();

  if ( CUPRINTF == true ){
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
  }






  ///////////////////////////////////////////////////////////////////////////////////////////

  // transfer results to matlab /////////////////////////////////////////////////////////////

  if ( is_parallel ){
    for (size_t z=0; z<model_elements.size(); z++){
      std::stringstream Zn;
      Zn << 'Z' << z;
      transferFromDevice(output_data_ptr[z], Zn.str().c_str());
    }
    //transferFromDevice(m_Z1, "Z1");
    //transferFromDevice(m_Z2, "Z2");
  }else{
    for (size_t z=0; z<model_elements.size(); z++){
      memcpy(output_data_ptr[z], Z_tensors[z].data, Z_tensors[z].mem_size);
    }
    //memcpy(m_Z1, Z1.data, Z1.mem_size);
    //memcpy(m_Z2, Z2.data, Z2.mem_size);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////


  // reset device
  if ( is_parallel )
    resetDevice();

  cudaThreadExit();

  ///////////////////////////////////////////////////////////////////////////////////////////


  /*

    std::vector<operation> operation_chain;

    for ( size_t t=0; t<model_elements.size(); t++){
    // zN update

    // Z0 * Z1 -> Xhat
    // Z2 * Xhat -> Xhat
    // ...
    //oc_push_back(&operation_chain, false, 1, ndims, "Z1", "Z2", "Xhat", is_parallel);


    for (size_t z=0; z<Z_tensors.size(); z++){
    if (z==0) continue;
    if (z==1) {
    // if this is the last operation (2 factor problem) store result in Xhat
    if ( Z_tensors.size() == 2 ){
    oc_push_back(&operation_chain, GMULT, ndims, "Z0", "Z1", "Xhat", is_parallel);
    }else{
    //oc_push_back(&operation_chain, true, 1, ndims, "Fzeros", "Fzeros", "F", is_parallel); // reset F to zero
    oc_push_back(&operation_chain, GMULT, ndims, "Z0", "Z1", "F", is_parallel);
    }
    }
    else{
    std::stringstream Zn;
    Zn << 'Z' << z;

    if (z!=(Z_tensors.size()-1)){
    // in all non last operations store result in F to avoid mis-contraction
    oc_push_back(&operation_chain, GMULT, ndims, Zn.str().c_str(), "F", "F", is_parallel);
    }else{
    // in last operation store result into Xhat
    oc_push_back(&operation_chain, GMULT, ndims, Zn.str().c_str(), "F", "Xhat", is_parallel);
    }
    }
    }


    oc_push_back(&operation_chain, HADAMARD_DIV, ndims, "X", "Xhat", "A", is_parallel);
    oc_push_back(&operation_chain, HADAMARD_MUL, ndims, "M", "A", "A", is_parallel);

    std::stringstream d1;
    d1 << "D1_Z" << t;
    size_t tmp_op_count=0;
    for (size_t other_z=0; other_z < Z_tensors.size(); other_z++){
    if (other_z == t) continue;

    std::stringstream other_z_name;
    other_z_name << "Z" << other_z;

    if ( tmp_op_count == 0 ){
    if ( Z_tensors.size() == 2){
    oc_push_back(&operation_chain, GMULT, ndims, "A", other_z_name.str().c_str(), d1.str().c_str(), is_parallel);
    }else{
    //oc_push_back(&operation_chain, true, 1, ndims, "Fzeros", "Fzeros", "F", is_parallel); // reset F to zero
    oc_push_back(&operation_chain, GMULT, ndims, "A", other_z_name.str().c_str(), "F", is_parallel);
    }
    }else{
    if (tmp_op_count!=Z_tensors.size()-2){ // -1 for index starts from 0 -1 for Zn itself does not loop
    // in all non last operations store result in F to avoid mis-contraction
    oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", "F", is_parallel);
    }else{
    // in last operation store result into d1
    oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", d1.str().c_str(), is_parallel);
    }
    }
    tmp_op_count++;
    }

    std::stringstream d2;
    d2 << "D2_Z" << t;
    tmp_op_count=0;
    for (size_t other_z=0; other_z < Z_tensors.size(); other_z++){
    if (other_z == t) continue;

    std::stringstream other_z_name;
    other_z_name << "Z" << other_z;

    if ( tmp_op_count == 0 ){
    if ( Z_tensors.size() == 2) {
    oc_push_back(&operation_chain, GMULT, ndims, "M", other_z_name.str().c_str(), d2.str().c_str(), is_parallel);
    }else{
    //oc_push_back(&operation_chain, true, 1, ndims, "Fzeros", "Fzeros", "F", is_parallel); // reset F to zero
    oc_push_back(&operation_chain, GMULT, ndims, "M", other_z_name.str().c_str(), "F", is_parallel);
    }
    }else{
    if (tmp_op_count!=Z_tensors.size()-2){ // -1 for index starts from 0 -1 for Zn itself does not loop
    // in all non last operations store result in F to avoid mis-contraction
    oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", "F", is_parallel);
    }else{
    // in last operation store result into d2
    oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", d2.str().c_str(), is_parallel);
    }
    }
    tmp_op_count++;
    }

    oc_push_back(&operation_chain, HADAMARD_DIV, ndims, d1.str().c_str(), d2.str().c_str(), d1.str().c_str(), is_parallel);

    std::stringstream Zn;
    Zn << 'Z' << t ;
    oc_push_back(&operation_chain, HADAMARD_MUL, ndims, Zn.str().c_str(), d1.str().c_str(), Zn.str().c_str(), is_parallel);
    }



    if (PRINT_CHAIN) print_oc(&operation_chain);


    for (int iter=0; iter<op_iter_count; iter++){

    if (COUT) std::cout << "iteration number "<< iter << std::endl;
    //if (is_parallel == nmf_gpu)
    operate(&operation_chain);

    //////////////////////////////////////////////////////////////////////////////////



    //    if (iter % 10 == 0 || iter==99 || iter == 98){
    // std::cout << "iter " << iter << std::endl;
    // if (is_parallel == nmf_gpu) transferFromDevice(Z1.data, "Z1");
    // print_ct("current Z1", &Z1, true);

    // if (is_parallel == nmf_gpu) transferFromDevice(Z2.data, "Z2");
    // print_ct("current Z2", &Z2, true);
    //}

    }
  */


}
