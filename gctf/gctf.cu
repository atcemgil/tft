/*
 * author: ck
 * created: 16.02.2012
 * advisor: atc
 */

#include <string.h>
#include <sstream>

#include "../common/utils.cuh"



#include "cutil_inline.h"
#include "../common/kernels.cuh"
#include "../common/cuPrintf.cuh"



void gctf(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool is_parallel){
  // prepare model elements  //////////////////////////////////////////////////////

  size_t op_iter_count = ((double *)mxGetData(prhs[0]))[0];
  mxChar* V_card_sym = mxGetChars(prhs[1]);
  size_t ndims = mxGetNumberOfElements(prhs[1]);
  double* V_cards = (double*) mxGetData(prhs[2]);
  size_t p = ((double *)mxGetData(prhs[3]))[0];


  size_t max_v = mxGetM(prhs[4]);
  size_t max_alpha = mxGetN(prhs[4]);
  bool* R = (bool*) malloc( sizeof(bool) * max_v * max_alpha); // dynamic allocation may not be initialized with = {0} syntax
  if(COUT) std::cout << "init R" << std::endl;
  for (size_t i=0; i<max_v; i++){
    for (size_t j=0; j<max_alpha; j++){
      R[i + j*max_v] = (bool) (((double *)mxGetData(prhs[4]))[i + j*max_v]);
      if(COUT) std::cout << R[i + j*max_v] << " ";
    }
    if(COUT) std::cout << std::endl;
  }

  std::vector<m_tensor> observed_elements;
  //size_t observed_element_num = max_v;
  size_t m_index=5;
  for (size_t t=0; t<max_v; t++){ // need to fill in v number of observed_elements
    const mxArray* m_observed_cards = prhs[m_index];
    m_index++;
    const mxArray* m_observed_data = prhs[m_index];
    m_index++;

    m_tensor tmp_m_tensor;
    tmp_m_tensor.is_updateable = false; // used with latent tensors only

    size_t m_observed_cards_elnum = mxGetNumberOfElements(m_observed_cards);
    tmp_m_tensor.cards_char = (char*) malloc( m_observed_cards_elnum + 1 );
    for (size_t i=0; i<=m_observed_cards_elnum ; i++)
      if ( i == m_observed_cards_elnum )
        tmp_m_tensor.cards_char[i] = '\0';
      else
        tmp_m_tensor.cards_char[i] = (char) mxGetChars(m_observed_cards)[i] ;

    if ( mxGetNumberOfElements(m_observed_data) == 0 ){
      // tensor init data is not given
      tmp_m_tensor.data = NULL;
    }else{
      // tensor init data is given, save pointer
      tmp_m_tensor.data = (double*) mxGetData(m_observed_data);
      if (COUT) std::cout << "found factor with init data. Data size " << mxGetNumberOfElements(m_observed_data) << std::endl;
    }

    observed_elements.push_back(tmp_m_tensor);
  }


  std::vector<m_tensor> latent_elements;
  //size_t latent_element_num = max_v;
  for (size_t t=0; t<max_alpha; t++){ // need to fill in alpha number of latent_elements
    const mxArray* m_latent_cards = prhs[m_index];
    m_index++;
    const mxArray* m_latent_data = prhs[m_index];
    m_index++;

    m_tensor tmp_m_tensor;
    tmp_m_tensor.is_updateable = (bool) (((double *)mxGetData(prhs[m_index]))[0]);
    m_index++;

    size_t m_latent_cards_elnum = mxGetNumberOfElements(m_latent_cards);
    tmp_m_tensor.cards_char = (char*) malloc( m_latent_cards_elnum + 1 );
    for (size_t i=0; i<=m_latent_cards_elnum ; i++)
      if ( i == m_latent_cards_elnum )
        tmp_m_tensor.cards_char[i] = '\0';
      else
        tmp_m_tensor.cards_char[i] = (char) mxGetChars(m_latent_cards)[i] ;

    if ( mxGetNumberOfElements(m_latent_data) == 0 ){
      // tensor init data is not given
      tmp_m_tensor.data = NULL;
    }else{
      // tensor init data is given, save pointer
      tmp_m_tensor.data = (double*) mxGetData(m_latent_data);
      if (COUT) std::cout << "found factor with init data. Data size " << mxGetNumberOfElements(m_latent_data) << std::endl;
    }

    latent_elements.push_back(tmp_m_tensor);
  }


  // prepare cards_numeric indices of model elements
  // input arrives like so:
  // A['i','k'], B['k', 'j'], C['i','j'] where V is ['i','k','j'] = [2 3 4]
  // here we convert indices to internal format:
  // A[2, 3, 0], B[0, 3, 4], C[2, 0, 4]
  for (size_t m=0; m<observed_elements.size(); m++){
    assign_m_tensor_cards_numeric(&(observed_elements[m]), V_card_sym, V_cards, ndims);
  }
  for (size_t m=0; m<latent_elements.size(); m++){
    assign_m_tensor_cards_numeric(&(latent_elements[m]), V_card_sym, V_cards, ndims);
  }


  if (COUT) {
    print_model_elements_text(&observed_elements, "printing observed model elements");
    print_model_elements_text(&latent_elements, "printing latent model elements");
  }

  // now all tensors have correct internal cardinalities.
  // all numeric cardinality arrays (m_tensor.char_numeric) are of same size as V
  // -> ndims





  // prepare output tensor in matlab  //////////////////////////////////////////////////////

  std::vector<double*> output_data_ptr;


  for (size_t t=0; t<latent_elements.size(); t++){
    mwSize argMatDims[ndims];
    for (size_t i=0; i<ndims; i++) {
      size_t val = latent_elements[t].cards_numeric[i];
      if (val == 0) argMatDims[i] = 1; // MATLAB needs to get 1 instead of 0
      else          argMatDims[i] = val;
    }

    plhs[t] = mxCreateNumericArray(ndims, argMatDims, mxDOUBLE_CLASS, mxREAL);

    output_data_ptr.push_back( (double*) mxGetPr(plhs[t]) );
  }




  // prepare host memory for tensors  ///////////////////////////////////////////////////////

  h_full_cardinalities = (size_t*) calloc(ndims, sizeof(size_t)); // defined in mct_tensorop_utils.cuh

  ///// cards_numeric are alligned according to the V cardinalities ///// above //
  for (size_t dim=0; dim<ndims; dim++){ // for each dimension
    size_t max_dim_card = 0;

    for (size_t t=0; t<observed_elements.size(); t++){ // for each model
      for (size_t card=0; card<strlen(observed_elements[t].cards_char); card++){ // for each dimension of the model
        if (observed_elements[t].cards_char[card] == V_card_sym[dim]){ // if this dimension character matches current dimension's
          size_t tensor_dim_card = observed_elements[t].cards_numeric[dim]; //see above//
          if ( max_dim_card < tensor_dim_card )
            max_dim_card = tensor_dim_card;
          break; // only one dimension of each model can match with current dimension
        }
      }
    }

    for (size_t t=0; t<latent_elements.size(); t++){ // for each model
      for (size_t card=0; card<strlen(latent_elements[t].cards_char); card++){ // for each dimension of the model
        if (latent_elements[t].cards_char[card] == V_card_sym[dim]){ // if this dimension character matches current dimension's
          size_t tensor_dim_card = latent_elements[t].cards_numeric[dim]; //see above//
          if ( max_dim_card < tensor_dim_card )
            max_dim_card = tensor_dim_card;
          break; // only one dimension of each model can match with current dimension
        }
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

  std::vector<ct> X_tensors;
  std::vector<ct> A_tensors;
  std::vector<ct> hat_X_tensors;

  for (size_t el=0; el<observed_elements.size(); el++){
    ct tmp_ct_X;
    ct tmp_ct_A;
    ct tmp_ct_hatX;

    size_t X_card[ndims];
    for (size_t i=0; i<ndims; i++) X_card[i] = observed_elements[el].cards_numeric[i];

    std::stringstream x, hatx, xa;
    x << "Host X" << el;
    hatx << "Host hatX" << el;
    xa << "Host A_X" << el;
    prepareHostTensorFromCpp(&tmp_ct_X, observed_elements[el].data, X_card, ndims, x.str().c_str(), true); // init with given data, if null init with rand
    prepareHostTensorFromCpp(&tmp_ct_A, NULL, X_card, ndims, xa.str().c_str(), false, true); // rand=false, init_to_one=true -> init with 1
    prepareHostTensorFromCpp(&tmp_ct_hatX, NULL, X_card, ndims, hatx.str().c_str(), true);

    X_tensors.push_back(tmp_ct_X);
    A_tensors.push_back(tmp_ct_A);
    hat_X_tensors.push_back(tmp_ct_hatX);
  }


  std::vector<ct> Z_tensors;
  std::vector<ct> Z_update_tensors; // stores each one of sub-update equation results, then all are summed together
  std::vector<ct> D_tensors;
  for (size_t el=0; el<latent_elements.size(); el++){
    ct tmp_ct;

    size_t Z_card[ndims];
    for (size_t i=0; i<ndims; i++) Z_card[i] = latent_elements[el].cards_numeric[i];
    std::stringstream z;
    z << "Host Z" << el;
    prepareHostTensorFromCpp(&tmp_ct, latent_elements[el].data, Z_card, ndims, z.str().c_str(), true); // init with given data, if null init with rand
    Z_tensors.push_back(tmp_ct);

    for (size_t v=0; v<max_v; v++){
      ct tmp_ct_D1;
      ct tmp_ct_D2;

      std::stringstream d1;
      d1 << "Host D1_Z" << el << "X" << v;
      prepareHostTensorFromCpp(&tmp_ct_D1, NULL, Z_card, ndims, d1.str().c_str());

      std::stringstream d2;
      d2 << "Host D2_Z" << el << "X" << v;
      prepareHostTensorFromCpp(&tmp_ct_D2, NULL, Z_card, ndims, d2.str().c_str());

      D_tensors.push_back(tmp_ct_D1);
      D_tensors.push_back(tmp_ct_D2);

      // ct tmp_ct_update;
      // std::stringstream z_update;
      // z_update << "Host Z_update" << el << "X" << v;
      // prepareHostTensorFromCpp(&tmp_ct_update, NULL, Z_card, ndims, z_update.str().c_str(), false, false); // rand=false, init_to_one=false -> init with 0
      // Z_update_tensors.push_back(tmp_ct_update);
    }

    // for summation of division operands
    // ct tmp_ct_D1_sum;
    // ct tmp_ct_D2_sum;
    // std::stringstream d1;
    // d1 << "Host D1_Z" << el << "sum";
    // prepareHostTensorFromCpp(&tmp_ct_D1_sum, NULL, Z_card, ndims, d1.str().c_str());

    // std::stringstream d2;
    // d2 << "Host D2_Z" << el << "sum";
    // prepareHostTensorFromCpp(&tmp_ct_D2_sum, NULL, Z_card, ndims, d2.str().c_str());

    // D_tensors.push_back(tmp_ct_D1_sum);
    // D_tensors.push_back(tmp_ct_D2_sum);
  }


  ct F;
  prepareHostTensorFromCpp(&F, NULL, h_full_cardinalities, ndims, "Host F", true, true, false);


  ///////////////////////////////////////////////////////////////////////////////////////////

  // register & transfer objects to device //////////////////////////////////////////////////

  size_t k=0;
  for (size_t alpha=0; alpha<max_alpha; alpha++){
    for (size_t v=0; v<max_v; v++){
      std::stringstream d_name1;
      d_name1 << "D1_Z" << alpha << "X" << v;
      register_ct( d_name1.str().c_str(), &D_tensors[k]);
      k++;

      std::stringstream d_name2;
      d_name2 << "D2_Z" << alpha << "X" << v;
      register_ct( d_name2.str().c_str(), &D_tensors[k]);
      k++;

      //     std::stringstream name_update;
      //     name_update << "Zup" << alpha << "X" << v;
      //     register_ct( name_update.str().c_str(), &(Z_update_tensors[k]) );
    }

    // std::stringstream d_name1, d_name2;
    // d_name1 << "D1_Z" << alpha << "sum";
    // d_name2 << "D2_Z" << alpha << "sum";
    // register_ct( d_name2.str().c_str(), &D_tensors[k]);
    // k++;
    // register_ct( d_name2.str().c_str(), &D_tensors[k]);
    // k++;
  }

  for (size_t z=0; z<Z_tensors.size(); z++){
    std::stringstream name;
    name << 'Z' << z;
    register_ct( name.str().c_str(), &(Z_tensors[z]) );
  }


  for (size_t x=0; x<X_tensors.size(); x++){
    std::stringstream name;
    name << "X" << x;
    register_ct( name.str().c_str(), &(X_tensors[x]) );

    std::stringstream a_name;
    a_name << "A" << x;
    register_ct( a_name.str().c_str(), &(A_tensors[x]) );

    std::stringstream hat_X_name;
    hat_X_name << "hatX" << x;
    register_ct( hat_X_name.str().c_str(), &(hat_X_tensors[x]) );
  }


  REGISTER_CT(F);

  if (CUPRINTF == true)
    cudaPrintfInit();


  size_t cur_mem;
  if (is_parallel)
    cur_mem = transferToDevice(ndims);

  if( COUT ) std::cout << "transferToDevice " << cur_mem << " bytes " << std::endl;
  ///////////////////////////////////////////////////////////////////////////////////////////

  // perform GCTF operation //////////////////////////////////////////////////////////////////


  for (int iter=0; iter<op_iter_count; iter++){
    std::cout << "iter " << iter << std::endl;



    // calculate all hatX_v and A_v
    for (size_t alpha=0; alpha<max_alpha; alpha++){
      if ( latent_elements[alpha].is_updateable == false) continue;



      // update all hatX
      for ( size_t cur_v=0; cur_v<max_v; cur_v++){
        std::stringstream hat_Xv;
        hat_Xv << "hatX" << cur_v;


        operands ops_Z0_ZN_Xhat;
        std::vector<std::string> z_tensors_str;
        for (size_t tmp_alpha=0; tmp_alpha<max_alpha; tmp_alpha++){
          if ( R[cur_v + tmp_alpha*max_v] == false )  continue;

          std::stringstream name;
          name << 'Z' << tmp_alpha;
          z_tensors_str.push_back(name.str());
        }

	std::cout << "operand num z_tensors_str.size() " << z_tensors_str.size() << std::endl;
	for( size_t i=0; i<z_tensors_str.size(); i++){
	  std::cout << "z_tensors_str[" << i << "] = " << z_tensors_str[i] << std::endl;
	}
	
        cur_mem = gen_operation_arguments(z_tensors_str, &ops_Z0_ZN_Xhat, cur_mem);

        // Z0 * Z1 * ... * ZN -> Xhat
        //std::cout << "Z0 * Z1 * ... * ZN -> " << hat_Xv.str() << std::endl;
        calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                            (size_t) (z_tensors_str.size()),

                                                            (size_t**) (ops_Z0_ZN_Xhat.d_strides_operand_pointers),

                                                            (size_t*) (get_d_obj_strides()[hat_Xv.str()]),
                                                            (size_t*) (get_d_obj_cards()["F"]),

                                                            (size_t**) (ops_Z0_ZN_Xhat.d_cards_operand_pointers),
                                                            (double**) (ops_Z0_ZN_Xhat.d_operand_pointers),

                                                            (double*) (get_d_obj_data()[hat_Xv.str()]),
                                                            //(double*) (get_d_obj_data()["Z0"]),
                                                            (size_t) (h_objs[hat_Xv.str()]->element_number),
                                                            (size_t) 1,
                                                            CUPRINTF,1);

	////std::cout << " z0 * z1 * .. * zn -> " << hat_Xv << " done " << std::endl;
        std::stringstream Xv;
        Xv << 'X' << cur_v ;

        std::stringstream Av;
        Av << 'A' << cur_v;

        hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>( get_d_obj_data()[hat_Xv.str().c_str()],
                                                         get_d_obj_data()[Xv.str().c_str()],
                                                         get_d_obj_data()[Av.str().c_str()],
                                                         h_objs[Av.str().c_str()]->element_number,
                                                         CUPRINTF,
                                                         p, 1);
      }
      return;








      // for each Xv
      for (size_t cur_v=0; cur_v<max_v; cur_v++){
        if ( R[cur_v + alpha*max_v] == false ) continue; // if this Xv does not have this Zalpha dothing to do

        std::stringstream hat_Xv;
        hat_Xv << "hatX" << cur_v;

        // calculate D1_Zalpha_Xv
        std::stringstream d1;
        d1 << "D1_Z" << alpha << "X" << cur_v;

        // calculate D2_Zalpha_Xv
        std::stringstream d2;
        d2 << "D2_Z" << alpha << "X" << cur_v;

        std::stringstream Av;
        Av << 'A' << cur_v;


        operands ops_A;
        operands ops_M;


        std::vector<std::string> tmp_A;
        std::vector<std::string> tmp_M;

        tmp_A.push_back(Av.str());
        tmp_M.push_back(hat_Xv.str());


        for (size_t other_z=0; other_z < max_alpha; other_z++){
          //std::cout << " process alpha " << alpha << " other_z " << other_z << std::endl;
          if (other_z == alpha || R[cur_v + other_z*max_v] == false ) continue;

          std::stringstream other_z_name;
          other_z_name << "Z" << other_z;

          tmp_A.push_back(other_z_name.str());
          tmp_M.push_back(other_z_name.str());
          //std::cout << "pushing to tmp_A and tmp_M: " << other_z_name.str()  << std::endl;

        }

	//std::cout << "operand num tmp_A.size() " << tmp_A.size() << std::endl;
	for( size_t i=0; i<tmp_A.size(); i++){
	}


        cur_mem = gen_operation_arguments(tmp_A, &ops_A, cur_mem);

        //oc_push_back(&operation_chain, GMULT, ndims, Av.str().c_str(), other_z_name.str().c_str(), d1.str().c_str(), is_parallel);
        calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                            (size_t) (tmp_A.size()),

                                                            (size_t**) (ops_A.d_strides_operand_pointers),

                                                            (size_t*) (get_d_obj_strides()[d1.str().c_str()]),
                                                            (size_t*) (get_d_obj_cards()["F"]),

                                                            (size_t**) (ops_A.d_cards_operand_pointers),
                                                            (double**) (ops_A.d_operand_pointers),

                                                            (double*) (get_d_obj_data()[d1.str().c_str()]),
                                                            //(double*) (get_d_obj_data()["Z0"]),
                                                            (size_t) (h_objs[d1.str()]->element_number),
                                                            (size_t) 1,
                                                            CUPRINTF,2);

        //oc_push_back(&operation_chain, GMULT, ndims, hat_Xv.str().c_str(), other_z_name.str().c_str(), d2.str().c_str(), is_parallel, "F", p+1, 1);

	int to_power[tmp_M.size()];
	to_power[0]=p+1;
	for (size_t i=0; i<tmp_M.size(); i++){
	  to_power[i] = 0;
	}
        cur_mem = gen_operation_arguments(tmp_M, &ops_M, cur_mem, to_power);

        calculate_C_mops<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>((size_t) ndims,
                                                            (size_t) (tmp_M.size()),

                                                            (size_t**) (ops_M.d_strides_operand_pointers),

                                                            (size_t*) (get_d_obj_strides()[d2.str().c_str()]),
                                                            (size_t*) (get_d_obj_cards()["F"]),

                                                            (size_t**) (ops_M.d_cards_operand_pointers),
                                                            (double**) (ops_M.d_operand_pointers),

                                                            (double*) (get_d_obj_data()[d2.str().c_str()]),
                                                            //(double*) (get_d_obj_data()["Z0"]),
                                                            (size_t) (h_objs[d2.str()]->element_number),
                                                            (size_t) 1,
                                                            CUPRINTF,3,
							    ops_M.d_to_power
							    );


      }







      // sum D1_Zalpha_Xv and D2_Zalpha_Xv for all v to update Zalpha
      std::stringstream D1_Zalpha_sum, D2_Zalpha_sum; // will sum into these

      bool first = true;
      for (size_t v=0; v<max_v; v++){
        if ( R[v + alpha*max_v] ){
          if ( first ){
            D1_Zalpha_sum << "D1_Z" << alpha << "X" << v;
            D2_Zalpha_sum << "D2_Z" << alpha << "X" << v;
            first = false;
          }else{
            std::stringstream other_d1, other_d2;
            other_d1 << "D1_Z" << alpha << "X" << v;
            other_d2 << "D2_Z" << alpha << "X" << v;

	    hadamard_sum<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(get_d_obj_data()[D1_Zalpha_sum.str().c_str()],
							    get_d_obj_data()[other_d1.str().c_str()],
							    get_d_obj_data()[D1_Zalpha_sum.str().c_str()],
							    h_objs[D1_Zalpha_sum.str().c_str()]->element_number,
							    CUPRINTF);

	    hadamard_sum<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>(get_d_obj_data()[D2_Zalpha_sum.str().c_str()],
							    get_d_obj_data()[other_d2.str().c_str()],
							    get_d_obj_data()[D2_Zalpha_sum.str().c_str()],
							    h_objs[D2_Zalpha_sum.str().c_str()]->element_number,
							    CUPRINTF);

          }
        }
      }

      hadamard_div<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>( get_d_obj_data()[D1_Zalpha_sum.str().c_str()],
						       get_d_obj_data()[D2_Zalpha_sum.str().c_str()],
						       get_d_obj_data()[D1_Zalpha_sum.str().c_str()],
						       h_objs[D1_Zalpha_sum.str().c_str()]->element_number,
						       CUPRINTF);

      std::stringstream Zalpha;
      Zalpha << 'Z' << alpha ;
      hadamard_mul<<<NUM_BLOCKS, THREADS_FOR_BLOCK>>>( get_d_obj_data()[Zalpha.str().c_str()],
						       get_d_obj_data()[D1_Zalpha_sum.str().c_str()],
						       get_d_obj_data()[Zalpha.str().c_str()],
						       h_objs[Zalpha.str().c_str()]->element_number,
						       CUPRINTF);

    }

  }





  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");
  cudaDeviceSynchronize();

  if ( CUPRINTF == true ){
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
  }





































  /*

  // perform GCTF operation //////////////////////////////////////////////////////////////////

  std::vector<operation> operation_chain;




  // all hat_Xv are recalculated

  // update each Zn
  for (size_t alpha=0; alpha<max_alpha; alpha++){
    if ( latent_elements[alpha].is_updateable == false) continue;






    // update all hatX
    for ( size_t cur_v=0; cur_v<max_v; cur_v++){

      std::stringstream hat_Xv;
      hat_Xv << "hatX" << cur_v;


      // zN update

      // Z0 * Z1 -> hatX1
      // Z2 * Xhat -> hatX1
      // ...

      int used_Z_count = 0;
      std::stringstream first_Z;

      for ( size_t alpha=0; alpha<max_alpha; alpha++){
        //for (size_t z=0; z<Z_tensors.size(); z++){

        if ( R[cur_v + alpha*max_v] == false )  continue;

        else if ( used_Z_count == 0 ) {
          first_Z << "Z" << alpha;
          used_Z_count++;
          continue;
        }

        std::stringstream Zn;
        Zn << 'Z' << alpha;

        // in the second operand must check if this is the last operation (2 factor problem) store result in Xhat
        if (used_Z_count==1){
          if ( get_latent_tensor_num(R, cur_v, max_alpha, max_v) == 2 ){
            oc_push_back(&operation_chain, GMULT, ndims, first_Z.str().c_str(), Zn.str().c_str(), hat_Xv.str().c_str(), is_parallel);
            used_Z_count++;
            break; // do not need to loop any more for this hatX
          }else{
            // there will be more Zn to operate on save the result in F
            oc_push_back(&operation_chain, GMULT, ndims, first_Z.str().c_str(), Zn.str().c_str(), "F", is_parallel);
            used_Z_count++;
          }

        }else{
          // third or later Zn

          if (alpha != (Z_tensors.size()-1)){
            // in all non last operations store result in F to avoid mis-contraction
            oc_push_back(&operation_chain, GMULT, ndims, Zn.str().c_str(), "F", "F", is_parallel);
          }else{
            // in last operation store result into Xhat
            oc_push_back(&operation_chain, GMULT, ndims, Zn.str().c_str(), "F", hat_Xv.str().c_str(), is_parallel);
          }
        }
      }

      std::stringstream Xv;
      Xv << 'X' << cur_v ;

      std::stringstream Av;
      Av << 'A' << cur_v;

      oc_push_back(&operation_chain, HADAMARD_MUL, ndims, hat_Xv.str().c_str(), Xv.str().c_str(), Av.str().c_str(), is_parallel, "F", p, 1);
    }













    // for each Xv
    for (size_t cur_v=0; cur_v<max_v; cur_v++){
      if ( R[cur_v + alpha*max_v] == false ) continue; // if this Xv does not have this Zalpha dothing to do

      std::stringstream hat_Xv;
      hat_Xv << "hatX" << cur_v;


      // calculate D1_Zalpha_Xv

      std::stringstream d1;
      d1 << "D1_Z" << alpha << "X" << cur_v;
      size_t tmp_op_count=0;

      std::stringstream Av;
      Av << 'A' << cur_v;

      for (size_t other_z=0; other_z < max_alpha; other_z++){

        if (other_z == alpha || R[cur_v + other_z*max_v] == false ) continue;

        std::stringstream other_z_name;
        other_z_name << "Z" << other_z;

        if ( tmp_op_count == 0 ){
          if ( get_latent_tensor_num(R, cur_v, max_alpha, max_v) == 2 ){
            oc_push_back(&operation_chain, GMULT, ndims, Av.str().c_str(), other_z_name.str().c_str(), d1.str().c_str(), is_parallel);
            break; // do not need to loop any more for this D1_Zn with this Xv
          }else{
            oc_push_back(&operation_chain, GMULT, ndims, Av.str().c_str(), other_z_name.str().c_str(), "F", is_parallel);
          }
        }else{
          //if ( tmp_op_count != Z_tensors.size()-2){ // -1 for index starts from 0 -1 for Zn itself does not loop
          if ( tmp_op_count != get_latent_tensor_num(R, cur_v, max_alpha, max_v) - 2) { // -1 for D1_Zn does not loop for Zn, -1 for current op. is not incremented yet
            // in all non last operations store result in F to avoid mis-contraction
            oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", "F", is_parallel);
          }else{
            // in last operation store result into d1
            oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", d1.str().c_str(), is_parallel);
          }
        }

        tmp_op_count++;
      }





      // calculate D2_Zalpha_Xv

      std::stringstream d2;
      d2 << "D2_Z" << alpha << "X" << cur_v;
      tmp_op_count=0;

      for (size_t other_z=0; other_z < max_alpha; other_z++){

        if (other_z == alpha || R[cur_v + other_z*max_v] == false ) continue;

        std::stringstream other_z_name;
        other_z_name << "Z" << other_z;

        if ( tmp_op_count == 0 ){
          if ( get_latent_tensor_num(R, cur_v, max_alpha, max_v) == 2 ){
            oc_push_back(&operation_chain, GMULT, ndims, hat_Xv.str().c_str(), other_z_name.str().c_str(), d2.str().c_str(), is_parallel, "F", p+1, 1);
            break; // do not need to loop any more for this D2_Zn with this Xv
          }else{
            oc_push_back(&operation_chain, GMULT, ndims, hat_Xv.str().c_str(), other_z_name.str().c_str(), "F", is_parallel, "F", p+1, 1);
          }
        }else{
          // third or later Zn

          if ( tmp_op_count != get_latent_tensor_num(R, cur_v, max_alpha, max_v) - 2 ){
            // in all non last operations store result in F to avoid mis-contraction
            oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", "F", is_parallel);
          }else{
            // in last operation store result into d2
            oc_push_back(&operation_chain, GMULT, ndims, other_z_name.str().c_str(), "F", d2.str().c_str(), is_parallel);
          }
        }
        tmp_op_count++;
      }

      //std::stringstream Zup_v;
      //Zup_v << "Zup" << alpha << "X" << cur_v;
      //oc_push_back(&operation_chain, HADAMARD_DIV, ndims, d1.str().c_str(), d2.str().c_str(), Zup_v.str().c_str(), is_parallel);
    }





    // sum D1_Zalpha_Xv and D2_Zalpha_Xv for all v to update Zalpha
    std::stringstream D1_Zalpha_sum, D2_Zalpha_sum; // will sum into these

    bool first = true;
    for (size_t v=0; v<max_v; v++){
      if ( R[v + alpha*max_v] ){
        if ( first ){
          D1_Zalpha_sum << "D1_Z" << alpha << "X" << v;
          D2_Zalpha_sum << "D2_Z" << alpha << "X" << v;
          first = false;
        }else{
          std::stringstream other_d1, other_d2;
          other_d1 << "D1_Z" << alpha << "X" << v;
          other_d2 << "D2_Z" << alpha << "X" << v;
          oc_push_back(&operation_chain, HADAMARD_SUM, ndims, D1_Zalpha_sum.str().c_str(), other_d1.str().c_str(), D1_Zalpha_sum.str().c_str(), is_parallel);
          oc_push_back(&operation_chain, HADAMARD_SUM, ndims, D2_Zalpha_sum.str().c_str(), other_d2.str().c_str(), D2_Zalpha_sum.str().c_str(), is_parallel);
        }
      }
    }

    oc_push_back(&operation_chain, HADAMARD_DIV, ndims, D1_Zalpha_sum.str().c_str(), D2_Zalpha_sum.str().c_str(), D1_Zalpha_sum.str().c_str(), is_parallel);

    std::stringstream Zalpha;
    Zalpha << 'Z' << alpha ;
    oc_push_back(&operation_chain, HADAMARD_MUL, ndims, Zalpha.str().c_str(), D1_Zalpha_sum.str().c_str(), Zalpha.str().c_str(), is_parallel);
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

















  ///////////////////////////////////////////////////////////////////////////////////////////

  // transfer results to matlab /////////////////////////////////////////////////////////////

  if ( is_parallel ){
    for (size_t z=0; z<latent_elements.size(); z++){
      std::stringstream Zn;
      Zn << 'Z' << z;
      transferFromDevice(output_data_ptr[z], Zn.str().c_str());
    }
    //transferFromDevice(m_Z1, "Z1");
    //transferFromDevice(m_Z2, "Z2");
  }else{
    for (size_t z=0; z<latent_elements.size(); z++){
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
}
