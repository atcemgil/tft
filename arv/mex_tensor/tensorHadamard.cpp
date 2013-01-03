#include "mex.h"
#include "tensor.h"
#include <iostream>
using namespace std;
// Tensor .* operation. Multiply corresponding entries of tensors A,B of same size
// Store the result in tensor C

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int i, j, m, n;
    double *data1, *data2;


    Tensor<float> A((size_t) mxGetNumberOfDimensions(prhs[0]), (size_t*) mxGetDimensions(prhs[0]) );
    Tensor<float> B((size_t) mxGetNumberOfDimensions(prhs[1]), (size_t*) mxGetDimensions(prhs[1]) );

    // assume same cardinalities
    for( size_t k=0; k<(size_t) mxGetNumberOfElements(prhs[0]); k++){
      (*A.va)[k] = ( (double*)mxGetData(prhs[0])) [k];
      (*B.va)[k] = ( (double*)mxGetData(prhs[1])) [k];
    }

    // assume same cardinalities
    Tensor<float> C((size_t) mxGetNumberOfDimensions(prhs[1]), (size_t*) mxGetDimensions(prhs[1]) );
    Tmult_contract(C, A, B, true);

    plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxREAL);
    double* outputdata = mxGetPr(plhs[0]);
    // assume same cardinalities
    for( size_t k=0; k<(size_t) mxGetNumberOfElements(prhs[0]); k++){
      outputdata[k] = (*C.va)[k] ;
    }


/*     for (i = 0; i < nrhs; i++) { */
/*         /\* Find the dimensions of the data *\/ */
/*         m = mxGetM(prhs[i]); */
/*         n = mxGetN(prhs[i]); */
        
        
/*         /\* Create an mxArray for the output data *\/ */
/*         plhs[i] = mxCreateDoubleMatrix(m, n, mxREAL); */
        
        
/*         /\* Retrieve the input data *\/ */
/*         data1 = mxGetPr(prhs[i]); */
        
        
/*         /\* Create a pointer to the output data *\/ */
/*         data2 = mxGetPr(plhs[i]); */
        
        
/*         /\* Put data in the output array *\/ */
/*         for (j = 0; j < m*n; j++){ */
/*             data2[j] = 2 * data1[j]; */
/*         } */
/*     } */
    
}
