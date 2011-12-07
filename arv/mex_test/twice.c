#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int i, j, m, n;
    double *data1, *data2;
    
    if (nrhs != nlhs)
        mexErrMsgTxt("The number of input and output arguments must be the same.");
    
    
    for (i = 0; i < nrhs; i++) {
        /* Find the dimensions of the data */
        m = mxGetM(prhs[i]);
        n = mxGetN(prhs[i]);
        
        
        /* Create an mxArray for the output data */
        plhs[i] = mxCreateDoubleMatrix(m, n, mxREAL);
        
        
        /* Retrieve the input data */
        data1 = mxGetPr(prhs[i]);
        
        
        /* Create a pointer to the output data */
        data2 = mxGetPr(plhs[i]);
        
        
        /* Put data in the output array */
        for (j = 0; j < m*n; j++){
            data2[j] = 2 * data1[j];
        }
    }
}