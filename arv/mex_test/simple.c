#include <string.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) 
{
int j;
double *output;
double data[] = {1.0, 2.1, 3.0};


/* Create the array */
plhs[0] = mxCreateDoubleMatrix(1,3,mxREAL);
output = mxGetPr(plhs[0]);


/* Populate the output */
memcpy(output, data, 3*sizeof(double));

/* Alternate method to populate the output, one element at a time */
/* for (j = 0; j < 3; j++) */
/* { */
/*   output[j] = data[j]; */
/* } */

}