/*
 * author: ck
 * 06.06.2011
 * advisor: atc
 */

#include "mex.h"
#include "cublas.h"
#include "cutil_inline.h"

#include <iostream>
#include <algorithm>

#include "cuPrintf.cu"

#include "tensor.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  std::cout << "mex: found " << nrhs << " number of arguments " << std::endl;

  ct_config h_ctc;
  ct_config* d_ctc = getDeviceTensorContractConfig(&h_ctc,prhs[0],prhs[1],prhs[2],prhs[3]);

  print_ct_config("Host ctc", &h_ctc);

  print_device_ctc("Device tmp ctc",d_ctc);

}
