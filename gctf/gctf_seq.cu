/*
 * author: ck
 * created: 16.02.2012
 * advisor: atc
 */

#include "gctf.cuh"
#include <iostream>
#include "../common/utils.cuh"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if ( COUT ) std::cout << "gctf_seq: found " << nrhs << " number of arguments " << std::endl;

  // read config from file
  read_config();
  //print_config();

  //gctf(nlhs, plhs, nrhs, prhs, false);
  std::cout << "not implemented" << std::endl;
}
