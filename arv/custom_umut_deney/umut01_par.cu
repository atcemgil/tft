/*
 * author: ck
 * created: 16.02.2012
 * advisor: atc
 */

#include "umut01.cuh"
#include <iostream>
#include "../common/utils.cuh"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if ( COUT ) std::cout << "umut01_par: found " << nrhs << " number of arguments " << std::endl;

  // read config from file
  read_config();
  //print_config();

  umut01(nlhs, plhs, nrhs, prhs, true);
}
