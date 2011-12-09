/*
 * author: ck
 * created: 08.12.2011
 * advisor: atc
 */

#include "gmult.cuh"
#include <iostream>
#include "../common/utils.cuh"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if ( COUT ) std::cout << "gmult_par: found " << nrhs << " number of arguments " << std::endl;

  // read config from file
  read_config();
  //print_config();

  tensorop(nlhs, plhs, nrhs, prhs, true);
}
