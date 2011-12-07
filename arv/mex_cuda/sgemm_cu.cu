#include "mex.h"
#include "cublas.h"

/* sgemm_cu.cu - Gateway function for subroutine sgemm
   C = sgemm_cu(transa,transb,single(alpha),single(beta),single(A),single(B),single(C))
   transa,transb = 0/1 for no transpose/transpose of A,B
   Input arrays must be single precision.
*/

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  cublasStatus status;
  int M,K,L,N,MM,NN,KK;
  int Mc,Kc,Lc,Nc,MMc,NNc,KKc;
  mwSize dims0[2];
  int ta,tb;
  float alpha,beta;
  float *a,*b,*c,*cc;
  float *ga,*gb,*gc;
  char transa,transb;
  cublasStatus retStatus;

  if (nrhs != 7) { mexErrMsgTxt("sgemm requires 7 input arguments");
  } else if (nlhs != 1) { mexErrMsgTxt("sgemm requires 1 output argument");  }

  if ( !mxIsSingle(prhs[4]) || !mxIsSingle(prhs[5]) || !mxIsSingle(prhs[6])) {
    mexErrMsgTxt("Input arrays must be single precision.");
  }

  ta = (int) mxGetScalar(prhs[0]);
  tb = (int) mxGetScalar(prhs[1]);
  alpha = (float) mxGetScalar(prhs[2]);
  beta = (float) mxGetScalar(prhs[3]);
  M = mxGetM(prhs[4]); /* gets number of rows of A */
  K = mxGetN(prhs[4]); /* gets number of columns of A */
  L = mxGetM(prhs[5]); /* gets number of rows of B */
  N = mxGetN(prhs[5]); /* gets number of columns of B */

  if (ta == 0) { transa='n'; MM=M; KK=K;
  } else { transa='t'; MM=K; KK=M; }

  if (tb == 0) { transb='n'; NN=N; }
  else { transb='t'; NN=L;
  }

  /* 
     printf("transa=%c\n",transa); 
     printf("transb=%c\n",transb); 
     printf("alpha=%f\n",alpha); 
     printf("beta=%f\n",beta);
   */

  /* Left hand side matrix set up */ 
  dims0[0]=MM; 
  dims0[1]=NN; 
  plhs[0] = mxCreateNumericArray(2,dims0,mxSINGLE_CLASS,mxREAL); 
  cc = (float*) mxGetData(plhs[0]);

  /* Three single-precision arrays */ 
  a = (float*) mxGetData(prhs[4]); 
  b = (float*) mxGetData(prhs[5]); 
  c = (float*) mxGetData(prhs[6]);

  /* STARTUP CUBLAS */ 
  retStatus = cublasInit();

  // test for error 
  retStatus = cublasGetError (); 
  if (retStatus != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS: an error occurred in cublasInit\n"); }

  Mc=M+32-M%32; 
  Kc=K+32-K%32;

  /* ALLOCATE SPACE ON THE GPU AND COPY a INTO IT */ 
  cublasAlloc (Mc*Kc, sizeof(float), (void**)&ga); 

  // test for error 
  retStatus = cublasGetError (); 
  if (retStatus != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS: an error occurred in cublasAlloc\n");} 

  cudaMemset(ga,0,Mc*Kc*4);
  Lc=L+32-L%32; Nc=N+32-N%32;

  /* SAME FOR B, C */ 
  cublasAlloc (Lc*Nc, sizeof(float), (void**)&gb); 
  cudaMemset(gb,0,Lc*Nc*4); 
  retStatus = cublasSetMatrix (L, N, sizeof(float), b, L, (void*)gb, Lc);

  MMc=MM+32-MM%32; 
  NNc=NN+32-NN%32; 
  KKc=KK+32-KK%32; 

  cublasAlloc (MMc*NNc, sizeof(float), (void**)&gc); 

  if (beta != 0.0 ) { 
    cudaMemset(gc,0,MMc*NNc*4); 
    retStatus = cublasSetMatrix (MM, NN, sizeof(float), c, MM, (void*)gc, MMc);
  }

  /* PADDED ARRAYS */ 
  /*
    printf("Op(A) has No. rows = %i\n",MMc); 
    printf("Op(B) has No. cols = %i\n",NNc); 
    printf("Op(A) has No. cols = %i\n",KKc); 
    printf("A has leading dimension = %i\n",Mc); 
    printf("B has leading dimension = %i\n",Lc); 
    printf("C has leading dimension = %i\n",MMc); 
  */
  
  /* READY TO CALL SGEMM */ 
  (void) cublasSgemm (transa, transb, MMc, NNc, KKc, alpha, ga, Mc, gb, Lc, beta, gc, MMc);
  
  status = cublasGetError(); 
  if (status != CUBLAS_STATUS_SUCCESS) { fprintf (stderr, "!!!! kernel execution error.\n"); }


  /* NOW COPY THE RESULTING gc ON THE GPU TO THE LOCAL c */ 
  retStatus = cublasGetMatrix (MM, NN, sizeof(float), gc, MMc, cc, MM); 
  if (retStatus != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS: an error occurred in cublasGetMatrix\n"); }

  /* FREE UP GPU MEMORY AND SHUTDOWN (OPTIONAL?) */ 
  cublasFree (ga); 
  cublasFree (gb); 
  cublasFree (gc); 
  cublasShutdown();
}
