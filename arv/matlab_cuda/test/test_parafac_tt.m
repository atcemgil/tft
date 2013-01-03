addpath('tensortoolbox/tensor_toolbox')

% I=10;
% J=11;
% K=12;
% A=10;

I=7;
J=8;
K=9;
A=10;


rand('state',0);
X = tenrand(I, J, K);

opts=struct('maxiters',100,'tol',1e-5,'printitn',0);

tic; P = parafac_als(X,A,opts); toc
