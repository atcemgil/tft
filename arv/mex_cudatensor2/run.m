!/usr/local/cuda/bin/nvcc -c cudatensor2.cu -arch sm_13 -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/NVIDIA_GPU_Computing_SDK/C/common/inc/

mex -largeArrayDims cudatensor2.o cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o cutil/cutil.cpp.o cutil/multithreading.cpp.o cutil/param.cpp.o cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o   -L /usr/local/cuda/lib64 -lcudart -lcufft

rand('state',0);

dim=2

A=magic(dim)
B=round(rand(dim,dim,dim)*10)

tic; C=cudatensor2(A,[0 dim dim],B,[dim dim dim], [dim dim dim] ); toc;

display('mex result is')
display(C)

%display('matlab kendi result is')
%display(X*X)

%if sum(sum(C==X*X)) == (dim*dim) 
%	display('olleeey')
%else
%	display('öff')
%end

exit

