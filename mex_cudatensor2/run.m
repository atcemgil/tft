!/usr/local/cuda/bin/nvcc -c cudatensor2.cu -arch sm_13 -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/NVIDIA_GPU_Computing_SDK/C/common/inc/

mex -largeArrayDims cudatensor2.o cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o cutil/cutil.cpp.o cutil/multithreading.cpp.o cutil/param.cpp.o cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o   -L /usr/local/cuda/lib64 -lcudart -lcufft

dim=2

X=magic(dim)
%X=round(rand(dim)*100)

%C=cudatensor(X,X);
%C=cudatensor2(X,[dim dim 0],X,[0 dim dim], [dim 0 dim] );
C=cudatensor2(X,[0 dim dim],X,[dim dim dim], [dim dim dim] );

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

