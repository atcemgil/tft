!/usr/local/cuda/bin/nvcc -c cudatensor.cu -arch sm_11 -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/NVIDIA_GPU_Computing_SDK/C/common/inc/

mex -largeArrayDims cudatensor.o cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o cutil/cutil.cpp.o cutil/multithreading.cpp.o cutil/param.cpp.o cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o   -L /usr/local/cuda/lib64 -lcudart -lcufft -L /usr/local/cuda/lib64/

dim=3

X=single(magic(dim))
%X=single(round(rand(2)*100))

%C=cudatensor(X,X);
C=cudatensor(X,single([dim dim 0]),X,single([0 dim dim]), single([dim 0 dim]) );

display('mex result is')
display(C)

display('matlab kendi result is')
display(X*X)

if sum(sum(C==X*X)) == (dim*dim) 
	display('olleeey')
else
	display('öff')
end

exit
