!/usr/local/cuda/bin/nvcc -c cudatensor.cu -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/NVIDIA_GPU_Computing_SDK/C/common/inc/

mex -largeArrayDims cudatensor.o cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o cutil/cutil.cpp.o cutil/multithreading.cpp.o cutil/param.cpp.o cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o   -L /usr/local/cuda/lib64 -lcudart -lcufft -L /usr/local/cuda/lib64/

X=single(magic(2))

%C=cudatensor(X,X);
C=cudatensor(X,single([2 2 0]),X,single([0 2 2]), single([2 0 2]) );

display('matlab result is')
display(C)
