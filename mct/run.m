!/usr/local/cuda/bin/nvcc -c mct.cu -arch sm_13 -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/nvidia/NVIDIA_GPU_Computing_SDK/C/common/inc/

mex -largeArrayDims mct.o cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o cutil/cutil.cpp.o cutil/multithreading.cpp.o cutil/param.cpp.o cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o   -L /usr/local/cuda/lib64 -lcudart -lcufft

rand('state',0);

dim=5;

A=magic(dim);
B=round(rand(dim,dim,dim)*10);

% GPU code
display('GPU run');
tic; G=mct(A,[0 dim dim],B,[dim dim dim], [dim dim 0], 0, 1); toc;
% C code
display('C code run')
tic; C=mct(A,[0 dim dim],B,[dim dim dim], [dim dim 0], 1, 1); toc;

display('comparing GPU and C code')

numeldiff = numel(C) - numel(G);
allequal=0;
if numeldiff == 0
    display(['numeldiff ok : ' num2str(numeldiff)]);
    for n=1:numel(C)
	if C(n) ~= G(n)
	    allequal = allequal+1;
	end
    end
end

if allequal ~= 0 || numeldiff ~= 0
	display('ERROR: GPU and C code results do not match');
else
	display('OK: GPU and C code results match');
end

exit

