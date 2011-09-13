format 'compact'

display('compiling');

!/usr/local/cuda/bin/nvcc -c -g  mct_tensorop_utils.cu mct_tensorop_cpp.cu mct_tensorop_gpu.cu mct_kernels.cu mct.cu -arch sm_13 -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/nvidia/NVIDIA_GPU_Computing_SDK/C/common/inc/

display('linking');
mex -largeArrayDims ...
    mct.o mct_tensorop_utils.o mct_tensorop_cpp.o ...
    mct_tensorop_gpu.o mct_kernels.o ...
    cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o ...
    cutil/cutil.cpp.o ...
    cutil/multithreading.cpp.o cutil/param.cpp.o ...
    cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o  ...
    -L /usr/local/cuda/lib64 -lcudart -lcufft


display('running');

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I = 2;
F = 2;
T = 2;

A_true = 10*rand(F, I);
B_true = 10*rand(I, T);
L = A_true*B_true;

X = poissrnd(L);

M=ones(size(X));


display('target X');
display(X);



% GPU code
display('GPU run');
tic; [Z1_gpu Z2_gpu]=mct('nmf_gpu',X, M); toc;
display('GPU result');
display('Z1 result');
display(Z1_gpu);
display('Z2 result');
display(Z2_gpu);
display('X result');
display(Z1_gpu*Z2_gpu);
exit
% C code
display('C code run')
tic; [Z1_cpp Z2_cpp]=mct('nmf_cpp',X, M); toc;
display('CPU result');
display(C);


exit


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

