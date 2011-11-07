format 'compact'

for i=1:40
    display('****');
end

%!rm *.o *.mexa64

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

X = 10*rand(2,4)



% GPU code
display('GPU run');
tic; [Z1_gpu Z2_gpu]=mct('pltf_gpu', ['i','k','j'], [2 3 4], ['i','j'], X, ['i','k'], ['k','j']); toc;

display('GPU result');
display('Z1 result');
display(Z1_gpu);
display('Z2 result');
display(Z2_gpu);
display('X result');
display(Z1_gpu*reshape(Z2_gpu,3,4));


% C code
display([char(10) char(10) 'C code run'])
tic; [Z1_cpp Z2_cpp]=mct('pltf_cpp', ['i','k','j'], [2 3 4], ['i','j'], X, ['i','k'], ['k','j']); toc;
display('CPU result');
display('Z1 result');
display(Z1_cpp);
display('Z2 result');
display(Z2_cpp);
display('X result');
display(Z1_cpp * reshape(Z2_cpp,3,4));

return


display('comparing GPU and C code')

epsilon=0.001

numeldiff_z1 = numel(Z1_cpp) - numel(Z1_gpu);
allequal_z1=0;
if numeldiff_z1 == 0
    display(['numeldiff ok : ' num2str(numeldiff_z1)]);
    for n=1:numel(Z1_cpp)
	%if Z1_cpp(n) - Z1_gpu(n) > epsilon
	if Z1_cpp(n) ~= Z1_gpu(n) 
	    allequal_z1 = allequal_z1+1;
	end
    end
end
numeldiff_z2 = numel(Z2_cpp) - numel(Z2_gpu);
allequal_z2=0;
if numeldiff_z2 == 0
    display(['numeldiff ok : ' num2str(numeldiff_z2)]);
    for n=1:numel(Z2_cpp)
	%if Z2_cpp(n) - Z2_gpu(n) > epsilon
	if Z2_cpp(n) ~= Z2_gpu(n)
	    allequal_z2 = allequal_z2+1;
	end
    end
end

if allequal_z1 ~= 0 || numeldiff_z1 ~= 0 || allequal_z2 ~= 0 || numeldiff_z2 ~= 0 

	display('ERROR: GPU and C code results do not match');
else
	display('OK: GPU and C code results match');
end

return

