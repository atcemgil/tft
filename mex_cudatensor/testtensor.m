addpath('/home/can2/tensor_toolbox/tensor_toolbox_2.4/')
addpath('/home/can2/tensor_toolbox/tensor_toolbox_2.4/algorithms')
addpath('/home/can2/tensor_toolbox/tensor_toolbox_2.4/met')

!/usr/local/cuda/bin/nvcc -c cudatensor.cu -arch sm_11 -Xcompiler -fPIC -I /opt/matlab/extern/include -I /home/can2/NVIDIA_GPU_Computing_SDK/C/common/inc/

mex -largeArrayDims cudatensor.o cutil/bank_checker.cpp.o cutil/cmd_arg_reader.cpp.o cutil/cutil.cpp.o cutil/multithreading.cpp.o cutil/param.cpp.o cutil/stopwatch.cpp.o cutil/stopwatch_linux.cpp.o   -L /usr/local/cuda/lib64 -lcudart -lcufft

dims=[2,2,2];

total_card=1;
for k=1:length(dims)
    total_card = total_card * dims(k);
end

rand('state',0);
data = single(round(rand(dims)*10));

Xten = tensor(data,dims)
X = data

A = single(round(rand(2,1)*10))
Aten = tensor(A,[2 1 1])


C=cudatensor(X,single([2 2 2]),A,single([2 0 0]), single([2 0 2]) );


display('mex result is')
display(C)

display('tensor toolbox result is')
%Yten = ttv(Xten, A, 1);
Yten = ttt(Xten,Aten,[1],[1]);
Y=Yten.data;
display(Y)

if sum(sum(C==Y)) == (total_card)
        display('olleeey')
else
        display('öff')
        display('boyutlar')
        total_card
        display('farklı hane sayisi')
        total_card - sum(sum(C==Y))
        %display('farklı hanelerin toplami')
        %sum( C( (C~=Y) ) - X( (C~=Y) ) )
end

display( ['sumsum c ' num2str(sum(sum(C))) ])
display( ['sumsum y ' num2str(sum(sum(Y))) ])
exit

