format 'compact'

for i=1:40
    display('****');
end


S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=3;
K=2;
J=3;

X = 10*rand(I,J)

% GPU code
display([char(10) char(10) 'GPU run']);
tic; [Z1_gpu Z2_gpu]=mct('pltf_gpu', ['i','k','j'], [I K J], ['i','j'], X, ['i','k'], ['k','j']); toc;

display('GPU result');
display('Z1 result');
display(Z1_gpu);
display('Z2 result');
display(Z2_gpu);
display('X result');
display(reshape(Z1_gpu,I,K)*reshape(Z2_gpu,K,J));


% C code
display([char(10) char(10) 'C code run'])
tic; [Z1_cpp Z2_cpp]=mct('pltf_cpp', ['i','k','j'], [I K J], ['i','j'], X, ['i','k'], ['k','j']); toc;
display('CPU result');
display('Z1 result');
display(Z1_cpp);
display('Z2 result');
display(Z2_cpp);
display('X result');
display(reshape(Z1_cpp,I,K) * reshape(Z2_cpp,K,J));


display([char(10) char(10) 'comparing GPU and C code'])

%epsilon=0.001

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

