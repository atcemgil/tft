format 'compact'
rand('state',0);
test_range=2:10:300;
t_gpu=zeros(length(test_range),1);
t_c=zeros(length(test_range),1);
i=0;

dim1=100;

for dim2=test_range
    i=i+1;
    display(['testing dim' num2str(dim2)]);
    A_card = [dim1 0 dim2 ];
    B_card = [dim1 dim1 dim1 ];
    C_card = [dim1 dim1 0 ];

    A=round(rand(dim1, 1, dim2)*100);
    B=round(rand(dim1, dim1, dim1)*10);
    % C code
    tic; C_c_code=mct('tensor_cpp',A,A_card,B,B_card, C_card, 1) ;
    t_c(i)=toc;
    % GPU code
    tic; C_gpu_code=mct('tensor_gpu',A,A_card,B,B_card, C_card, 1) ;
    t_gpu(i)=toc;




	numeldiff = numel(C_c_code) - numel(C_gpu_code);
	allequal=0;
	if numeldiff == 0
	    display(['numeldiff ok : ' num2str(numeldiff)]);
	    for n=1:numel(C_c_code)
		if C_c_code(n) ~= C_gpu_code(n)
		    allequal = allequal+1;
		end
	    end
	end

	if allequal ~= 0 || numeldiff ~= 0
	    display(['Something is wrong allequal ' num2str(allequal) ' numeldiff ' num2str(numeldiff)])
	else
	    display('C code and GPU code results match')
	end


end

plot (test_range,t_c,'--', test_range, t_gpu, '-')
legend('C code', 'GPU code')
xlabel('Third dimension cardinality of output tensor')
ylabel('seconds')
title('C code vs GPU code')

