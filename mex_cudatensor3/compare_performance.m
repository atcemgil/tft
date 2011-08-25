
rand('state',0);
test_range=2:30:300;
t_gpu=zeros(length(test_range),1);
t_c=zeros(length(test_range),1);
i=0;

for dim=test_range
    i=i+1;
    display(['testing dim' num2str(dim)]);

    A=magic(dim);
    B=round(rand(dim,dim,dim)*10);

    % GPU code
    tic; C_gpu_code=cudatensor3(A,[0 dim dim],B,[dim dim dim], ...
                                [dim dim 0], 0, 1); t_gpu(i)=toc;
    % C code
    tic; C_c_code=cudatensor3(A,[0 dim dim],B,[dim dim dim], ...
                              [dim dim 0], 1, 1); t_c(i)=toc;

    diff = C_gpu_code ~= C_c_code;
    diff_sum=0;
    for j=1:dim
        diff_sum = sum(diff);
    end

    if diff_sum == 0
        display('diff_sum ok');
    else
        display(['diff_sum ERROR: ' num2str(diff_sum)]);
    end

end

plot (test_range,t_c,'b-', test_range, t_gpu, 'r-')
legend('C code', 'GPU code')
xlabel('dimension')
ylabel('seconds')
title('C code vs GPU code')

