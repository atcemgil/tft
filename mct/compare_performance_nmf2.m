format 'compact'
rand('state',0);
test_range=2:8:50;
t_gpu=zeros(length(test_range),1);
t_c=zeros(length(test_range),1);
i=0;

for card=test_range
    i=i+1;
    dim1=100
    display(['testing cardinality ' num2str(dim1) 'x' num2str(card)]);

    X = 100*rand(dim1,card);
    M=ones(size(X));

    % GPU code
    tic; [Z1_gpu Z2_gpu]=mct('nmf_gpu',X, M);
    t_gpu(i)=toc;
    
    % C code
    tic; [Z1_cpp Z2_cpp]=mct('nmf_cpp',X, M);
    t_c(i)=toc;

    diff_z1 = Z1_gpu ~= Z1_cpp;
    diff_z2 = Z2_gpu ~= Z2_cpp;

    diff_sum=0;
    for j=1:card
        diff_sum = diff_sum + sum(diff_z1);
        diff_sum = diff_sum + sum(diff_z2);
    end

    if diff_sum == 0
        display('diff_sum ok');
    else
        display(['diff_sum ERROR: ' num2str(diff_sum)]);
    end

end
display(t_gpu);
display(t_c);
plot (test_range,t_c,'b-', test_range, t_gpu, 'r-')
legend('C code', 'GPU code')
xlabel('one dimension cardinality')
ylabel('seconds')
title('C code vs GPU code (NMF)')
