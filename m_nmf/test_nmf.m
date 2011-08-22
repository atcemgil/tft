function [] = test_nmf()
    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);

    d = 5;
    X=rand(d,d)*5;

    Z1_orig=rand(d,d)*1;
    Z2_orig=rand(d,d)*1;
    
    M=ones(size(X));

    display('X')
    display(X)
    display('M')
    display(M)
    
    trials=100;
    diff_data=zeros(1,trials);
    for i=1:trials
        [Z1 Z2] = m_nmf(i, Z1_orig, Z2_orig, M, X, 0.00001);
        diff_data(i) = get_mean_diff(X,Z1*Z2);
        %display(Z1*Z2);
    end

    plot(diff_data);
    display(Z1*Z2);
    display(diff_data);
end
