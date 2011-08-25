function [] = test_nmf()
    close all
    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);
    
    I = 10;
    F = 20;
    T = 2;

    X=rand(F,T)*5;

    Z1_orig=rand(F,I);
    Z2_orig=rand(I,T);
    
    M=ones(size(X));
 
    SHOW_ITERS=0;
    
    trials=500;
    
    if SHOW_ITERS==1
        diff_data=zeros(1,trials);
        for i=1:trials
            [Z1 Z2] = m_nmf(i, Z1_orig, Z2_orig, M, X, 0.00001);
            diff_data(i) = get_mean_diff(X,Z1*Z2);
            %display(Z1*Z2);
        end
    else
        [Z1 Z2] = m_nmf(trials, Z1_orig, Z2_orig, M, X, 0.00001);
    end
    
    if SHOW_ITERS==1, subplot(221); else subplot(121); end
    imagesc(X);
    title('X')
    if SHOW_ITERS==1, subplot(222); else subplot(122); end
    imagesc(Z1*Z2);
    title('Z1*Z2');
    if SHOW_ITERS == 1
        subplot(212);
        plot(diff_data);
        title('Difference in iterations');
    end

end