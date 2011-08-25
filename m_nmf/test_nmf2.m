function [] = test_nmf2()
    close all
    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);
    
    I = 10;
    F = 20;
    T = 2;

    A_true = 10*rand(F, I);
    B_true = 10*rand(I, T);
    L = A_true*B_true;

    X = poissrnd(L);

    M=ones(size(X));
    
    [A B kl_data] = m_nmf(500, A_true, B_true, M, X, 0.00001);
    
    subplot(221);
    imagesc(X);
    title('X')
    subplot(222);
    imagesc(A*B);
    title('A*B');
	subplot(212);
    plot(kl_data);
    title('KL(L||X) in iterations');

    figure;
    subplot(221)
    imagesc(A);
    title('A');
    subplot(222)
    imagesc(B);
    title('B');
    subplot(223)
    imagesc(A_true);
    title('A true');
    subplot(224)
    imagesc(B_true);
    title('B true');

end