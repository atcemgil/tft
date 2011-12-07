function [] = test_nmf2()
    close all
    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);
    
    I = 5;
    F = 5;
    T = 5;

    A_true = 10*rand(F, I);
    B_true = 10*rand(I, T);
    L = A_true*B_true;

    X = poissrnd(L);

    M=ones(size(X));
    
    %Z1= [0.0600514, 0.788318 ; 0.203068 , 0.348563]';
    %Z2= [0.361609, 0.134639 ; 0.375968, 0.259322]';
    %[A B kl_data] = m_nmf(500, Z1, Z2, M, X, 0.00001)
    [A B kl_data] = m_nmf(10, A_true, B_true, M, X, 0.00001)

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