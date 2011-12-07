format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=5;
J=6;
K=7;

V=['i','j','k','a']
A_cards=['i','a']
B_cards=['j','a']
C_cards=['k','a']
X_cards = ['i','j','k'];

a=10;
A_true = round(10*rand(I,1,1,a));
B_true = round(20*rand(1,J,1,a));
C_true = round(30*rand(1,1,K,a));
X_true = get_parafac(A_true,B_true,C_true,I,J,K,a,[I J K]);
X = poissrnd(X_true);
X(X==0)=0.000001; % suppress zeros, division/log problems

a_range=2:1:50;

iter=50;

gpu_times = zeros(1,length(a_range));
sequential_times = zeros(1,length(a_range));

gpu_error = zeros(1,length(a_range));
sequential_error = zeros(1,length(a_range));

for i=1:length(a_range)
    a = a_range(i);

    display(['testing a ' num2str(a)]);

    cards=[I, J, K, a];

    %tic; [A B C]=mct('pltf_cpp', iter, V, cards, X_cards, X, A_cards, B_cards, C_cards); sequential_times(i)=toc;
    %sequential_error(i) = sum(get_KL_div(X, get_parafac(A,B,C,I,J,K,a,size(X))));

    tic; [A B C]=mct('pltf_gpu', iter, V, cards, X_cards, X, A_cards, B_cards, C_cards); gpu_times(i)=toc;
    gpu_error(i) = sum(get_KL_div(X, get_parafac(A,B,C,I,J,K,a,size(X))));

end

format long eng
display('gpu error');
display(gpu_error');
display('sequential error');
display(sequential_error');
display('gpu times')
display(gpu_times');
display('sequential times');
display(sequential_times');


plot (a_range, gpu_times, '-', ...
      a_range, sequential_times, '--')
legend('GPU code', 'Sequential code')
xlabel('Cardinality of dimension a')
ylabel('seconds')
title(['PARAFAC run times ' num2str(iter) ' iterations I ' num2str(I) ' J ' num2str(J) ...
       ' K ' num2str(K)])

figure

plot (a_range, gpu_error, '-',...
      a_range, sequential_error, '--')
legend('GPU code', 'Sequential code')
xlabel('Cardinality of dimension a')
ylabel('KL divergence (X || Xhat)')
title(['PARAFAC KL divergence ' iter ' iterations I ' num2str(I) ' J ' num2str(J) ...
       ' K ' num2str(K)])


