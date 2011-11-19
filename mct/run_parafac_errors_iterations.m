format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=10;
J=20;
K=30;
a=5;

V_card_sym=['i','j','k','a']
V_cards=[I, J, K, a];

A_card_sym=['i','a']
A_true = round(10*rand(I,1,1,a));

B_card_sym=['j','a']
B_true = round(20*rand(1,J,1,a));

C_card_sym=['k','a']
C_true = round(30*rand(1,1,K,a));

X_card_sym = ['i','j','k'];
X_true = get_parafac(A_true,B_true,C_true,I,J,K,a,[I J K])

X = poissrnd(X_true)

%iter_range = 1:300:1000;
iter_range = 1:50:160;

gpu_times = zeros(1,length(iter_range));
sequential_times = zeros(1,length(iter_range));

gpu_error = zeros(1,length(iter_range));
sequential_error = zeros(1,length(iter_range));

for i=1:length(iter_range)
  iter = iter_range(i);
  display(['testing iteration ' num2str(iter)]);

  tic; [A B C]=mct('pltf_cpp', iter, V_card_sym, V_cards, X_card_sym, X, A_card_sym, B_card_sym, C_card_sym); sequential_times(i)=toc;
  sequential_error(i) = sum(get_KL_div(X, get_parafac(A,B,C,I,J,K,a,size(X))));

  tic; [A B C]=mct('pltf_gpu', iter, V_card_sym, V_cards, X_card_sym, X, A_card_sym, B_card_sym, C_card_sym); gpu_times(i)=toc;
  gpu_error(i) = sum(get_KL_div(X, get_parafac(A,B,C,I,J,K,a,size(X))));

end

display(gpu_times);
display(sequential_times);
plot (iter_range, gpu_times, '-', ...
      iter_range, sequential_times, '--')
legend('GPU code', 'Sequential code')
xlabel('Number of iterations')
ylabel('seconds')
title(['PARAFAC run times I ' num2str(I) ' J ' num2str(J) ...
       ' K ' num2str(K) ' a ' num2str(a)])

figure

display(gpu_error);
display(sequential_error);
plot (iter_range, gpu_error, '-',...
      iter_range, sequential_error, '--')
legend('GPU code', 'Sequential code')
xlabel('Number of iterations')
ylabel('KL divergence (X || Xhat)')
title([ 'PARAFAC KL divergence I ' num2str(I) ' J ' num2str(J) ...
        ' K ' num2str(K) ' a ' num2str(a)])


