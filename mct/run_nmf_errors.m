format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=10;
J=15;
K=20;

V=['i','k','j'];

A_cards=['i','k'];
A_true=round(10*rand(I,K));

B_cards=['k','j'];
B_true=round(20*rand(K,J));

X_cards=['i','j'];
X_true=A_true*B_true;

X=poissrnd(X_true);

iter_range = 2:100:1000;
gpu_times = zeros(1,length(iter_range));
sequential_times = zeros(1,length(iter_range));
gpu_error = zeros(1,length(iter_range));
sequential_error = zeros(1,length(iter_range));

for i=1:length(iter_range)
  iter = iter_range(i);
  display(['testing iteration ' num2str(iter)]);

  cards=[I, K, J];

  tic; [A B]=mct('pltf_gpu', iter, V, cards, X_cards, X, A_cards, B_cards); gpu_times(i)=toc;
  gpu_error(i) = sum(get_KL_div(X, reshape(A,I,K)*reshape(B,K,J)));

  tic; [A B]=mct('pltf_cpp', iter, V, cards, X_cards, X, A_cards, B_cards); sequential_times(i)=toc;
  sequential_error(i) = sum(get_KL_div(X, reshape(A,I,K)*reshape(B,K,J)));

end

display(gpu_times);
plot (iter_range, gpu_times, '-',...
      iter_range, sequential_times, '--')
legend('GPU code', 'Sequential_code')
xlabel('Number of iterations')
ylabel('seconds')
title(['NMF run times I ' num2str(I) ' J ' num2str(J) ' K ' num2str(K)])

figure

plot (iter_range, gpu_error, '-',...
      iter_range, sequential_error, '--')
legend('GPU code', 'Sequential code')
xlabel('Number of iterations')
ylabel('KL divergence (X || Xhat)')
title(['NMF KL divergence I ' num2str(I) ' J ' num2str(J) ' K ' num2str(K)])

format 'long'
display('gpu error')
display(gpu_error');
display('sequential error')
display(sequential_error');