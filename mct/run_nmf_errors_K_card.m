format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=10;
J=15;
V=['i','k','j'];

A_cards=['i','k'];
B_cards=['k','j'];
X_cards=['i','j'];

iter=50;

K_range = 2:40:200;
gpu_times = zeros(1,length(K_range));
sequential_times = zeros(1,length(K_range));
gpu_error = zeros(1,length(K_range));
sequential_error = zeros(1,length(K_range));

for i=1:length(K_range)
  K = K_range(i);
  A_true=round(10*rand(I,K));
  B_true=round(20*rand(K,J));
  X_true=A_true*B_true;
  X=poissrnd(X_true);

  display(['testing K ' num2str(K)]);

  cards=[I, K, J];

  tic; [A B]=mct('pltf_gpu', iter, V, cards, X_cards, X, A_cards, B_cards); gpu_times(i)=toc;
  gpu_error(i) = get_KL_div(X, reshape(A,I,K)*reshape(B,K,J));

  tic; [A B]=mct('pltf_cpp', iter, V, cards, X_cards, X, A_cards, B_cards); sequential_times(i)=toc;
  sequential_error(i) = get_KL_div(X, reshape(A,I,K)*reshape(B,K,J));

end

display(gpu_times);
plot (K_range, gpu_times, '-',...
      K_range, sequential_times, '--')
legend('GPU code', 'Sequential_code')
xlabel('K dimension cardinality')
ylabel('seconds')
title(['NMF run times ' num2str(iter) ' iterations I ' num2str(I) ' J ' num2str(J) ])

%figure

%plot (K_range, gpu_error, '-',...
%      K_range, sequential_error, '--')
%legend('GPU code', 'Sequential code')
%xlabel('K dimension cardinality')
%ylabel('KL divergence (X || Xhat)')
%title(['NMF KL divergence ' num2str(iter) ' iterations I ' num2str(I) ' J ' num2str(J) ])


format 'long'
display('gpu error')
display(gpu_error');
display('sequential error')
display(sequential_error');