format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=2
J=3
K=4

V=['i','k','j'];

A_cards=['i','k'];
A_true=round(10*rand(I,K));

B_cards=['k','j'];
B_true=round(20*rand(K,J));

X_cards=['i','j'];
X_true=A_true*B_true;

X=poissrnd(X_true);

iter_range = 1:50;
gpu_times = zeros(1,length(iter_range));
sequential_times = zeros(1,length(iter_range));
gpu_error = zeros(1,length(iter_range));
sequential_error = zeros(1,length(iter_range));

for i=1:length(iter_range)
  iter = iter_range(i);
  display(['testing iteration ' num2str(iter)]);

  cards=[I, K, J];

  tic; [A B]=mct('pltf_gpu', iter, V, cards, X_cards, X_true, A_cards, B_cards); gpu_times(i)=toc;
  gpu_error(i) = sum(get_KL_div(X_true, reshape(A,I,K)*reshape(B,K,J)));

end

display(gpu_times);
plot (iter_range, gpu_times, '-')
legend('GPU code')
xlabel('Number of iterations')
ylabel('seconds')
title('GPU code (NMF)')

figure

display(gpu_error);
plot (iter_range, gpu_error, '-')
legend('GPU code')
xlabel('Number of iterations')
ylabel('KL divergence')
title('GPU code (NMF)')
