format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=3;
J=4;
K=5;
a=2;

V=['i','j','k','a']

A_cards=['i','a']
A_true = round(10*rand(I,a));

B_cards=['j','a']
B_true = round(20*rand(J,a));

C_cards=['k','a']
C_true = round(30*rand(K,a));

X_cards = ['i','j','k'];
X_true = get_parafac(A_true,B_true,C_true,I,J,K,a,[I J K])

X = poissrnd(X_true)


iter_range = 1:11;
gpu_times = zeros(1,length(iter_range));
sequential_times = zeros(1,length(iter_range));
gpu_error = zeros(1,length(iter_range));
sequential_error = zeros(1,length(iter_range));

for i=1:length(iter_range)
  iter = iter_range(i);
  display(['testing iteration ' num2str(iter)]);

  cards=[I, J, K, a];

  tic; [A B C]=mct('pltf_cpp', iter, V, cards, X_cards, X_true, A_cards, B_cards, C_cards); gpu_times(i)=toc;
  gpu_error(i) = sum(get_KL_div(X_true, get_parafac(A,B,C,I,J,K,a,size(X_true))));

end

display(gpu_times);
plot (iter_range, gpu_times, '-')
legend('GPU code')
xlabel('Number of iterations')
ylabel('seconds')
title('GPU code (PARAFAC)')

figure

display(gpu_error);
plot (iter_range, gpu_error, '-')
legend('GPU code')
xlabel('Number of iterations')
ylabel('KL divergence')
title('GPU code (PARAFAC)')
