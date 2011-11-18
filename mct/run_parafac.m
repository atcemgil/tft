format 'compact'

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=10;
J=20;
K=30;

X = 10*rand(I,J,K);
X_cards = ['i','j','k'];

A_cards=['i','a']
B_cards=['j','a']
C_cards=['k','a']

V=['i','j','k','a']

a_range=1:3:10;
gpu_times = zeros(1,length(a_range));
sequential_times = zeros(1,length(a_range));

for i=1:length(a_range)
  a = a_range(i);
  display(['testing a ' num2str(a)]);

  cards=[I, J, K, a];

  tic; [A B C]=mct('pltf_gpu', 30, V, cards, X_cards, X, A_cards, B_cards, C_cards); gpu_times(i)=toc;
  test_parafac(A,B,C,I,J,K,a,X,'gpu');
  tic; [A B C]=mct('pltf_cpp', 30, V, cards, X_cards, X, A_cards, B_cards, C_cards); sequential_times(i)=toc;
  test_parafac(A,B,C,I,J,K,a,X,'sequential');

end

display(gpu_times);
display(sequential_times);
plot (a_range,sequential_times,'--', a_range, gpu_times, '-')
legend('Sequential code', 'GPU code')
xlabel('Second dimension cardinality of factors')
ylabel('seconds')
title('Sequential code vs GPU code (PARAFAC)')


