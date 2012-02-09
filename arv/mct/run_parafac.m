format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=2;
J=3;
K=4;
a=2; %????

I=7;
J=8;
K=9;
A=10;

V_card_sym=['i','j','k','a']
V_cards=[I, J, K, a];

A_card_sym=['i','a']
A_true = round(10*rand(I,a));

B_card_sym=['j','a']
B_true = round(20*rand(J,a));

C_card_sym=['k','a']
C_true = round(30*rand(K,a));

X_card_sym = ['i','j','k'];
X_true = get_parafac(A_true,B_true,C_true,I,J,K,a,[I J K])

X = poissrnd(X_true)

iter=1;

display(['testing iteration ' num2str(iter)]);

tic; [A B C]=pltf_cpp(iter, V_card_sym, V_cards, X_card_sym, X, A_card_sym, B_card_sym, C_card_sym); sequential_times=toc;
sequential_error = sum(get_KL_div(X, get_parafac(A,B,C,I,J,K,a,size(X))));

%tic; [A B C]=mct('pltf_gpu', iter, V_card_sym, V_cards, X_card_sym, X, A_card_sym, B_card_sym, C_card_sym); gpu_times=toc;
gpu_error = sum(get_KL_div(X, get_parafac(A,B,C,I,J,K,a,size(X))));


%display(gpu_times);
display(sequential_times);

%display(gpu_error);
display(sequential_error);