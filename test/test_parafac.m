%rand('state',0);
rng(1)

I=7;
J=8;
K=9;
A=10;

I=3;
J=4;
K=5;
A=2;


% I=10;
% J=11;
% K=12;
% A=10;

V_card_sym=['i','j','k','a'];
V_cards=[I, J, K, A];

A_card_sym=['i','a'];
A_true = round(10*rand(I,1,1,A));

B_card_sym=['j','a'];
B_true = round(20*rand(1,J,1,A));

C_card_sym=['k','a'];
C_true = round(30*rand(1,1,K,A));

X_card_sym = ['i','j','k'];
X_true = get_parafac(A_true,B_true,C_true,I,J,K,A,[I J K]);

X = poissrnd(X_true);
X(X==0)=0.000001; % suppress zeros, division/log problems, not the best method


init=0;

if init == 1
    A_init = rand(I,A);
    B_init = rand(J,A);
    C_init = rand(K,A);
else
    A_init = [];
    B_init = [];
    C_init = [];
end

iter_num=2;
kl_parafac_seq = zeros(1,length(1:iter_num));
kl_parafac_par = zeros(1,length(1:iter_num));

for i = [ 1:iter_num ]

    tic; [factor_A factor_B factor_C] = pltf_par ( i, ...
                                                   V_card_sym, V_cards, ...
                                                   X_card_sym, X, ...
                                                   A_card_sym, A_init, ...
                                                   B_card_sym, B_init, ...
                                                   C_card_sym, C_init ); toc;
    kl_parafac_seq(i) = get_KL_div(X, get_parafac(factor_A,factor_B,factor_C,I,J,K,A,size(X)))
end
plot ( [1:iter_num ], kl_parafac_seq)

%tic; [factor_A factor_B factor_C] = pltf_par ( iter_num, ...
%                                               V_card_sym, V_cards, ...
%                                               X_card_sym, X, ...
%                                               A_card_sym, A_init, ...
%                                               B_card_sym, B_init, ...
%                                               C_card_sym, C_init ); toc;
%get_KL_div(X, get_parafac(factor_A,factor_B,factor_C,I,J,K,A,size(X)))




% iter_num=100;
% tic; [factor_A factor_B factor_C] = pltf_seq ( iter_num, V_card_sym, V_cards, X_card_sym, X, ...
%                                                A_card_sym, B_card_sym, C_card_sym); toc;
% get_KL_div(X, get_parafac(factor_A,factor_B,factor_C,I,J,K,A,size(X)))
% 
% 
% tic; [factor_A factor_B factor_C] = pltf_par ( iter_num, V_card_sym, V_cards, X_card_sym, X, ...
%                                                A_card_sym, B_card_sym, C_card_sym); toc;
% get_KL_div(X, get_parafac(factor_A,factor_B,factor_C,I,J,K,A,size(X)))
