
% parafac + paracan

format ('compact')
clear

I=3;
J=4;
K=5;
A=2;

iter_num=10;

V_card_sym=['i','j','k','a'];
V_cards=[I, J, K, A];

p=-1;

R=[ 1, 1, 1; ...
    1, 1, 0 ];

Z1_card_sym=['i','a'];
Z1_true = round(10*rand(I,1,1,A));

Z2_card_sym=['j','a'];
Z2_true = round(20*rand(1,J,1,A));

Z3_card_sym=['k','a'];
Z3_true = round(30*rand(1,1,K,A));

X1_card_sym = ['i','j','k'];
X1_true = get_parafac(Z1_true,Z2_true,Z3_true,I,J,K,A,[I J K]);
X1 = poissrnd(X1_true);
X1(X1==0)=0.000001; % suppress zeros, division/log problems, not the best method

X2_card_sym = ['i','j'];
X2_true = get_paracan(Z1_true,Z2_true,I,J,A,[I J]);
X2 = poissrnd(X2_true);
X2(X2==0)=0.000001; % suppress zeros, division/log problems, not the best method

updateZ1=0;
updateZ2=1;
updateZ3=1;

kl_parafac_seq = zeros(1, length(1:iter_num))
kl_paracan_seq = zeros(1, length(1:iter_num))

init_z1=rand(size(Z1_true))
init_z2=rand(size(Z2_true))
init_z3=rand(size(Z3_true))

for i = [ 1:iter_num ]
%i=iter_num

    tic; [factor_A factor_B factor_C] = gctf_seq ( i, ...
                                                   V_card_sym, ...
                                                   V_cards, ...
                                                   p, ...
                                                   R, ...
                                                   X1_card_sym, X1, ...
                                                   X2_card_sym, X2, ...
                                                   Z1_card_sym, init_z1, updateZ1, ...
                                                   Z2_card_sym, init_z2, updateZ2, ...
                                                   Z3_card_sym, init_z3, updateZ3 ...
                                                   );
    toc

    kl_parafac_seq(i)= get_KL_div(X1, get_parafac(factor_A, factor_B,factor_C,I,J,K,A,size(X1)));
    kl_paracan_seq(i)= get_KL_div(X2, get_paracan(factor_A,factor_B,I,J, A,size(X2)));
end

plot ( [1:iter_num ], kl_parafac_seq)
figure
plot ( [1:iter_num ], kl_paracan_seq)



kl_parafac_par = zeros(1, length(1:iter_num))
kl_paracan_par = zeros(1, length(1:iter_num))

for i = [ 1:iter_num ]

    tic; [factor_A factor_B factor_C] = gctf_par ( i, ...
                                                   V_card_sym, ...
                                                   V_cards, ...
                                                   p, ...
                                                   R, ...
                                                   X1_card_sym, X1, ...
                                                   X2_card_sym, X2, ...
                                                   Z1_card_sym, init_z1, updateZ1, ...
                                                   Z2_card_sym, init_z2, updateZ2, ...
                                                   Z3_card_sym, init_z3, updateZ3 ...
                                                   );
    toc

    kl_parafac_par(i)= get_KL_div(X1, get_parafac(factor_A, factor_B,factor_C,I,J,K,A,size(X1)));
    kl_paracan_par(i)= get_KL_div(X2, get_paracan(factor_A,factor_B,I,J, A,size(X2)));
end

figure
plot ( [1:iter_num ], kl_parafac_par)
figure
plot ( [1:iter_num ], kl_paracan_par)


kl_parafac_seq
kl_paracan_seq
kl_parafac_par
kl_paracan_par