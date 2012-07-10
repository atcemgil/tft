
rng(1)

i=2;
j=3;
k=4;
p=5;
q=6;
r=7;

rand_max=10;

A=randi(rand_max,i,p);
B=randi(rand_max,j,q);
C=randi(rand_max,k,r);
D=randi(rand_max,p,q,r);



factor_chars={ 'A', 'B', 'C', 'G' };
index_chars='ijkpqr';
index_cards=[i j k p q r];
output_indices='ijk';
factor1_indices='ip';
factor2_indices='jq';
factor3_indices='kr';
factor4_indices='pqr';

model={ factor_chars, index_chars, index_cards, output_indices, ...
        factor1_indices, factor2_indices, factor3_indices, factor4_indices ...
        }

schedule_1(model)