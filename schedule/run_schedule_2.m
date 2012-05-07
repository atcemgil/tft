
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
G=randi(rand_max,p,q,r);
F=randi(rand_max,p,q);


model_tucker3=struct('latent_factors', {'ABCG'}, ...
                     'input_factors', {'B'}, ...
                     'observed_factor', {'X'} )

model_rand=struct('latent_factors', {'AFH'}, ...
                  'input_factors', {'F'}, ...
                  'observed_factor', {'C'} )

%model_tucker3=struct('latent_factors', {'G'}, ...
%                     'input_factors', {'B'}, ...
%                     'observed_factor', {'X'} )
%
%model_rand=struct('latent_factors', {'AF'}, ...
%                  'input_factors', {'F'}, ...
%                  'observed_factor', {'C'} )
%


indices='ijkpqr';
index_cards=[i j k p q r];

factor_indices={'A', 'ip', 'B', 'jq', 'C' 'kr', 'G', 'pqr', 'F', ...
                'pq', 'X', 'ijk', 'H', 'ij'}

model={  ...
    {model_tucker3, model_rand}, ...
    indices, ...
    index_cards, ...
    factor_indices }

a=schedule_2(model);