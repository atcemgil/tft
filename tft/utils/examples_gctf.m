% Examples for Tensor Factorization Toolbox
% for details see index.html page

% tucker_model

dim_i=TFDimension('name', 'i', 'cardinality', 5);
dim_j=TFDimension('cardinality', 6, 'name', 'j');
dim_k=TFDimension('cardinality', 7, 'name', 'k');
dim_p=TFDimension('cardinality', 8, 'name', 'p');
dim_q=TFDimension('cardinality', 9, 'name', 'q');
dim_r=TFDimension('cardinality', 10, 'name', 'r');

A=TFFactor('name', 'A', 'type', 'latent', 'dims', [dim_i dim_p]);
B=TFFactor('name', 'B', 'type', 'latent', 'dims', [dim_j dim_q]);
C=TFFactor('name', 'C', 'type', 'latent', 'dims', [dim_k dim_r], ...
           'isClamped', true);
G=TFFactor('name', 'G', 'type', 'latent', 'dims', ...
           [dim_p, dim_q, dim_r]);
X=TFFactor('name', 'X', 'type', 'observed', 'dims', ...
           [dim_i, dim_j, dim_k]);

%tucker_model = PLTFModel('name', 'Tucker3', 'factors', [A B C G X], ...
%                       'dims', [dim_i dim_j dim_k dim_p dim_q dim_r]);


% parafac model

p_X=TFFactor('name', 'p_X', 'type', 'observed', 'dims', ...
             [dim_i, dim_j, dim_k]);

p_A=TFFactor('name', 'p_A', 'type', 'latent', 'dims', [dim_i dim_r]);
p_B=TFFactor('name', 'p_B', 'type', 'latent', 'dims', [dim_j dim_r]);
%p_C=TFFactor('name', 'p_C', 'type', 'latent', 'dims', [dim_k dim_r]);

%parafac_model = PLTFModel('name', 'Parafac', ...
%                        'factors', [p_A p_B C p_X], ...
%                        'dims', [dim_i dim_j dim_k dim_r]);


tucker_parafac_model = GCTFModel( ...
    'name', 'tucker3_parafac', ...
    'dims', [dim_i dim_j dim_k dim_p dim_q dim_r] , ...
    'observed_factors', [ X p_X ], ...
    'R', { [A B C G], [p_A p_B C] } );

tucker_parafac_model.rand_init_latent_factors('all');

X.rand_init(tucker_parafac_model.dims, 100) % init observation
p_X.rand_init(tucker_parafac_model.dims, 100) % init observation
C.rand_init(tucker_parafac_model.dims, 100) % init clamped
