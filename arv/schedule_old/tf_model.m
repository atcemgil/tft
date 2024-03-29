% 1 : {1x(V*8)} cell
%     each cell must contain following
%     'latent_factors', 'input_factors', 'observed_factors',
%     'temp_factors'
% 2 : [1x\alpha] struct
%     one struct per factor containing
%     'factor_char', 'index_char'
% 3 : [1xN] struct 
%     one struct per model index containing
%     'index_char', 'cardinality'

% example

tucker_model={ 'latent_factors', {'A', 'B', 'C', 'G'}, ...
               'input_factors', {'C'}, ...
               'observed_factors', {'X'}, ...
               'temp_factors', {} };

model{1}={tucker_model};
model{2}=[ struct('factor_char', 'A', ...
                  'index_char', 'ip'), ...
           struct('factor_char', 'B', ...
                  'index_char', 'jq'), ...
           struct('factor_char', 'C', ...
                  'index_char', 'kr'), ...
           struct('factor_char', 'G', ...
                  'index_char', 'pqr'), ...
           struct('factor_char', 'X', ...
                  'index_char', 'ijk') ];

model{3}=[ struct('index_char', 'i', ...
                  'cardinality', 5), ...
           struct('index_char', 'j', ...
                  'cardinality', 6), ...
           struct('index_char', 'k', ...
                  'cardinality', 7), ...
           struct('index_char', 'p', ...
                  'cardinality', 8), ...
           struct('index_char', 'q', ...
                  'cardinality', 9), ...
           struct('index_char', 'r', ...
                  'cardinality', 10) ];

