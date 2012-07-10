% given a tf_model, model number (assumed 1 if not given) 
% returns contraction dimension characters

function [contract_dims] = get_contraction_dims(tf_model, model_num)

if nargin==1
    model_num=1;
end

index_chars=get_index_characters(tf_model);

models = tf_model{1};
output_factor = char(models{model_num}{3*2});
output_chars = get_factor_indices(tf_model, output_factor);

contract_dims='';
for i = 1:length(index_chars)
    if sum( index_chars(i) == output_chars ) == 0
        contract_dims = [contract_dims index_chars(i)];
    end
end
