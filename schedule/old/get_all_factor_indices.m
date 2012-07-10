% given a tf_model returns indices of all factors in a cell

function [index_chars] = get_all_factor_indices(tf_model)

index_chars='';
for i = 1:length(tf_model{2})
    index_chars = [index_chars {tf_model{2}(i).index_char}];
end
