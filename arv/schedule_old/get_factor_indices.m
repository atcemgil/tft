% returns indices of a selected factor

function [inds] = get_factor_indices(tf_model, factor_char)

factor_indices=tf_model{2};

for i = 1:length(factor_indices)
    if factor_indices(i).factor_char == factor_char
        inds = factor_indices(i).index_char;
        return
    end
end
