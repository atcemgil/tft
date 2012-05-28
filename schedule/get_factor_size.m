% returns element number of a given factor of a tf_model

function [size] = get_factor_size(tf_model, factor_char)

factor_inds = get_factor_indices(tf_model, factor_char);

size=1;
for i = 1:length(factor_inds)
    size = size * get_index_card(tf_model, factor_inds(i));
end
