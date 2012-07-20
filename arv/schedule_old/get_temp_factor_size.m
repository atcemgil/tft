% returns element number of a (temporary) factor generated with
% given index_chars character array

function [size] = get_temp_factor_size(tf_model, index_chars)

size=1;
for i = 1:length(index_chars)
    size = size * get_index_card(tf_model, index_chars(i));
end
