% orders given index_char array according to the order given in
% index_chars in the model
% returns ordered index char array and corresponding cardinality array

function [ordered_index_chars ordered_index_cards] = order_index_chars(tf_model, index_chars_array)

tmp=zeros(length(index_chars_array), 3);
default_order = get_default_order(tf_model);

for i = 1:length(index_chars_array)
    tmp(i,1) = index_chars_array(i);
    tmp(i,2) = get_index_card(tf_model, index_chars_array(i));
    tmp(i,3) = find(default_order==index_chars_array(i)); % order of the index
                                                          % in the model
end

tmp=sortrows(tmp,3); % sort by the order of the model indices
ordered_index_chars=char(tmp(:,1)');
ordered_index_cards=tmp(:,2)';