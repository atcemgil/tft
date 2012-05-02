% orders given index_char array according to the order given in
% index_chars in the model
% returns ordered index char array and corresponding cardinality array

function [ordered_index_chars ordered_index_cards] = order_index_chars(model, index_chars)

% index_chars index_in_model;
tmp=zeros(length(index_chars), 3);
for i = 1:length(index_chars)
    tmp(i,1) = index_chars(i);
    tmp(i,2) = get_index_card(model, index_chars(i));
    tmp(i,3) = find(model{2}==index_chars(i));
end

tmp=sortrows(tmp,3);
ordered_index_chars=char(tmp(:,1)');
ordered_index_cards=tmp(:,2)';