% returns cardinality of an index

function [card] = get_index_card(tf_model, index_char)

for i = 1:length(tf_model{3})
    if tf_model{3}(i).index_char == index_char
        card = tf_model{3}(i).cardinality;
        return
    end
end
