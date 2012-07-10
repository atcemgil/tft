% given a tf_model returns index characters used by the factors

function [index_chars] = get_index_characters(tf_model)

index_chars='';
for i = 1:length(tf_model{3})
    index_chars = [index_chars tf_model{3}(i).index_char];
end
