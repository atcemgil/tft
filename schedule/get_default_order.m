% return character array of all indices as given in the model

function [inds] = get_default_order(tf_model)

inds=[];
for i=1:length(tf_model{3})
    inds = [ inds tf_model{3}(i).index_char ];
end
