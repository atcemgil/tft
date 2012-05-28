% given a cell of factor dimension characters and a dimension chracter to
% contract with returns a new cell with contracted dimension
% chracters

function [newdims] = get_contracted_dims(tf_model, dims, contract_char)

newdims={};

contracted_dim_chars=[];
count=1;
for i=1:length(dims)
    if sum( dims{i} == contract_char ) == 0
        % contract_char does not exist for this factor
        newdims{count} = dims{i};
        count = count + 1;
    else
        % contract_char exists, remove it and store other chars
        contracted_dim_chars = ...
            [ contracted_dim_chars ...
              setdiff(dims{i}, contract_char) ];
    end
end

newdims{count} = order_index_chars(tf_model, contracted_dim_chars);

newdims = unique(newdims);