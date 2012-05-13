% returns indices of a selected factor

function [inds] = get_factor_indices(gctf_model, factor_char)

factor_ind=find(not(cellfun('isempty', ...
                            strfind(gctf_model{4}, factor_char) )));
inds = char(gctf_model{4}( factor_ind + 1 ));

% if same factor appears in to models pick one
inds=inds(1,:);