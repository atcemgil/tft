% f is 1 indexed
function num = factor_numel(gctf_model, f)

factor_inds=get_factor_indices(gctf_model, f);
ind_cards=gctf_model{3};

num=1;
for i = 1:length(factor_inds)
    num = num * get_index_card(gctf_model, factor_inds(i));
end
