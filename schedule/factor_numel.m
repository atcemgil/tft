% f is 1 indexed
function num = factor_numel(model, f)

factor_inds=cell2mat(model(4+f));
ind_cards=cell2mat(model(3));

num=1;
for i = 1:length(factor_inds)
    num = num * get_index_card(model, factor_inds(i));
end