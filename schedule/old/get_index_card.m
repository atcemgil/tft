function num = get_index_card(gctf_model, index_char)

ind_chars=gctf_model{2};
ind_cards=gctf_model{3};
num = ind_cards(find(ind_chars==index_char));