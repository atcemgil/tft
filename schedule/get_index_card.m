function num = get_index_card(model, index_char)

ind_chars=cell2mat(model(2));
ind_cards=cell2mat(model(3));
num = ind_cards(find(ind_chars==index_char));