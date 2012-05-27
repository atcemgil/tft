tf_model;

all_sequences=perms(get_contraction_dims(model));

states = [];

initstate = ElimState;
initstate.elimination_dims = get_contraction_dims(model);
initstate.index_list = get_all_factor_indices(model);

states = [states initstate];

for s=1:size(all_sequences,1)
    parent=1; % reset parent

    for d=1:size(all_sequences,2)
        newstate = ElimState;

        newstate.elimination_dims = setdiff( all_sequences(s,:), ...
                                             all_sequences(s,1:d) );

        newstate.index_list = get_contracted_dims(states(parent).index_list, ...
                                                  all_sequences(s,d));

        found=0;
        for i=1:length(states)
            if states(i) == newstate
                found=i;
                break;
            end
        end

        if found == 0
            newstate.parents = [parent];
            states = [states newstate];
            parent = length(states);
        else
            if sum( states(found).parents == parent ) == 0
                states(found).parents = [ states(found).parents parent ];
            end

            parent = found;
        end
    end
end




p='[], ';

for s=2:length(states)
    p = [ p '[ ' ];
    for i =1:length(states(s).parents)
        if i ~= 1
            p = [ p ',' ];
        end
        p = [ p num2str(states(s).parents(i)) ];
    end
    p = [ p ' ] ' ];
    if s ~= length(states)
        p = [ p ' , ' ];
    end
end


l='';
for s=1:length(states)
    if s ~= 1
        l = [ l ',' ];
    end

    if length(states(s).elimination_dims) == 0
        l = [ l ' '''' ' ];
    else
        l = [ l ' ''' states(s).elimination_dims '''' ];
    end
end

system(['python treeplot.py "[' p ']" "[' l  ']"']);