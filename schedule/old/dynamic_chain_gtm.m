tf_model;

all_sequences=perms(get_contraction_dims(model));

states = [];

initstate = ElimState;
initstate.elimination_dims = get_contraction_dims(model);
[d initstate.index_list] = get_all_latent_factor_indices(model);
%initstate = initstate.gen_cost(model);

states = [states initstate];
laststate=0;

for s=1:size(all_sequences,1)
    parent=1; % reset parent

    for d=1:size(all_sequences,2)
        newstate = ElimState;

        newstate.elimination_dims = setdiff( all_sequences(s,:), ...
                                             all_sequences(s,1:d) );

        newstate.index_list = get_contracted_dims(model, ...
                                                  states(parent).index_list, ...
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
            newstate = newstate.gen_cost(model);
            states = [states newstate];
            states(parent).children = ...
                [ states(parent).children length(states) ];
            
            parent = length(states);

            % mark last state for backward message
            if length(newstate.elimination_dims) == 0
                laststate=length(states);
            end            
        else
            if sum( states(found).parents == parent ) == 0
                states(found).parents = [ states(found).parents ...
                                    parent ];

                states(parent).children = ...
                [ states(parent).children found ];

            end

            parent = found;
        end
    end
end



% calculate cumulative costs
states=min_backward_message(states, [laststate]);


s=1;
global_min_path=[];
while s~=laststate
    costs=[];
    inds=[];
    for i = 1:length(states(s).children)
        costs = [costs states(states(s).children(i)).cost];
        inds = [inds states(s).children(i)];
    end

    global_min_path = [global_min_path inds(find(costs == ...
                                                 min(costs)))];
    s=s+1;
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
c='';
for s=1:length(states)
    if s ~= 1
        l = [ l ',' ];
        c = [ c ',' ];
    end

    if length(states(s).elimination_dims) == 0
        l = [ l ' '''' ' ];
    else
        l = [ l ' ''' states(s).elimination_dims '''' ];
    end

    c=[c num2str(states(s).cost) ];
end

g='';
for i=1:length(global_min_path)
    if i ~= 1
        g = [ g ','];
    end
    g = [g num2str(global_min_path(i))];
end


system(['python treeplot.py "[' p ']" "[' l ']" "[' c ']" "[' g ']"' ]);