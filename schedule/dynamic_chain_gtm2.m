tf_model2

all_sequences= ...
    perms(TFDimensionList2cell(tucker_model.get_contraction_dims));

initmodel = tucker_model;
models = [initmodel];

for s=1:size(all_sequences,1)

    parent=1; % reset parent

    for d=1:size(all_sequences,2)

        newmodel = models(parent).contract(all_sequences(s,d));

        found=0;
        for i=1:length(models)
            if models(i) == newmodel
                found=i;
                break;
            end
        end

        if found == 0
            newmodel.parent_tree_indices = [parent];
            newmodel.cost = newmodel.get_element_size;
            models = [models newmodel];
            models(parent).children_tree_indices = ...
                [ models(parent).children_tree_indices length(models) ];

            parent = length(models);

            % mark last model for backward message
            if length(newmodel.get_contraction_dims()) == 0
                lastmodel = length(models);
            end

        else
            if sum( models(found).parent_tree_indices == parent ) == 0
                models(found).parent_tree_indices = ...
                    [ models(found).parent_tree_indices parent ];

                models(parent).children_tree_indices = ...
                    [ models(parent).children_tree_indices found ];
            end

            parent = found;
        end
    end
end








% calculate cumulative costs
models=min_backward_message2(models, [lastmodel]);

s=1;
global_min_path=[];
while s~=lastmodel
    costs=[];
    inds=[];
    for i = 1:length(models(s).children_tree_indices)
        costs = [costs models(models(s).children_tree_indices(i)).cost];
        inds = [inds models(s).children_tree_indices(i)];
    end

    global_min_path = [global_min_path inds(find(costs == ...
                                                 min(costs)))];
    s=s+1;
end



p='[], ';

for s=2:length(models)
    p = [ p '[ ' ];
    for i =1:length(models(s).parent_tree_indices)
        if i ~= 1
            p = [ p ',' ];
        end
        p = [ p num2str(models(s).parent_tree_indices(i)) ];
    end
    p = [ p ' ] ' ];
    if s ~= length(models)
        p = [ p ' , ' ];
    end
end


l='';
c='';
for s=1:length(models)
    if s ~= 1
        l = [ l ',' ];
        c = [ c ',' ];
    end


    if length(models(s).get_contraction_dims()) == 0
        l = [ l ' '''' ' ];
    else
        l = [ l ' ''' ...
              char(TFDimensionList2cell(models(s).get_contraction_dims()))' '''' ];
        
    end

    c=[c num2str(models(s).cost) ];
end


g='';
for i=1:length(global_min_path)
    if i ~= 1
        g = [ g ','];
    end
    g = [g num2str(global_min_path(i))];
end


system(['python treeplot.py "[' p ']" "[' l ']" "[' c ']" "[' g ']"' ]);