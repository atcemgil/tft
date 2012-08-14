% sends backward (child to parents) message calculating cumulative
% costs

function [models] = min_backward_message2(models, lastmodel)

done=[];
%min_state_inds=[];
while lastmodel ~= 1

    for l=1:length(lastmodel)

        for p=1:length(models(lastmodel(l)).parent_tree_indices)

            if sum (done == models(lastmodel(l)).parent_tree_indices(p)) == 0
                costs=[];
                inds=[];
                for c=1:length(models(models(lastmodel(l)).parent_tree_indices(p)).children_tree_indices)
                    costs = [costs ...
                             models(models(models(lastmodel(l)) ...
                                           .parent_tree_indices(p)).children_tree_indices(c)).cost ];
                    inds = [inds models(models(lastmodel(l)).parent_tree_indices(p)).children_tree_indices(c)];
                end

                %min_state_inds = [ min_state_inds inds(find(min(costs)==costs)) ];

                models(models(lastmodel(l)).parent_tree_indices(p)).cost = ...
                    models(models(lastmodel(l)).parent_tree_indices(p)).cost + ...
                    min(costs);
                done = [done models(lastmodel(l)).parent_tree_indices(p)];
            end
        end
    end

    lastmodel2=[];
    for l=1:length(lastmodel)
        lastmodel2 = [lastmodel2 models(lastmodel(l)).parent_tree_indices];
    end
    lastmodel=unique(lastmodel2);

end
