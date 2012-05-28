classdef ElimState
    properties
        elimination_dims='';
        index_list={};
        cost=0;

        % index of parent ElimState in the array
        parents=[];
        children=[];
    end

    methods
        function r = eq(a,b)
            r=logical(1);
            if strcmp(a.elimination_dims, b.elimination_dims)
                for i = 1:length(a.index_list)
                    found = 0;
                    for j = 1:length(b.index_list)
                        if strcmp(a.index_list{i}, b.index_list{j})
                            found=1;
                            break
                        end
                    end
                    % one index of a does not exist in b
                    if found == 0
                        r=logical(0)
                        return
                    end
                end
            else
                % elimination dimensions are different
                r=logical(0);
            end
        end

        function [obj] = gen_cost(obj, tf_model)
            obj.cost=0;
            for i = 1:length(obj.index_list)
                obj.cost = obj.cost + get_temp_factor_size(tf_model, obj.index_list{i});
            end
        end
    end
end
