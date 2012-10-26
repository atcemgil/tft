% Represents an Optimal temporary tensor Re-Use graph.
%
%   For a given set of GTM operations (as an array of TFModel),
%   TFORU table generates all possible contraction sequences over
%   all GTM operations. Then this table can be searched for the
%   least memory using path ie: line with least cost column 
%


classdef TFORUTable

    properties
        % array of costs
        costs=[];

        contraction_perms;
    end


    methods 
        function obj = TFORUTable(model_list)
            % find total number of permutations
            perm_count = 1;
            obj.contraction_perms = repmat(TFORUTablePermSet, 1, length(model_list));
            for mind = length(model_list):-1:1
                obj.contraction_perms(mind) = ...
                    TFORUTablePermSet( perms(model_list(mind).get_contraction_dims()), ...
                                       perm_count, mind, model_list(mind) );
                perm_count = perm_count * ...
                    obj.contraction_perms(mind).perm_num
            end
            obj.costs = zeros(1, perm_count);



            % for each permutation calculate cost
            for permi = 0:perm_count-1
                if mod(permi, 1000) == 0
                    display(['complete ' sprintf('%5.2f', (permi / perm_count ...
                                                 * 100)) ' %']);
                end

                line_temps = containers.Map();

                % calculate temp_factors for each model for this
                % permutation set
                for mind = 1:length(model_list)
                    perm_ind = int32(mod( floor(permi/ ...
                                               obj.contraction_perms(mind).perm_mod), ...
                                         obj.contraction_perms(mind).perm_num ) ...
                                    + 1);
                    gat = ...
                        obj.contraction_perms(mind).all_temps{perm_ind};

                    for gati = 1:length(gat)
                        line_temps(gat(gati).name) = gat(gati).size;
                    end
                end

                k = values(line_temps);
                obj.costs(permi+1) = sum([k{:}]);
            end



            'min'
            [v permi] = min(obj.costs)
            obj.display_path(model_list, permi);

            'max'
            [v permi] = max(obj.costs)
            obj.display_path(model_list, permi);
        end




        function [] = display_path(obj, model_list, permi)
            line_temps = containers.Map();
            for mind = 1:length(model_list)
                perm_ind = int32(mod( floor(permi/ ...
                                           obj.contraction_perms(mind).perm_mod), ...
                                     obj.contraction_perms(mind).perm_num ) ...
                                + 1);

                contract_dims = ...
                    obj.contraction_perms(mind) ...
                    .contraction_sequence_perms(perm_ind,:);

                display(['selected contract dims for model ' ...
                         num2str(mind) ' for permindex ' num2str(permi) ...
                         ' ']);
                str = '';
                for i = 1:length(contract_dims)
                    if i ~= 1
                        str = [str ','];
                    end
                    str = [str contract_dims(i).name];
                end

                gat = ...
                    obj.contraction_perms(mind).all_temps{perm_ind};

                str = [ str ' temp factors: ' ];
                for gati = 1:length(gat)
                    line_temps(gat(gati).name) = gat(gati).size;
                    str = [str ' (' gat(gati).name ') '];
                end

                display([char(str)]);
            end
            k = values(line_temps);
            display(['cost ' num2str(sum([k{:}])) char(10)]);
        end

    end

end