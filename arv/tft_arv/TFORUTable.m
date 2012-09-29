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

        model_list;
    end


    methods 
        function obj = TFORUTable(model_list)
            obj.model_list = model_list;
            global line_cache numcache;
            line_cache = containers.Map();

            % find total number of permutations
            perm_count = 1;
            contraction_perms = repmat(TFORUTablePermSet, 1, 5);
            for mind = length(model_list):-1:1
                contraction_perms(mind) = ...
                    TFORUTablePermSet( perms(model_list(mind).get_contraction_dims()), ...
                                       perm_count, mind, model_list(mind) );

                perm_count = perm_count * ...
                    contraction_perms(mind).perm_num

                numcache(mind) = num2str(mind);
            end

            obj.costs = zeros(1, perm_count);






            % for each permutation calculate cost

            new_lt = containers.Map();
            for permi = 0:perm_count-1
                if mod(permi, 1000) == 0
                    permi / perm_count * 100
                end

                %line_temps = new_lt;
                %line_temps = zeros(1,20); % todo fix this number to
                %                          % correct temp count
                lti = 1;
                line_temps = {};

                % calculate temp_factors for each model for this
                % permutation set
                for mind = 1:length(model_list)
                    perm_ind = int32(mod( floor(permi/ ...
                                               contraction_perms(mind).perm_mod), ...
                                         contraction_perms(mind).perm_num ) ...
                                    + 1);
                    %gat = ...
                    %    contraction_perms(mind).all_temps{perm_ind};

                    %for gati = 1:length(gat)
                    %line_temps(gat(gati).name) = gat(gati).size;
                    %line_temps{lti:length(gat)} = gat(:).name;
                    %lti = lti+length(gat);
                    %end
                end

                %k = values(line_temps);
                
                %obj.costs(permi+1) = sum([k{:}]);
            end




            'min'
            [v permi] = min(obj.costs)
            for mind = 1:length(model_list)
                perm_ind = int32(mod( floor(permi/ ...
                                           contraction_perms(mind).perm_mod), ...
                                     contraction_perms(mind).perm_num ) ...
                                + 1);

                contraction_perms(mind).contraction_sequence_perms(perm_ind,:)
            end

            'max'
            [v permi] = max(obj.costs)
            for mind = 1:length(model_list)
                perm_ind = int32(mod( floor(permi/ ...
                                           contraction_perms(mind).perm_mod), ...
                                     contraction_perms(mind).perm_num ) ...
                                + 1);

                contraction_perms(mind).contraction_sequence_perms(perm_ind,:)
            end

        end


        function [all_temps] = get_all_temps(obj, model_ind, ...
                                                  contraction_sequence, ...
                                                  key)
            global line_cache numcache;

            %key = numcache(model_ind);
            %for i = 1:length(contraction_sequence)
            %    key = [ key '_' contraction_sequence.name ];
            %end

            if isKey(line_cache, key)
                all_temps = line_cache(key);
                return
            end



            tmp_model = obj.model_list(model_ind);
            all_temps = [];

            for ci = 1:length(contraction_sequence)
                tmp_model = tmp_model.contract(contraction_sequence(ci), ...
                                               'mem_analysis', ...
                                               '');
                tfs = tmp_model.temp_factors();
                for ti=1:length(tfs)
                    tfs(ti).get_element_size();
                end

                all_temps = [ all_temps  tfs ];
            end

            line_cache( key  ) = all_temps;
        end

    end

end