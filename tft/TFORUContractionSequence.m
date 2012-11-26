% Represents possible permutations related with a single line of TFORUTable

classdef TFORUContractionSequence

    properties
        % cell of permutations represented by this table line
        contraction_sequence_perms;

        % temporaries created by each contraction of each
        % contraction permutation
        all_temps;

        % cost of each path for each GTM
        costs;
    end

    methods
        function obj = TFORUContractionSequence(model)
            if nargin ~= 0
                obj.contraction_sequence_perms = perms(model.get_contraction_dims());

                obj.all_temps = {};
                orig_model = model;

                for i = 1:size(obj.contraction_sequence_perms, 1)
                    at = [];
                    atc = 0;
                    model = orig_model;
                    % -1 for last contraction will output to non-temporary
                    for j = 1:(length(obj.contraction_sequence_perms(i,:))-1)
                        model = model.contract(obj.contraction_sequence_perms(i, j), ...
                                               'mem_analysis', ...
                                               '');
                        tfs = model.temp_factors();
                        for ti=1:length(tfs)
                            tfs(ti).size = tfs(ti).get_element_size();
                            %display([tfs(ti).name ' size '
                            %num2str(tfs(ti).size)]);

                            % change factor name to globally
                            % identifyable name
                            % ( all models have same dims)
                            coded_name = ...
                                model.get_coded_factor_name(tfs(ti));
                            tfs(ti).name = coded_name;

                            atc = atc + tfs(ti).size();
                        end

                        at = [ at  tfs ];
                    end
                    obj.all_temps{i} = at;
                    %display(['atc ' num2str(atc)]);
                    obj.costs{i} = atc;
                end

            end
        end
    end
end