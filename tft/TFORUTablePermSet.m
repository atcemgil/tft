% Represents possible permutations related with a single line of TFORUTable

classdef TFORUTablePermSet

    properties
        % cell of permutations represented by this table line
        contraction_sequence_perms;

        % modulus operand for permutation set
        perm_mod;

        perm_num;

        % temporaries created by each contraction of each
        % contraction permutation
        all_temps;
    end

    methods
        function obj = TFORUTablePermSet(csp, current_perm_count, ...
                                         mind, model)
            if nargin ~= 0
                obj.contraction_sequence_perms = csp;
                obj.perm_mod = current_perm_count;
                obj.perm_num = size(csp, 1);

                obj.all_temps = {};
                orig_model = model;

                for i = 1:size(obj.contraction_sequence_perms, 1)
                    at = [];
                    model = orig_model;
                    for j = 1:length(obj.contraction_sequence_perms(i,:))
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
                        end

                        at = [ at  tfs ];
                    end
                    obj.all_temps{i} = at;
                end

            end
        end
    end
end