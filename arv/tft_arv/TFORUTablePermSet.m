% Represents possible permutations related with a single line of TFORUTable

classdef TFORUTablePermSet

    properties
        contraction_sequence_perms;

        % modulus operand for permutation set
        perm_mod;

        perm_num;

        line_perms_key;

        all_temps;
    end

    methods
        function obj = TFORUTablePermSet(csp, current_perm_count, ...
                                         mind, model)
            if nargin ~= 0
                obj.contraction_sequence_perms = csp;
                obj.perm_mod = current_perm_count;
                obj.perm_num = size(csp, 1);

                obj.line_perms_key = {};
                for i = 1:size(obj.contraction_sequence_perms,1)
                    obj.line_perms_key{i} = num2str(mind);
                    for j = 1:size(obj.contraction_sequence_perms,2)
                        obj.line_perms_key{i} = [ ...
                            num2str(obj.line_perms_key{i}) ...
                            '_' obj.contraction_sequence_perms(i,j).name ...
                            ];
                    end
                end


                obj.all_temps = {};

                for i = 1:size(obj.contraction_sequence_perms, 1)
                    at = [];
                    for j = 1:length(obj.contraction_sequence_perms(i,:))
                        model = model.contract(obj.contraction_sequence_perms(i, j), ...
                                               'mem_analysis', ...
                                               '');
                        tfs = model.temp_factors();
                        for ti=1:length(tfs)
                            tfs(ti).get_element_size();
                        end
                        
                        at = [ at  tfs ];
                    end
                    obj.all_temps{i} = at;
                end
            end
        end
    end

end