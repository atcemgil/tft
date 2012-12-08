% Represents possible permutations related with a single line of TFORUTable

classdef TFORUContractionSequence

    properties
        % cell of permutations represented by this table line
        contraction_sequence_perms;

        % temporaries created by each contraction of each
        % contraction permutation
        all_temps;

        % cost of each path for each path in this GTM
        mem_costs;

        % computational cost for each path in this GTM
        cmt_output_names = {};
        % indexed by output_names
        cmt_input_names = {};

        % stores cost of cmt s
        cmt_computation_costs;
    end

    methods
        function obj = TFORUContractionSequence(model)
            if nargin ~= 0
                obj.cmt_computation_costs = containers.Map(); % amazing! if initialized as property appears to be globally accessible!

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
                    obj.mem_costs{i} = atc;
                end



                % store cmt i/o names for each contraction
                for i = 1:size(obj.contraction_sequence_perms, 1)
                    model = orig_model;
                    on = {};
                    in = {};
                    jlimit = length(obj.contraction_sequence_perms(i,:));
                    for j = 1:jlimit
                        model = model.contract(obj.contraction_sequence_perms(i, j), ...
                                               'mem_analysis', ...
                                               '');
                        % intermediate contractions are stored in temporary tensors
                        tfs = model.temp_factors();
                        for ti=1:length(tfs)
                            % last contraction is stored to first observed (non-latent) factor
                            if j == jlimit
                                on{end+1} = model.observed_factor().name;
                                obj.cmt_computation_costs( char(on{end}) ) = model.observed_factor().get_element_size();

                                %on{end}
                                %model.observed_factor()
                                %model.observed_factor().get_element_size()
                            else
                                on{end+1} = tfs(ti).get_short_name();
                                obj.cmt_computation_costs( char(on{end}) ) = tfs(ti).get_element_size();

                                %on{end}
                                %tfs(ti)
                                %tfs(ti).get_element_size()
                            end

                            in{end+1} = { tfs(ti).source_factor_names };
                        end
                    end
                    obj.cmt_output_names{i} = on;
                    obj.cmt_input_names{i} = in;
                end
            end
        end
    end
end