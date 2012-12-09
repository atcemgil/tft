% Represents an Optimal temporary tensor Re-Use graph.
%
%   For a given set of GTM operations (as an array of TFModel),
%   TFORU graph generates all possible contraction sequences over
%   all GTM operations. Then this graph can be searched for the
%   least memory using path ie: contraction path with optimal
%   temporary tensor re-use.
%
%   See also TFModel

classdef TFORUGraph

    properties
        contraction_perms;
    end

    methods

        function obj = TFORUGraph(model_list)
        % assume first model is Xhat update
        % rest are latent factor updates


            % model list -> augmented model list
            % ( hatX, Z_1, Z_2, Z_3 ) -> ( hatX, Z_1, hatX, Z2, hatX, Z_3 )
            orig_model_list = model_list
            model_list = [];
            for i = 2:length(orig_model_list)
                model_list = [ model_list orig_model_list(1) ];
                model_list = [ model_list orig_model_list(i) ];
            end
                


            obj.contraction_perms = repmat(TFORUContractionSequence, 1, length(model_list));
            for mind = 1:length(model_list)
                obj.contraction_perms(mind) = TFORUContractionSequence( model_list(mind) );
            end


            %latent_num = length(obj.contraction_perms)-1;
            %max_level = latent_num*2;
            max_xhat_path_num = size(obj.contraction_perms(1).contraction_sequence_perms,1);
            model_num = length(obj.contraction_perms);

            global cumulative_card cumulative_cards
            % make an array of number of paths for factors
            cumulative_cards = zeros(1, model_num);

            cumulative_card = 0;
            for m = 1:model_num
                if m == 1
                    cumulative_cards(m) = size(obj.contraction_perms(m).contraction_sequence_perms, 1);
                else
                    cumulative_cards(m) = cumulative_cards(m-1) + size(obj.contraction_perms(m).contraction_sequence_perms, 1);
                end

                cumulative_card = cumulative_card + size(obj.contraction_perms(m).contraction_sequence_perms, 1);
            end


            % convinience memory cost array
            global path_mem_costs path_output_names path_input_names
            path_mem_costs = ones(1, cumulative_card) * Inf;
            path_output_names = {};
            path_input_names = {};
            i=1;
            for li = 1:model_num
                for ci = 1:size(obj.contraction_perms(li).contraction_sequence_perms,1)
                    cost = 0;
                    for cj = 1:size(obj.contraction_perms(li).contraction_sequence_perms,2)
                        output_name = obj.contraction_perms(li).cmt_output_names{ci}{cj};
                        cost = cost + obj.contraction_perms(li).cmt_computation_costs( output_name );
                    end
                    path_mem_costs(i) = cost;

                    path_output_names{i} = obj.contraction_perms(li).cmt_output_names{ci};
                    path_input_names{i} = obj.contraction_perms(li).cmt_input_names{ci};
                    i = i+1;
                end
            end

            path_mem_costs
            path_output_names
            path_input_names














            global S costs;
            S = {};
            costs = {};
            % insert first level elements
            for i = 1:size(obj.contraction_perms(1).contraction_sequence_perms,1)
                S{i} = [i zeros(1, model_num) ];
                costs{i} = path_mem_costs(i);
            end
            bestsofar = Inf;
            best_solution = {};

            node_num = 0;            % nodes added to S as subproblem
            pruned_node_num = 0; % nodes not added to S as subprobelm
            while length( S )
                % remove a subproblem from tail of S, tail seems to be more efficient
                p = S{end};
                S = S(1:end-1);

                p_cost = costs{end};
                costs = costs(1:end-1);





                % expand p into smaller subproblems

                % generate children of p
                zero_childs = find( p == 0 );
                first_zero_child_index = zero_childs(1);

                % add subproblems by incrementing in first zero child GTM

                % calculate child_index_range
                if mod(first_zero_child_index, 2) == 1
                    child_index_range = 1:max_xhat_path_num;
                else
                    % must not include indices of latent factors which are selected in previous paths
                    child_index_range = [];
                    prev_latent_inds = (first_zero_child_index-2):-2:2;
                    % search for each factor in previous indices
                    % if not found insert to child_index_range
                    for mi = 2:2:model_num % even ones are latent indices
                        found = false;
                        for peii = 1:length(prev_latent_inds)
                            % identify factor used in this previous path
                            if p(prev_latent_inds(peii)) > cumulative_cards(mi) && ...
                                    p(prev_latent_inds(peii)) <= cumulative_cards(mi+1)
                                found = true;
                                break;
                            end
                        end

                        if found == false
                            child_index_range = [child_index_range ...
                                                [(cumulative_cards(mi-1)):cumulative_cards(mi) ] ];
                        
                        end
                    end
                end

                %display(['child_index_range ' num2str(child_index_range)]);



                for cind = child_index_range
                    node_num = node_num + 1;
                    if mod(node_num, 100000) == 0
                        display([num2str(node_num) ' ' num2str(pruned_node_num) ' ' num2str(length(S)) char(9) 'p ' num2str(p) ' cost: ' num2str(p_cost)])
                    end


                    p_new = p;
                    p_new(first_zero_child_index) = cind;


                    % calcuate p_new_cost
                    p_new_cost = p_cost + path_mem_costs(cind);


                    % for each output factor of this path
                    for lpoind = 1:length(path_output_names{cind})
                        % check all previous paths
                        for pi = 1:(first_zero_child_index-1)
                            poutput_names = path_output_names{ p_new(pi) };
                            pinput_names = path_input_names{ p_new(pi) };
                            for pki = 1:length(poutput_names)
                                % if this previous path has this output factor
                                if strcmp(poutput_names(pki), path_output_names{cind}{lpoind})
                                    %display(['possible reuse ' char(poutput_names(pki))]);
                                    found = false;
                                    % check if input of previous calculation are clean

                                    % ASSUME inputs cannot be updated in path pi (they originate from)
                                    % IF sub-contraction is considered this may not be valid!

                                    clean = true;
                                    for check_prange = (pi+1):(first_zero_child_index-1) % for each following path
                                        next_poutput_names = path_output_names{ p_new(check_prange) };
                                        next_pinput_names = path_input_names{ p_new(check_prange) };
                                        for npoutni = 1:length(next_poutput_names) % for each output of this path
                                            for npinni = 1:length(next_pinput_names{npoutni}) % for each input of this output
                                                for pini = 1:length(pinput_names{pki}) % for each input of possible reuse calculation
                                                    if strcmp( pinput_names{pini}, pinput_names{pki}{pini})
                                                        display(['reuse abort']);
                                                        clean = false;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                    if clean
                                        %p_new
                                        %display(['reuse detected ' char(poutput_names(pki))]);
                                    else
                                        display(['dirty reuse aborted ' char(poutput_names(pki))]);
                                    end
                                end
                            end
                        end
                    end






                    % if p_new is a complete solution update bestsofar
                    if sum( p_new == 0 ) == 0
                        % siblings arrive here at the same iteration so must check
                        if bestsofar > p_new_cost
                            bestsofar = p_new_cost
                            best_solution = p_new
                        end
                    elseif p_new_cost < bestsofar
                        %display(['add for inspection ' num2str(p_new_cost)]);
                        % else if p_new is still viable add to S for further inspection
                        S{end+1} = p_new;
                        costs{end+1} = p_new_cost;
                    else
                        pruned_node_num = pruned_node_num + 1;
                        %p_new
                    end
                end
            end

            display(['considered ' num2str(node_num) ' number of nodes'])
            display(['pruned ' num2str(pruned_node_num) ' number of nodes'])
            bestsofar
            best_solution
        end

        



        %function [n] = nodes_number_under_subproblem(obj, p)
        % indices which are zero get pruned so count them
        %end










        function [str] = print_dot(obj, filename)
        % generates dot string of this graph

            str = [ 'digraph structs{' char(10) ...
                    'node [shape=plaintext];' char(10) ...
                    'splines=false; ' char(10)];

            for nid = 1:length(obj.nodes)
                str = [ str 'struct' num2str(nid) ' [label=< <TABLE FIXEDSIZE="FALSE" CELLBORDER="0" STYLE="ROUNDED"><TR><TD>' ...
                        obj.nodes(nid).model.name ...
                        char(10) '</TD></TR> <HR/> <TR><TD FIXEDSIZE="FALSE">' ...
                        char(TFDimensionList2cell(obj.nodes(nid).contraction_dims))' '</TD></TR> <TR><TD FIXEDSIZE="FALSE"></TD></TR></TABLE> >];' ...
                        char(10) 
                      ];
            end

            fid = fopen(filename,'w');
            myformat = '%s';
            fprintf(fid, myformat, str);

            [is,js,~] = find(obj.edges);
            myformat = ['struct%u -> struct%u [ label = "%u"]' char(10) ];
            for ind = 1:length(is)

                fprintf(fid, myformat, ...
                        [is(ind) js(ind) obj.nodes(js(ind)).cost ]);


                %str = [ str ...
                %        'struct' num2str(is(ind)) ' ->' ...
                %        'struct' num2str(js(ind)) ' ' ...
                %        '[ label=" - '  ...
                %        '(' num2str(0)  ...
                %        ')", color = black '  ...
                %        char(10) ];
            end

            str = [ str char(10) '}' ];
            myformat = '%s';
            fprintf(fid, myformat, '}');
            fclose(fid);
        end


        function [] = print_ubigraph(obj, filename)
        % generates output to be used by TFORUplot.py
            fmt = '%s';
            fid = fopen(filename, 'w');

            % nodes
            fprintf(fid, fmt, '[');
            for nid = 1:length(obj.nodes)
                if nid ~= 1
                    fprintf(fid, fmt, ',');
                end

                fprintf(fid, fmt, [ '"' obj.nodes(nid).model.name ' '...
                                    char(TFDimensionList2cell(obj ...
                                                              .nodes(nid).contraction_dims))' ...
                                  '"']);
            end
            fprintf(fid, fmt, [']' char(10)]);


            % edges
            fprintf(fid, fmt, '[');
            [is,js,~] = find(obj.edges);
            for ind = 1:length(is)
                if ind ~= 1
                    fprintf(fid, fmt, ',');
                end
                
                fprintf(fid, fmt, ...
                        [num2str(is(ind)) ',' num2str(js(ind)) ]);
            end
            fprintf(fid, fmt, [']' char(10)]);

            fclose(fid);
        end
    end
end