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
        % array of TFORUNode objects in first level
        nodes = [];

        % edges between nodes
        edges = [];

        % store edge costs
        edge_costs = [];


        contraction_perms;
    end

    methods

        function obj = TFORUGraph(model_list)
        % assume first model is Xhat update
        % rest are latent factor updates

            obj.contraction_perms = repmat(TFORUContractionSequence, 1, length(model_list));
            for mind = length(model_list):-1:1
                obj.contraction_perms(mind) = TFORUContractionSequence( model_list(mind) );
            end

            latent_num = length(obj.contraction_perms)-1;
            max_level = latent_num*2;
            max_xhat_path_num = size(obj.contraction_perms(1).contraction_sequence_perms,1);


            global latent_cumulative_card latent_cumulative_cards
            % make an array of number of paths for latent factors
            latent_cumulative_cards = zeros(1, latent_num+1);
            % for now every latent update node will expand with latent_cumulative_card
            % must update with correct setting for each level
            latent_cumulative_card = 0;
            for li = 1:latent_num
                if li == 1
                    latent_cumulative_cards(li+1) = size(obj.contraction_perms(li+1).contraction_sequence_perms,1);
                else
                    latent_cumulative_cards(li+1) = latent_cumulative_cards(li) + size(obj.contraction_perms(li+1).contraction_sequence_perms,1);
                end

                latent_cumulative_card = latent_cumulative_card + size(obj.contraction_perms(li+1).contraction_sequence_perms,1);
            end


            % convinience cost array
            global latent_path_costs;
            latent_path_costs = ones(1, latent_cumulative_card) * Inf;
            i=1;
            for li = 1:latent_num
                for ci = 1:size(obj.contraction_perms(li+1).contraction_sequence_perms,1)
                    latent_path_costs(i) = obj.contraction_perms(li+1).costs{ci};
                    i = i+1;
                end
            end


            global S costs;
            S = {};
            costs = {};
            % insert first level elements
            for i = 1:size(obj.contraction_perms(1).contraction_sequence_perms,1)
                S{i} = [i zeros(1, max_level-1) ];
                costs{i} = obj.contraction_perms(1).costs{i};
            end
            bestsofar = Inf;


            node_num = 0;
            while length( S )
                % remove a subproblem from tail of S
                p = S{end};
                S = S(1:end-1);
                p_cost = costs{end};
                costs = costs(1:end-1);

                %ind = 1;
                %p_cost = costs{ind};
                %while p_cost == -1
                %    ind = ind+1;
                %    p_cost = costs{ind};
                %end
                %costs{ind} = -1;
                %p = S{ind};
                %S{ind} = [];
                
                %S = S(2:end);
                %costs = costs(2:end);



                % expand p into smaller subproblems
                % generate children of p
                zero_childs = find( p == 0 );
                first_zero_child_index = zero_childs(1);

                % add subproblems by incrementing in first zero child GTM

                
                for i = 1:latent_cumulative_card
                    node_num = node_num + 1;                    
                    if mod(node_num, 10000) == 0
                        display([num2str(node_num) ' ' num2str(length(S)) ' ' num2str(S{end}) ])
                    end



                    % break if in Xhat path, i > max_xhat_path_num
                    if mod(first_zero_child_index, 2) == 1 && i > max_xhat_path_num
                        break
                    end

                    % continue: if in latent update path and if one of this latent factor's paths were selected in parent nodes
                    if mod(first_zero_child_index, 2) == 0
                        % identify factor of this index
                        for lfi = 1:latent_num
                            if i > latent_cumulative_cards(lfi) && ...
                               i < latent_cumulative_cards(lfi+1)
                                i_factor_ind = lfi;
                                break;
                            end
                        end

                        prev_even_inds = (first_zero_child_index-2):-2:2;
                        found = false;
                        for peii = 1:length(prev_even_inds)
                            % check if path in prev_even_ind belongs to the same latent factor as i

                            % identify factor used in this previous path
                            for lfi = 1:latent_num
                                if p(prev_even_inds(peii)) > latent_cumulative_cards(lfi) && ...
                                   p(prev_even_inds(peii)) <= latent_cumulative_cards(lfi+1)
                                    prev_path_factor_ind = lfi;
                                    break;
                                end
                            end

                            if prev_path_factor_ind == i_factor_ind
                                found = true;
                                break;
                            end
                        end

                        if found
                            % a path belonging to this path's factor was used
                            continue
                        end
                    end





                    p_new = p;
                    p_new(first_zero_child_index) = i;


                    % cost calculation: looking for max mem along all paths
                    if mod(first_zero_child_index, 2) == 0
                        % latent update path
                        p_new_cost = latent_path_costs(i);
                    else
                        % Xhat update path
                        p_new_cost = obj.contraction_perms(1).costs{i};
                    end

                    if p_cost > p_new_cost
                        p_new_cost = p_cost;
                    end



                    % if p_new is a complete solution update bestsofar
                    if sum( p_new == 0 ) == 0
                        bestsofar = p_new_cost;
                    elseif p_new_cost < bestsofar
                        % else if p_new is still viable add to S for further inspection
                        S{end+1} = p_new;
                        costs{end+1} = p_new_cost;

                        %if costs{1} == -1
                        %    S{1} = p_new;
                        %    costs{1} = p_new_cost;
                        %else
                        %    S{end+1} = p_new;
                        %    costs{end+1} = p_new_cost;
                        %end
                    end
                end
            end

            bestsofar
        end

        














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