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

    end

    methods
        function obj = TFORUGraph(model_list)
        % generate a TFORUGraph from a list of TFModel objects

        %        function [] = oldconstructor(model_list)

            % generate tree
            parent_temp_count = 0;
            for mind = 1:length(model_list)
                % for each model
                mind

                contraction_perms = ...
                    perms(model_list(mind) ...
                          .get_contraction_dims());

                % nodes created with new model
                new_level = [];
                for cpind = 1:size(contraction_perms,1)
                    % for each contraction sequence generate a node
                    new_level = [ new_level ...
                                  TFORUNode( model_list(mind), ...
                                             contraction_perms(cpind,:), ...
                                             parent_temp_count)];
                    %length(new_level(1).temp_factors)
                end
                new_level_node_num = length(new_level);





                % connect previous level to new_level groups
                if mind == 1
                    % init TFORUGraph
                    obj.nodes = new_level;
                    obj.edges = sparse( 1, 1 );
                    new_level_indices = [0];
                else
                    % insert new nodes for all parents
                    new_level_indices = [ (length(obj.nodes)) : ...
                                        new_level_node_num : ...
                                        ( length(obj.nodes) + ...
                                        ( new_level_node_num * ...
                                          (prev_level_node_num) * ...
                                          (length(prev_level_indices))-1) ...
                                          ) ];
                    obj.nodes = [ obj.nodes ...
                                  repmat(new_level, 1, ...
                                         ( prev_level_node_num * ...
                                           length(prev_level_indices)) ) ...
                                ];

                    c = 1;
                    for plii = 1:length(prev_level_indices)
                        for pln = 1:(prev_level_node_num)
                            for nln = 1:new_level_node_num
                                % connect parent level to new nodes
                                pind = prev_level_indices(plii) + pln;
                                cind = new_level_indices( c ) + nln;
                                obj.edges( pind , cind ) = 1;
                                %obj.update_child_cost(pind, cind);

                                obj.nodes(cind).update_cost( ...
                                    obj.nodes(pind).temp_factors, ...
                                    obj.nodes(pind).own_temp_index, ...
                                    mind == length(model_list));
                            end
                            c = c+1;
                        end
                    end
                end

                prev_level_node_num = new_level_node_num;
                prev_level_indices = new_level_indices;
                max_temp_factor_count = 0;
                for i = 1:length(new_level)
                    if length(new_level(i).temp_factors) > ...
                            max_temp_factor_count
                        max_temp_factor_count = ...
                            length(new_level(i).temp_factors);
                    end
                end
                parent_temp_count = parent_temp_count + ...
                    max_temp_factor_count;
            end
        end


        function [] = update_child_cost(obj, pind, cind)
        % store parent temps in child
        % update cost for recurring temps

            %parent = pind;
            %while length(parent)
                
            for ptfi = 1:length(obj.nodes(pind).temp_factors)
                if obj.nodes(cind).temp_exists( ...
                    obj.nodes(pind).temp_factors(ptfi) )
                    % this temp was created
                    % before -> reduce cost
                    obj.nodes(cind).cost =  ...
                        obj.nodes(cind).cost - ...
                        obj.nodes(pind).temp_factors(ptfi).size;

                    % cost 0 is reserved for no link
                    if obj.nodes(cind).cost == 0
                        obj.nodes(cind).cost ...
                            = 0.000001;
                    end

                else
                    % copy this temp to child
                    % for use of children of
                    % child node
                    %obj.nodes(cind).temp_factors(obj.nodes(cind) ...
                    %                             .own_temp_index) ...
                    %    = obj.nodes(pind).temp_factors(ptfi);
                    %obj.nodes(cind).own_temp_index = ...
                    %    obj.nodes(cind).own_temp_index + 1;

                    obj.nodes(cind).update_cost(obj.nodes(pind).temp_factors);
                end
            end

            % % continue with parent's parent
            %    parent = find(obj.edges(:, parent) == 1);
            %end
        end


        function [cost] = get_node_cost(obj, child_index)
        % calculates memory cost for an attached child node
            % find all predecessors' temporary usage
            parent_list = [];
            ci = child_index;
            parent = find(obj.edges(:, ci) == 1);
            while length(parent)
                if length(parent) ~= 1
                    throw(MException('TFORUGraph:MultipleParents', ...
                                     ['>1 indegree! Tree construction ' ...
                                      'error ']));
                end

                % contract parent
                node_list(parent).find_temps();

                % store temporary factor dimensions
                parent = find(obj.edges(:, ci) == 1);
            end

            % contract child
            % cost = size of all temp factors not in parent temp
            % factors

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