% Represents a graph of TFModel objects. 
%
%   TFModel.schedule_dp() function generates trees of TFModel
%   objects. TFGraph object stores generated TFModel objects in its
%   node_list array and stores connectivity information in its
%   edges half full binary matrix. Cost of moving along an edge is
%   stored in the optimal_edges matrix.
%
%   See also TFModel

classdef TFGraph

    properties
        node_list = TFModel; % list TFModel objects
        edges;     % half full binary matrix of node_list relations

        optimal_edges % half full minimal cost node_list edges
    end

    methods
        function obj = TFGraph()
            obj.node_list = [];
        end

        function [r] = exists(obj, tfmodel)
        % returns index of given node
        % returns 0 if node does not exist
            r = 0;
            for nli = 1:length(obj.node_list)
                if obj.node_list(nli) == tfmodel
                    r = nli;
                    return
                end
            end
        end


        function [obj] = clear_edges(obj)
            obj.edges = zeros(length(obj.node_list));
            obj.optimal_edges = zeros(length(obj.node_list));
        end

        function [obj] = increment_edges(obj)
        % extend relationship matrices
            obj.edges = [ obj.edges ; ...
                          zeros(1,size(obj.edges, 2))];
            obj.edges = [ obj.edges ...
                          zeros(length(obj.node_list),1)];

            obj.optimal_edges = [ obj.optimal_edges ; ...
                                zeros(1, ...
                                      size(obj.optimal_edges,2 ))];
            obj.optimal_edges = [ obj.optimal_edges ...
                                zeros(length(obj.node_list),1)];
        end


        function [m] = get_min_arriving_cost(obj, node_index)
            A = obj.optimal_edges(:, node_index);
            A(~A) = inf;
            m = min(A);
            if isinf(m)
                m = 0;
            end
        end


        function [obj] = append_node(obj, parent_node, new_node)
        % adds a new node to the graph object
            obj.node_list = [obj.node_list new_node];

            obj = obj.increment_edges();

            % update relation matrices
            parent_index = obj.exists(parent_node);
            if ~parent_index
                display('ERROR: parent is not in the node_list')
            end
            obj.edges(parent_index, end) = 1;

            obj.optimal_edges(parent_index, end) = ...
                obj.get_min_arriving_cost(parent_index) + ...
                new_node.cost;
        end


        function [obj] = update_node(obj, parent_node, child_node, ...
                                     nnidx)
        % updates relation of parent_node and child_node

            pidx = obj.exists(parent_node);

            % create link between parent and child
            obj.edges(pidx, nnidx) = 1;

            obj.optimal_edges(pidx, nnidx) = ...
                obj.get_min_arriving_cost(pidx) + ...
                child_node.cost;
        end


        function [str] = get_factor_dim_string(obj, ...
                                               node_list_index)
            str = '';
            first=1;
            for find = 1: ...
                       length(obj.node_list(node_list_index) ...
                              .factors)
                if obj.node_list(node_list_index).factors(find).isLatent
                    if ~first
                        str = [ str ', ' ];
                    end
                    first=0;

                    for dind = length(...
                        obj.node_list(node_list_index).factors(find) ...
                        .dims):-1:1
                        str = [ str ...
                                char(obj.node_list(node_list_index) ...
                                     .factors(find).dims(dind).name) ];
                    end
                end
            end
        end


        function [str] = get_current_contraction_dims_string(obj, ...
                                                             node_list_index)
            str = '';
            cont_dims = ...
                obj.node_list(node_list_index) ...
                .get_current_contraction_dims;

            for cdi = 1:length(cont_dims)
                str = [ str ...
                        char(cont_dims(cdi)) ];
            end
        end


        function [str] = print_dot(obj)
            str= [ 'digraph structs{' char(10) ...
                   'rankdir=LR;' char(10) ...
                   'node [shape=Mrecord];' char(10) ...
                   'splines=false ' char(10)];

            for nid = 1:length(obj.node_list)
                str = [ str 'struct' num2str(nid) ' [label="<f0> ' ...
                        obj.get_current_contraction_dims_string(nid) ...
                        ' | <f1> ' obj.get_factor_dim_string(nid) '"];' ...
                        char(10) 
                      ];
            end

            for i = 1:length(obj.edges)
                for j = 1:i
                    if obj.edges(j,i)
                        str = [ str ...
                                'struct' num2str(j) ':f0 ->' ...
                                'struct' num2str(i) ':f0 ' ...
                                '[ label="' ...
                                setdiff(obj ...
                                        .get_current_contraction_dims_string(i), ...
                                        obj.get_current_contraction_dims_string(j)) ...
                                '(' num2str(obj.optimal_edges(j,i))  ')" ];' char(10) ];
                    end
                end
            end

            str = [ str char(10) '}' ];

        end


    end
end