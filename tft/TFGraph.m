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

        optimal_edges; % half full minimal cost node_list edges
    end

    methods
        function obj = TFGraph()
            obj.node_list = [];
        end

%        function [r] = eq(a,b)
%            r = false;
%            if length(a.node_list) == length(b.node_list)
%                for i_a = 1:length(a.node_list)
%                    found = false;
%                    for i_b = 1:length(b.node_list)
%                        if a.node_list(a_i) == b.node_list(b_i)
%                            found = true;
%                            break
%                        end
%                    end
%
%                    if ~found
%                        return
%                    end
%                end
%
%                r = true;
%            end
%        end


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
                    if first
                        first=0;
                        str = [ str '<FONT ' ];
                        if obj.node_list(node_list_index)...
                                .factors(find).isReUsed
                            str = [ str 'COLOR="red"' ];
                        end
                        str = [ str '>' ];
                    else
                        str = [ str ', ' ];
                    end

                    for dind = length(...
                        obj.node_list(node_list_index).factors(find) ...
                        .dims):-1:1
                        str = [ str ...
                                char(obj.node_list(node_list_index) ...
                                     .factors(find).dims(dind).name) ];
                    end
                end
            end
            if ~first
                str = [ str '</FONT>' ];
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


        function [ocs_dims] = optimal_sequence_from_graph(graph)
        % return optimal contraction sequence from a TFGraph
        % developed as a part of
        % TFModel.get_optimal_contraction_sequence_dims and then turned
        % into a helper function for use from other points in the
        % code
            t = graph.optimal_edges;
            t(t==0) = Inf;
            ocs_models = [];
            i = length(t);
            while i ~= 1
                ocs_models = [ ocs_models graph.node_list(i) ];
                i = find( t(:,i) == min(t(:, i)) );
            end
            ocs_models = [ ocs_models graph.node_list(i) ];

            ocs_dims = [];
            %for i = (length(ocs_models)):-1:2
            for i = 1:(length(ocs_models)-1)
                ocs_dims = [ ocs_dims ...
                             { setdiff( ...
                                 ocs_models(i)...
                                 .get_current_contraction_dims, ...
                                 ocs_models(i+1) ...
                                 .get_current_contraction_dims) }; ...
                           ];
            end
        end


        function [str, nid_end] = print_dot(obj, nid_start, subgraph_label)
        % nid_start: start from number nid, used when multiple TFGraph objects
        % are plotted in a single graph
        % subgraph_label: can be used to label subgraphs

            if nargin == 1
                nid_start = 0;

                % put header only if drawing single graph
                str = [ 'digraph structs{' char(10) ...
                        'rankdir=LR;' char(10) ...
                        'node [shape=plaintext];' char(10) ...
                        'splines=false; ' char(10)];
            else
                str = [ 'subgraph cluster_' num2str(nid_start) ' {' ...
                      ];
                if nargin == 3 
                    str = [str char(10) 'label = "' ...
                           char(subgraph_label) '"' ];
                end
            end


            for nid = 1:length(obj.node_list)
                top = obj.get_current_contraction_dims_string(nid);
                if ~length(top)
                    top = '&empty;';
                end

                str = [ str 'struct' num2str(nid+nid_start) ' [label=< <TABLE FIXEDSIZE="FALSE" CELLBORDER="0" STYLE="ROUNDED"><TR><TD>' ...
                        top ...
                        char(10) '</TD></TR> <HR/> <TR><TD FIXEDSIZE="FALSE">' ...
                        obj.get_factor_dim_string(nid) '</TD></TR> <TR><TD FIXEDSIZE="FALSE"></TD></TR></TABLE> >];' ...
                        char(10) 
                      ];
            end

            ocs_dims = obj.optimal_sequence_from_graph();
            % reverse ocs_dims for display
            ocs_dims = fliplr(ocs_dims);
            k = 1;
            next_optimal=0;
            for i = 1:length(obj.edges)
                for j = 1:i
                    if obj.edges(j,i)

                        lbl = setdiff(obj ...
                                      .get_current_contraction_dims_string(i), ...
                                      obj ...
                                      .get_current_contraction_dims_string(j));

                        if k <= length(ocs_dims) && ...
                                strcmp(lbl, ocs_dims{k}) && ...
                                (next_optimal == 0 || ...
                                 next_optimal == j)
                            lbl_color = 'blue';
                            k = k+1;
                            next_optimal = i;
                        else
                            lbl_color = 'black';
                        end

                        str = [ str ...
                                'struct' num2str(j + nid_start) ' ->' ...
                                'struct' num2str(i + nid_start) ' ' ...
                                '[ label="' lbl ...
                                '(' num2str(obj.optimal_edges(j,i))  ...
                                ')" color = ' lbl_color ' ];' char(10) ];

                    end
                end
            end


            str = [ str char(10) '}' ];
            nid_end = nid_start + nid;
        end


    end
end