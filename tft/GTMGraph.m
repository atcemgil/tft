% Represents a graph of PLTFModel objects. 
%
%   PLTFModel.schedule_dp() function generates trees of PLTFModel
%   objects. GTMGraph object stores generated PLTFModel objects in its
%   node_list array and stores connectivity information in its
%   edges half full binary matrix. Cost of moving along an edge is
%   stored in the optimal_edges matrix.
%
%   See also PLTFModel

classdef GTMGraph

    properties
        node_list = [PLTFModel]; % list PLTFModel objects
        edges;     % half full binary matrix of node_list relations

        optimal_edges; % half full minimal cost node_list edges
    end

    methods
        function obj = GTMGraph()
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


        function [edge_cost] = check_reuse(obj, parent_node, ...
                                           child_node)

            edge_cost = obj.get_min_arriving_cost(obj.exists(parent_node)) + ...
                child_node.cost;

            % if new node was on optimal path in any one of
            % previous GTM operations it has zero memory
            % cost
            global reused_temp_factor_names

            % for each temporary factor of child_node
            for cnfi = 1:length(child_node.factors)
                if child_node.factors(cnfi).isTemp
                    for ri = 1:length(reused_temp_factor_names)
                        coded_name = child_node.get_coded_factor_name(cnfi);

                        %['coded_name ' coded_name' ' reused ri ' num2str(ri) ...
                        % ' : ' char(reused_temp_factor_names{ri}') ]
                        if strcmp(coded_name, ...
                                  char(reused_temp_factor_names{ri}))
                            edge_cost = edge_cost - child_node.factors(cnfi).get_element_size();
                        end
                    end
                end
            end

            % 0 is reserved for 'no edge'
            if edge_cost == 0
                edge_cost = 0.000001;
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


            %obj.optimal_edges(parent_index, end) = ...
            %    obj.check_reuse(parent_node, new_node);
            obj.optimal_edges(parent_index, end) = ...
                obj.get_min_arriving_cost(parent_index) + ...
                new_node.cost;


            %['append node ' obj.node_list(parent_index).name ' -> ' obj.node_list(length(obj.optimal_edges)).name ' ' ...
            % num2str(obj.get_min_arriving_cost(parent_index)) ' + ' ...
            % num2str(new_node.cost) ' = ' ...
            % num2str(obj.optimal_edges(parent_index, end))]

        end


        function [obj] = update_node(obj, parent_node, child_node, ...
                                     nnidx)
        % updates relation of parent_node and child_node

            pidx = obj.exists(parent_node);

            % create link between parent and child
            obj.edges(pidx, nnidx) = 1;

            %obj.optimal_edges(pidx, nnidx) = ...
            %    obj.check_reuse( obj.node_list(pidx), child_node);

            obj.optimal_edges(pidx, nnidx) = ...
                obj.get_min_arriving_cost(pidx) + child_node.cost;

            %obj.get_min_arriving_cost(pidx) + ...
            %child_node.cost;

            %['update node ' obj.node_list(pidx).name ' -> ' obj.node_list(nnidx).name ' ' ...
            % num2str(obj.get_min_arriving_cost(pidx)) ' + ' ...
            % num2str(child_node.cost) ' = ' num2str(obj.optimal_edges(pidx, nnidx))]
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


        function [] = store_reused_temp_factor(obj, ocs_models)
            % store temporary variables on optimal path
            global reused_temp_factor_names
            for i = 1:length(ocs_models)
                for fi = 1:length(ocs_models(i).factors)
                    if ocs_models(i).factors(fi).isTemp
                        coded_name = ocs_models(i).get_coded_factor_name(fi);

                        % store if not already inserted
                        found = false;
                        for j = 1:length(reused_temp_factor_names)
                            if strcmp(char(reused_temp_factor_names{j}), ...
                                      coded_name)
                                found = true;
                                break
                            end
                        end

                        if ~found
                            reused_temp_factor_names = [ ...
                                reused_temp_factor_names 
                                {coded_name} ];
                        end
                    end
                end
            end
        end


        function [ocs_dims] = optimal_sequence_from_graph(obj)
        % return optimal contraction sequence from a GTMGraph
        % developed as a part of
        % PLTFModel.get_optimal_contraction_sequence_dims and then turned
        % into a helper function for use from other points in the
        % code
        % populates reused_temp_factor_names with the temporary
        % factors on the optimal path since optimal path is
        % detected at this point

            t = obj.optimal_edges;
            t(t==0) = Inf;
            ocs_models = [];
            i = length(t);
            while i ~= 1
                ocs_models = [ ocs_models obj.node_list(i) ];
                m = min(t(:, i)); % if same value appears twice
                                  % pick first
                i = find( t(:,i) == m(1) );
                i = i(1); % pick first
            end

            ocs_models = [ ocs_models obj.node_list(i) ];

            ocs_dims = [];
            for i = (length(ocs_models)):-1:2
            %for i = 1:(length(ocs_models)-1)
                ocs_dims = [ ocs_dims ...
                             { setdiff( ...
                                 ocs_models(i)...
                                 .get_current_contraction_dims, ...
                                 ocs_models(i-1) ...
                                 .get_current_contraction_dims) }; ...
                           ];
            end

            obj.store_reused_temp_factor(ocs_models);
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
                    else
                        str = [ str ', ' ];
                    end

                    if node_list_index == 1 || node_list_index == length(obj.node_list)
                        str = [ str '<FONT COLOR="green"' ];
                    else

                        if obj.node_list(node_list_index)...
                                .factors(find).isReUsed
                            str = [ str '<U>' ];
                        end

                        str = [ str '<FONT ' ];
                        if obj.node_list(node_list_index)...
                                .factors(find).isTemp
                            str = [ str 'COLOR="red"' ];
                        end
                    end
                    str = [ str '>' ];


                    nstr = {};
                    for dind = length(...
                        obj.node_list(node_list_index).factors(find) ...
                        .dims):-1:1
                        nstr = [ nstr ...
                                 {char(obj.node_list(node_list_index) ...
                                      .factors(find).dims(dind).name)} ];
                    end

                    nstr = char(obj.node_list(1).order_dims(nstr))';

                    str = [ str nstr '</FONT>' ];
                    if obj.node_list(node_list_index)...
                            .factors(find).isReUsed && ...
                            node_list_index ~= 1 ...
                        str = [ str '</U>' ];
                    end

                end
            end
        end

        function [optimal_cost] = get_optimal_path_cost(obj)
            optimal_cost = 0;
            ocs_dims = obj.optimal_sequence_from_graph();
            % reverse ocs_dims for display
            %ocs_dims = fliplr(ocs_dims);
            k = 1;
            next_optimal=0;
            for i = 1:length(obj.edges)
                for j = 1:i
                    if obj.edges(j,i)

                        lbl = setdiff(obj ...
                                      .get_current_contraction_dims_string(j), ...
                                      obj ...
                                      .get_current_contraction_dims_string(i));

                        if k <= length(ocs_dims) && ...
                                strcmp(lbl, char(ocs_dims{k})) && ...
                                (next_optimal == 0 || ...
                                 next_optimal == j)
                            k = k+1;
                            next_optimal = i;
                            c = obj.optimal_edges(j,i);
                            if c > 0.1
                                optimal_cost =  c;
                            end
                        end
                    end
                end
            end
        end


        function [str, nid_end] = print_dot(obj, nid_start, subgraph_label)
        % nid_start: start from number nid, used when multiple GTMGraph objects
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
            %ocs_dims = fliplr(ocs_dims);
            k = 1;
            next_optimal=0;
            for i = 1:length(obj.edges)
                for j = 1:i
                    if obj.edges(j,i)

                        lbl = setdiff(obj ...
                                      .get_current_contraction_dims_string(j), ...
                                      obj ...
                                      .get_current_contraction_dims_string(i));

                        if k <= length(ocs_dims) && ...
                                strcmp(lbl, char(ocs_dims{k})) && ...
                                (next_optimal == 0 || ...
                                 next_optimal == j)
                            lbl_color = 'blue';
                            style = 'dashed';
                            k = k+1;
                            next_optimal = i;
                        else
                            lbl_color = 'black';
                            style = '';
                        end

                        % display 10^-6 as 0
                        cost = obj.optimal_edges(j,i);
                        if cost < 0.1
                            cost = 0;
                        end

                        str = [ str ...
                                'struct' num2str(j + nid_start) ' ->' ...
                                'struct' num2str(i + nid_start) ' ' ...
                                '[ label="' lbl ...
                                '(' num2str(cost)  ...
                                ')", color = ' lbl_color ...
                                ', style = "' style '" ];' char(10) ];

                    end
                end
            end


            str = [ str char(10) '}' ];
            nid_end = nid_start + nid;
        end


    end
end