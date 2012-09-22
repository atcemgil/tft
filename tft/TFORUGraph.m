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
        nodes = []; % list of TFORUNode objects

        edges; % edges matrix
    end

    methods
        function obj = TFORUGraph(model_list)
        % generate a TFORUGraph from a list of TFModel objects
            length(model_list)
            for mind = 1:length(model_list)

                % for each graph object
                contraction_perms = ...
                    perms(model_list(mind) ...
                          .get_contraction_dims());

                for cpind = 1:size(contraction_perms,1)
                    % for each contraction sequence generate a node
                    obj.nodes = [ obj.nodes ...
                                  TFORUNode(model_list(mind), ...
                                            contraction_perms(cpind, ...
                                                              :)) ...
                                ];
                end

                new_model_node_num = size(contraction_perms, 1);

                new_model_start = length(obj.nodes) - ...
                                   new_model_node_num + 1;

                if mind == 1
                    % init edges
                    obj.edges = zeros(length(obj.nodes),1);
                else
                    % expand edges

                    % if i dimension needs expanding
                    if size(obj.edges, 1) < new_model_node_num
                        % expand i dimension
                        extra_i_num = new_model_node_num - ...
                                      size(obj.edges, 1);
                        obj.edges( end + extra_i_num , : ) = 0;
                    end

                    % expand y dimension
                    obj.edges( :, end+1 ) = 0;


                    % fully connect nodes of this model with nodes
                    % previous of previous model
                    obj.edges( prev_model_start:new_model_start, ...
                               new_model_start:end ) = 1;
                end

                prev_model_start = new_model_start;
            end
        end
    end
end