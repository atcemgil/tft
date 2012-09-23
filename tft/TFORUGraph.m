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
    end

    methods
        function obj = TFORUGraph(model_list)
        % generate a TFORUGraph from a list of TFModel objects

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
                                  TFORUNode(model_list(mind), ...
                                            contraction_perms(cpind, ...
                                                              :)) ...
                                ];
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
                                          ) ]
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
                                obj.edges( (prev_level_indices(plii) ...
                                            + pln ) , ...
                                           new_level_indices( c ) + ...
                                           nln ) = 1;
                            end
                            c = c+1;
                        end
                    end
                end

                prev_level_node_num = new_level_node_num;
                prev_level_indices = new_level_indices;
            end
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
            myformat = ['struct%u -> struct%u ' char(10) ];
            for ind = 1:length(is)

                fprintf(fid, myformat, [is(ind) js(ind) ]);


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
        end
    end
end