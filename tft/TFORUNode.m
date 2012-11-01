% Represents a TFORUGraph node.
%
% See also TFORUGraph

classdef TFORUNode

    properties
        % model this node represents
        model = PLTFModel; 

        % selected order of contraction
        contraction_dims = [TFDimension]

        % is model contracted before
        is_contracted = false;

        % populated with list of temporary factors required by this
        % contraction sequence on construction. stores parent
        % temporary factors as well while tree is built
        temp_factors = [];

        % calculated using element number of temp_factor elements
        % by removing parent temp_factors costs from children
        cost = 0;

        % index marking end of temp_factors belonging to this node
        % after construction this index points to parent node's
        % temp factors
        own_temp_index;
    end

    methods
        function obj = TFORUNode(model, contraction_dims, extra_temp_count)
            obj.model = model;
            obj.contraction_dims = contraction_dims;

            % generate temp_factors with extra factors
            d = TFDimension('name', '', 'cardinality', -1);
            t = TFFactor('name', 'extra', 'type', 'temp', 'dims', ...
                         [d]);
            obj.temp_factors = repmat( t, 1, ...
                                       length(contraction_dims)-1 + ...
                                       extra_temp_count);

            % finds temporary factors required by this model with this
            % contraction sequence
            if obj.is_contracted == false
                obj.own_temp_index = 1;
                for ci = 1:(length(obj.contraction_dims)-1)
                    obj.model = ...
                        obj.model.contract(obj.contraction_dims(ci), ...
                                           'mem_analysis', ...
                                           '');

                    new_temp_num = length(obj.model.temp_factors());
                    obj.temp_factors(obj.own_temp_index:(obj.own_temp_index+new_temp_num-1)) = obj.model.temp_factors();

                    % update model cost with these temp factors
                    for i = obj.own_temp_index:(obj.own_temp_index+new_temp_num-1)
                        obj.cost = obj.cost + ...
                            obj.temp_factors(i).get_element_size();
                    end
                    obj.own_temp_index = obj.own_temp_index + new_temp_num;
                end
                obj.is_contracted = true;

            else
                % must optimize if you arrive here
                throw(MException('TFORUNode:Why?', ...
                                 ['why calling find_temps second ' ...
                                  'time? ']));
            end
            %length(obj.temp_factors)
        end


        function [en] = get_extra_temp_factor_num(obj)
            en = 0;
            for i = 1:length(obj.temp_factors)
                if strcmp(obj.temp_factors(i), 'extra')
                    en = en + 1;
                end
            end
        end


        function [found] = temp_exists(obj, temp_factor)
        % checks if given temporary factor exists in object's
        % temp_factor list or not
            found = false;
            %for tfi = 1:length(obj.temp_factors)
            for tfi = 1:(obj.own_temp_index-1)
                if obj.temp_factors(tfi) == temp_factor
                    found = true;
                    return
                end
            end
        end




        function [] = update_cost(obj, parent_temps, parent_own_temp_index, ...
                                  final_level)
        % updates cost of node according to parent node's
        % temp_factors configuration

            pl = length(parent_temps);

            % update node cost
            for pti = 1:pl
                if obj.temp_exists(parent_temps(pti))
                    obj.cost = obj.cost - parent_temps(pti).size;
                end
            end

            % cost 0 is reserved for no link
            if obj.cost == 0
                obj.cost = 0.000001;
            end


            % store parent temp_nodes in this node for children
            % unless this is final level
            if ~final_level
                obj.temp_factors(obj.own_temp_index:(obj.own_temp_index-1)+(parent_own_temp_index-1)) = ...
                    parent_temps(1:(parent_own_temp_index-1));
            end
        end

    end

end