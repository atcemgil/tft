% Represents temporary factors used by a single model for a line of
% TFORUTable

classdef TFORUTableLineTemps
    properties
        % contains temporary factors for a single model for a line of
        % the table
        model_temps = containers.Map();
    end


    methods
        %function obj = add(obj, temp_factor)
        %    obj.model_temps(temp_factor.name) = temp_factor;
        %end

        %function obj = TFORUTableLineTemps()
        %    for i = 1:model_count
        %        line_temps(i) = containers.Map()
        %    end
        %end

        %function [] = add_value(obj, component_ind, temp_factor)
        %    line_temps(component_ind)
        %end
    end
end