% Represents a TFORUGraph node.
%
% See also TFORUGraph

classdef TFORUNode

    properties
        % model this node represents
        model = TFModel; 

        % selected order of contraction
        contraction_dims = [TFDimension]
    end

    methods
        function obj = TFORUNode(model, contraction_dims)
            obj.model = model;
            obj.contraction_dims = contraction_dims;
        end
    end
end