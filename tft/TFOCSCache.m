% Represents cached search results for optimal contraction sequence search operation
%   get_optimal_contraction_sequence_dims function searches for
%   optimal contraction sequence using schedule_dp. Over iterations
%   of PLTF same search operation must be conducted several
%   times. This object stores generated values to be used again in
%   following operations
%
% See also PLTFModel

classdef TFOCSCache

    properties
        model = PLTFModel;
        ocs_dims = {};
    end

    methods

        function obj = TFOCSCache(model, ocs_dims)
            obj.model = model;
            obj.ocs_dims = ocs_dims;
        end

    end
end
