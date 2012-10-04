% Represents data required to describe a GCTF model

classdef GCTFModel
    properties
        name = '';
        dims = [TFDimension];
        observed_factors = [TFFactor];
        R = { [ TFFactor] [TFFactor TFFactor] };
    end

    methods

        function obj = GCTFModel(varargin)
            p = inputParser;
            addParamValue(p, 'name', '', @isstr);
            addParamValue(p, 'dims', [], @isvector);
            addParamValue(p, 'observed_factors', [], @isvector);
            addParamValue(p, 'R', [], @isvector);

            parse(p, varargin{:});

            obj.name = p.Results.name;
            obj.dims = p.Results.dims;
            obj.observed_factors = p.Results.observed_factors;
            obj.R = p.Results.R;
        end

        function [] = gctf(obj, iternum, contraction_type, operation_type, ...
                           return_dot_data)
            dot_data = '';

            for i = 1:iternum
                for v = 1:length(obj.observed_factors)


                    % update each hatX_v
                    for alpha = 1:length(R)
                        
                    end


                    % update each Z_alpha
                end
            end
            
        end

    end
end