% Represents data required to describe a GCTF model

% Take care in naming factors. Shared factors must have the same
% TFFactor.name property in R cell. Factors are identified by their
% name properties

classdef GCTFModel
    properties
        name = '';
        dims = [TFDimension];
        observed_factors = [TFFactor];
        R = { [ TFFactor TFFactor TFFactor] [TFFactor TFFactor] };
        cost = 0;

        % stores unique factors in a Map structure
        % key is TFFactor.name property
        unique_factors = containers.Map()
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

            % generate unique_factors
            for Ri = 1:length(obj.R)
                for mi = 1:length(obj.R{Ri})
                    obj.unique_factors( obj.R{Ri}(mi).name ) = ...
                        obj.R{Ri}(mi);
                end
            end

        end

        function [] = gctf(obj, iternum, operation_type, ...
                           return_dot_data)
            dot_data = '';


            % build a PLTF for each observed_factor with relevant
            % latent factors
            pltf_models = [];
            for oi = 1:length(obj.observed_factors)
                dims = [];
                found = false;
                for di = 1:length(obj.dims)
                    % check observed factor
                    if obj.observed_factors(oi).got_dimension(obj.dims(di))
                        found = true;
                    end

                    % check latent factors
                    for lfi = 1:length(obj.R{oi})
                        if obj.R{oi}(lfi).got_dimension(obj.dims(di))
                            found = true;
                            break;
                        end
                    end

                    if found
                        dims = [ dims obj.dims(di) ];
                        found = false;
                    end

                end

                pltf_models = [ pltf_models PLTFModel('name', ['gctf_' num2str(oi)], ...
                                                      'factors', [ obj.observed_factors(oi) obj.R{oi} ], ...
                                                      'dims', dims) ];
            end


            % initalize obj.cost with memory requirements of the
            % model elements
            obj.cost = obj.get_element_size();
            display( [ 'obj.cost ' num2str(obj.cost) ] );

            % initialize hatX_v objects
            hat_X_v = obj.observed_factors;
            masks = cell(1, length(obj.observed_factors));
            for v = 1:length(obj.observed_factors)
                hat_X_v(v).name = ['hat_X_v' num2str(v)];
                obj.cost = obj.cost + ...
                    hat_X_v(v).get_element_size();
                display( [ 'obj.cost ' num2str(obj.cost) ] );

                if strcmp( operation_type, 'compute' )
                    % -yok- access factor data
                    %eval( [ 'global ' obj.get_factor_data_name(obj.observed_factors) ...
                    %        ';' ] );
                    eval( [ 'global hat_X_v' num2str(v) '_data'] );
                    eval( [ 'masks{v} = ones(size( hat_X_v' num2str(v) ...
                            '_data))' ] );
                    hat_X_v(v).rand_init(obj.dims, 100);
                end
            end





            for i = 1:iternum
                for p = 1:length(obj.observed_factors)
                    pltf_models(p).pltf_iteration('optimal', hat_X_v(p), ...
                                                  masks{p}, ...
                                                  operation_type)
                end
            end






%            if strcmp( operation_type, 'compute' )
%                %global mask_data;
%                %mask_data = ones(size(hat_X_data));
%
%                KL=zeros(1,iternum);
%                for iter = 1:iternum
%                    display(['iteration' char(9) num2str(iter)]);
%                    [ kl cost ] = obj.gctf_iteration( hat_X_v, ...
%                                                      operation_type);
%                    KL(iter) = kl;
%                end
%
%                display(['KL divergence over iterations: ']);
%                display(KL);
%                plot(KL);
%                title('KL divergence over iterations');
%                xlabel('iteration number');
%                ylabel('KL divergence');
%
%            elseif strcmp( operation_type, 'mem_analysis' )
%                [ kl cost dot_data ] = ...
%                    obj.gctf_iteration( hat_X_v, ...
%                                        operation_type, ...
%                                        return_dot_data );
%            end
%
%
%
%            obj.cost = obj.cost + cost;
%            display( ['e9 ' num2str(obj.cost) ' <- ' num2str(cost) ] );
%
%
%            display([char(10) ...
%                     'data elements required: ' num2str(obj.cost) ...
%                     char(10) ...
%                     ['memory size with (8 byte) double precision: ' ...
%                      num2str(8 * obj.cost / 1000 / 1000) ' MB' ] ] );
        end




        function [ kl cost dot_data ] = gctf_iteration( obj, ...
                                                        hat_X_v, ...
                                                        operation_type, ...
                                                        return_dot_data );

            if nargin < 4
                return_dot_data = 'no';
            end

            dot_data = '';
            cost = 0;


            % access global data
            %            for v = 1:length(obj.observed_factors)
            %    hat_X_v(v).name = ['hat_X_v' num2str(v)];
            %    eval( [ 'global hat_X_v' num2str(v) '_data' ];                


            % update each hatX_v
            for v = 1:length(obj.observed_factors)
                for alpha = 1:length(R)
                    
                end
            end

            % update each Z_alpha
            for v = 1:length(obj.observed_factors)
                for alpha = 1:legnth(R)
                    
                end
            end

        end


        function [graph] = schedule_dp(obj)
            
        end




        function [] = get_unique_factors(obj)
        % returns unique factors present in the model
            
        end

        function [size] = get_element_size(obj)
        % returns number of elements for this model
            size = 0;
            for ofi = 1:length(obj.observed_factors)
                size = size + ...
                       obj.observed_factors(ofi).get_element_size();
            end

            keys = obj.unique_factors.keys();
            for ufi = 1:length(keys)
                size = size + ...
                       obj.unique_factors(keys{ufi}) ...
                       .get_element_size();
            end
        end

        function [name] = get_factor_data_name(obj, factor)
        % returns global data name of the factor
        % 
        % To preserve model elements' data, their naming structure
        % is different than temporary factors. Model data elements
        % use TFFactor.get_data_name whereas temporary elements use
        % PLTFModel.get_coded_factor_name. This it is possible to
        % re-use memory structures with same dimensions
            if factor.isTemp
                name = obj.get_coded_factor_name(factor);
            else
                % assume we have model element
                name = factor.get_data_name();
            end
        end


    end
end