% Represents data required to describe a GCTF model

% Take care in naming factors. Shared factors must have the same
% TFFactor.name property in R cell. Factors are identified by their
% name properties

classdef GCTFModel
    properties
        name = '';

        dims = [TFDimension];

        observed_factors = [TFFactor];

        % latent factors
        R = { [ TFFactor TFFactor TFFactor] [TFFactor TFFactor] };

        cost = 0;

        % stores unique factors in a Map structure
        % key is TFFactor.name property
        unique_latent_factors = containers.Map()
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

            % generate unique_latent_factors
            for Ri = 1:length(obj.R)
                for mi = 1:length(obj.R{Ri})
                    obj.unique_latent_factors( obj.R{Ri}(mi).name ) = ...
                        obj.R{Ri}(mi);
                end
            end

        end

        function [KL] = gctf(obj, iternum, operation_type, ...
                           return_dot_data)
            if nargin == 2
                operation_type = 'compute';
                return_dot_data = 'no';
            end

            dot_data = '';

            % initalize obj.cost with memory requirements of the
            % model elements
            obj.cost = obj.get_element_size();
            %display( [ 'obj.cost ' num2str(obj.cost) ] );

            % initialize hatX_v objects
            hat_X_v = obj.observed_factors;
            masks = hat_X_v;
            for v = 1:length(obj.observed_factors)
                hat_X_v(v).name = ['hat_X_v' num2str(v)];
                masks(v).name = ['mask' num2str(v)];

                obj.cost = obj.cost + ...
                    hat_X_v(v).get_element_size() *2 ; % *2 for mask
                %display( [ 'obj.cost ' num2str(obj.cost) ] );

                if strcmp( operation_type, 'compute' )
                    hat_X_v(v).rand_init(obj.dims, 100);
                    eval( [ 'global ' hat_X_v(v).get_data_name() ]);
                    eval( [ 'global ' masks(v).get_data_name() ]);
                    eval( [ masks(v).get_data_name() ' = ones(size(' ...
                            hat_X_v(v).get_data_name() '));' ]);
                end
            end


            if strcmp( operation_type, 'compute' )

                KL=zeros(iternum, length(obj.observed_factors));
                for iter = 1:iternum
                    display(['iteration' char(9) num2str(iter)]);
                    [ kls cost ] = obj.gctf_iteration( hat_X_v, ...
                                                      masks, ...
                                                      operation_type, ...
                                                      'no');

                    %, ...
                    %                                 pltf_models);
                    KL(iter,:) = kls;
                end

                display(['KL divergence over iterations: ']);
                display(KL);
                plot(KL);
                title('KL divergence over iterations');
                xlabel('iteration number');
                ylabel('KL divergence');

            elseif strcmp( operation_type, 'mem_analysis' )
                [ kl cost dot_data ] = ...
                    obj.gctf_iteration(hat_X_v, ...
                                       masks, ...
                                       operation_type, ...
                                       'no' );
            end
        end




        function [ kl cost dot_data ] = gctf_iteration( obj, ...
                                                        hat_X_v, ...
                                                        masks, ...
                                                        operation_type, ...
                                                        return_dot_data)

            if nargin < 4
                return_dot_data = 'no';
            end

            dot_data = '';
            cost = 0;


            % access global data
            for v = 1:length(obj.observed_factors)
                eval( [ 'global hat_X_v' num2str(v) '_data' ] );
                eval( [ 'global ' obj.observed_factors(v).get_data_name() ...
                        ] );
            end





            % update each Z_alpha
            ulfk = obj.unique_latent_factors.keys();
            for alpha = 1:length(ulfk)





                % build a PLTF for each observed_factor with relevant
                % latent factors
                pltf_models = [];
                for oi = 1:length(obj.observed_factors)

                    pltf_models = [ pltf_models ...
                                    PLTFModel('name', ['gctf_' ...
                                        num2str(oi)], ...
                                              'factors', [ ...
                                                  obj.observed_factors(oi) obj.R{oi} ], ...
                                              'dims', obj.dims) ...
                                  ];
                end





                % update each hatX_v
                for v = 1:length(obj.observed_factors)
                    %display([ 'update hatX_v' num2str(v) ])
                    hat_X_data_name = hat_X_v(v).get_data_name();
                    newmodel = pltf_models(v);
                    % perform contraction
                    % store result in hat_X_data
                    [ ~ ] = ...
                        newmodel.contract_all('standard', ...
                                              operation_type, ...
                                              hat_X_data_name);

                    % store X / hat_X in hat_X data
                    if strcmp( operation_type, 'compute' )
                        eval( [ hat_X_data_name '  =  ' ...
                                obj.observed_factors(v).get_data_name() ...
                                ' ./ ' ...
                                hat_X_data_name ' ;' ] );
                    end

                end











                d1 = TFFactor('name', ...
                              ['D1_Z' num2str(alpha)], ...
                              'type', 'observed', ...
                              'dims', obj.unique_latent_factors(char(ulfk(alpha))).dims);
                d1.zero_init(obj.dims);

                d2 = TFFactor('name', ...
                              ['D2_Z' num2str(alpha)], ...
                              'type', 'observed', ...
                              'dims', obj.unique_latent_factors(char(ulfk(alpha))).dims);
                d2.zero_init(obj.dims);

                Z_name = obj.unique_latent_factors(char(ulfk(alpha))).get_data_name();
                d1_name = d1.get_data_name();
                d2_name = d2.get_data_name();
                eval([ 'global ' Z_name ' ' d1_name ' ' d2_name]);


                for v = 1:length(obj.observed_factors)
                    if ~obj.got_factor(v, obj.unique_latent_factors(char(ulfk(alpha))))
                        %display(['observed factor ' obj.observed_factors(v).name ...
                        %         ' does not use latent ' ...
                        %         'factor ' char(ulfk(alpha)) ]);
                        continue
                    end

                    %display(['observed factor ' obj.observed_factors(v).name ...
                    %         ' uses latent ' ...
                    %         'factor ' char(ulfk(alpha)) ]);


                    d1_x = TFFactor('name', ...
                                    ['D1_Z' num2str(alpha) '_X' num2str(v)], ...
                                    'type', 'observed', ...
                                    'dims', obj.unique_latent_factors(char(ulfk(alpha))).dims);
                    d2_x = TFFactor('name', ...
                                    ['D2_Z' num2str(alpha) '_X' num2str(v)], ...
                                    'type', 'observed', ...
                                    'dims', obj.unique_latent_factors(char(ulfk(alpha))).dims);
                    d1_x_name = d1_x.get_data_name();
                    d2_x_name = d2_x.get_data_name();
                    eval([ 'global ' d1_x_name ' ' ...
                           d2_x_name]);

                    other_factors = [];
                    %display(['other factors for factor ' char(ulfk(alpha)) ...
                    %         ' in model ' obj.observed_factors(v).name ...
                    %        ]);
                    for ofi = 1:length(obj.R{v})
                        if obj.R{v}(ofi) ~= obj.unique_latent_factors(char(ulfk(alpha)))
                            other_factors = [ other_factors ...
                                              obj.R{v}(ofi) ];
                            %display(obj.R{v}(ofi).name);
                        end
                    end

                    tmpmodel = PLTFModel('name',  ...
                                         ['tmpmodel_Z' num2str(alpha) '_X' num2str(v)], ...
                                         'factors' ,  ... 
                                         [hat_X_v(v) other_factors d1_x], ...
                                         'dims', obj.dims );
                    tmpmodel.factors(1).isLatent = 1;
                    tmpmodel.factors(1).isObserved = 0;
                    tmpmodel.contract_all('standard', operation_type, ...
                                          d1_x_name);
                    eval([ d1_name ' = ' d1_name ' + ' d1_x_name ';']);


                    tmpmodel = PLTFModel('name',  ...
                                         ['tmpmodel_Z' num2str(alpha) '_X' num2str(v)], ...
                                         'factors' ,  ... 
                                         [masks(v) other_factors d2_x], ...
                                         'dims', obj.dims );
                    tmpmodel.factors(1).isLatent = 1;
                    tmpmodel.factors(1).isObserved = 0;
                    tmpmodel.contract_all('standard', operation_type, ...
                                          d2_x_name);
                    eval([ d2_name ' = ' d2_name ' + ' d2_x_name ';']);
                end

                % update Z_alpha with d1/d2
                eval([ Z_name ' = ' Z_name ' .* ' d1_name ' ./ ' ...
                       d2_name ';' ]);
            end





            cost=0;
            
            if strcmp( operation_type, 'compute' )
                % calculate KL divergence
                kl = zeros(1, length(obj.observed_factors));
                for v = 1:length(obj.observed_factors)
                    hat_X_data_name = hat_X_v(v).get_data_name();
                    % restore hat_X_data
                    eval( [ hat_X_data_name '  =  ' ...
                            obj.observed_factors(v).get_data_name() ...
                            ' .* ' ...
                            hat_X_data_name ' ;' ] );

                    X_name = obj.observed_factors(v).get_data_name();
                    
                    eval ( [ 't = (' hat_X_data_name ' .* ' X_name ') .* ' ...
                             ' (log( (' hat_X_data_name ' .* ' X_name ') ) - ' ...
                             'log(' X_name ...
                             ') ) - ( ' hat_X_data_name ' .* ' X_name ')' ...
                             '+ ' X_name ...
                             ';' ]);
                    for di = 1:length(obj.observed_factors(v).dims)
                        t = sum(t);
                    end
                    kl(v) = t;
                end
            else
                kl = 0;
            end

        end





        function [found] = got_factor(obj, v, alpha)
        % does model v have latent factor alpha
            if ~isa(alpha, 'TFFactor')
                throw(MException('GCTFModel:GotFactor', ...
                                 'alpha must be a TFFactor instance'));
            end

            found = false;
            for mi = 1:length(obj.R{v})
                if obj.R{v}(mi) == alpha
                    found = true;
                    return
                end
            end
        end











        function [size] = get_element_size(obj)
        % returns number of elements for this model
            size = 0;
            for ofi = 1:length(obj.observed_factors)
                size = size + ...
                       obj.observed_factors(ofi).get_element_size();
            end

            keys = obj.unique_latent_factors.keys();
            for ufi = 1:length(keys)
                size = size + ...
                       obj.unique_latent_factors(keys{ufi}) ...
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

        function [code_name] = get_coded_factor_name(obj, index)
        % returns coded name of the factor at index, index must be
        % a TFFactor object
        % internally for detecting data using same dimensions
        % used with temporary factors in order to re-use same
        % dimension data structures

            dims = index.dims;

            code_name = ['factor_' ...
                char(obj.order_dims(TFDimensionList2cell(dims)))'];
        end

%        function [inds] = latent_factor_indices(obj)
%        % returns a matrix of latent factor indices
%        % column 1 -> observed factor index
%        % column 2 -> factor index
%            inds = [];
%            for ofi = 1:length(obj.observed_factors)
%                ind = [];
%                for offi = 1:length(obj.R{ofi})
%                    if obj.R{ofi}(offi).isLatent
%                        ind = [ ofi offi ];
%                    end
%                end
%                if length(ind)
%                    inds = [ inds ; [ofi offi] ];
%                end
%            end
%        end


        function [] = rand_init_latent_factors(obj, type, imax)

            if ~strcmp(type, 'all') && ~strcmp(type, 'nonClamped')
                throw(MException('PLTFModel:WrongInitType', ...
                                 ['Supported init type values: all, ' ...
                                  'nonClamped']));
            end

            ulfk = obj.unique_latent_factors.keys();
            for ki = 1:length(ulfk)

                if strcmp(type, 'all') || ...
                        ( strcmp(type, 'nonClamped') && ...
                          obj.unique_latent_factors(ulkf(ki)).isInput == 0 )

                    data_name = [obj.get_factor_data_name( ...
                        obj.unique_latent_factors(char(ulfk(ki))) ) ];

                    factor = obj.unique_latent_factors(char(ulfk(ki)));
                    if nargin==2
                        factor.rand_init(obj.dims, 100, data_name);
                    else
                        factor.rand_init(obj.dims, imax, data_name);
                    end

                end
            end

        end



    end
end