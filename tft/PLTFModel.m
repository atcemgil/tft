% Represents data required to describe a tensor factorization model
%
%   A model is represented with its name and factors array of
%   TFFactor objects. dims array of TFDimension objects define
%   dimensions used by this model. Dimensions are transfered to C++
%   in the order defined in dims array. See examples below for a
%   full model description.
%
%   Examples:
%   
%   dim_i = TFDimension('name', 'i', 'cardinality', 5);
%   dim_j = TFDimension('cardinality', 6, 'name', 'j');
%   dim_k = TFDimension('cardinality', 7, 'name', 'k');
%   dim_r = TFDimension('cardinality', 10, 'name', 'r');
%   
%   p_A=TFFactor('name', 'p_A', 'type', 'latent', 'dims', [dim_i dim_r]);
%   p_B=TFFactor('name', 'p_B', 'type', 'latent', 'dims', [dim_j dim_r]);
%   p_C=TFFactor('name', 'p_C', 'type', 'latent', 'dims', [dim_k dim_r]);
%   
%   parafac_model = PLTFModel('name', 'Parafac', 'factors', [p_A p_B p_C X], 'dims', [dim_i dim_j dim_k dim_r]);
%   
%   parafac_model.rand_init_latent_factors('all');
%
%   See also TFDimension, TFFactor

classdef PLTFModel


    properties
        name = '';            % name of the model
        factors = [TFFactor]; % array of TFFactor

        dims = TFDimension; % array of TFDimension used by factors defines
                            % order of dimensions on memory

        cost = 0;

    end


    methods

        function obj = PLTFModel(varargin)
            p = inputParser;
            addParamValue(p, 'name', '', @isstr);
            addParamValue(p, 'factors', [], @isvector);
            addParamValue(p, 'dims', [], @isvector);

            parse(p,varargin{:});

            % check if all dims elements are TFDimension objects
            for i = 1:length(p.Results.dims)
                if ~isa(p.Results.dims(i), 'TFDimension')
                    err = MException( ...
                        ['PLTFModel:DimensionNotTFDimension'], ...
                        ['Dimensions of PLTFModel must be ' ...
                         'TFDimension objects']);
                    throw(err);
                end
            end
            obj.dims = p.Results.dims;

            % check if all factors are TFFactor objects
            for i = 1:length(p.Results.factors)
                if ~isa(p.Results.factors(i), 'TFFactor')
                    err = MException( ...
                        ['PLTFModel:FactorsNotTFFactor'], ...
                        ['Factors of PLTFModel must be ' ...
                         'TFFactor objects']);
                    throw(err);
                end
            end
            obj.factors = p.Results.factors;

            obj.name = p.Results.name;

        end



        function [] = pltf_optimal_dot(obj, filename)
        % Calls PLTF function and collects output of print_dot from
        % each GTM operation and plots in a single graph
            global last_node_id;
            last_node_id = 1;

            dot_str = obj.pltf(1, 'optimal', 'mem_analysis', ...
                               'yes');

            str = [ 'digraph structs{' char(10) ...
                    'rankdir=LR;' char(10) ...
                    'node [shape=plaintext];' char(10) ...
                    'splines=false; ' char(10)];
            cmd = [ 'echo ''' str char(10) dot_str ...
                    '}'' | dot -T svg ' ];
            if nargin == 2
                system([ cmd ' -o ' filename ]);
            else
                system([ cmd ' | display ' ]);
            end
        end



        function [ dot_data ] = pltf(obj, iternum, contract_type, ...
                                     operation_type, ...
                                     return_dot_data)
        % performs PLTF update operations on the model
        % returns estimated TFFactor (\hat(X))
        %
        % iternum: number of iterations
        %
        % contract_type: if equals to 'optimal', contractions are
        % performed in least memory using sequence defines order of
        % contraction operations, if equals to 'full', full tensor
        % is used for contraction operations instead of generating
        % temporary factors.
        %
        % operation_type: if equals to 'compute' normal contraction
        % operations are performed. if equals to 'mem_analysis'
        % operation are not performed but memory requirement is
        % calculated and reported. In 'mem_analysis' mode iternum
        % argument is not considered only a single iteration is
        % performed.
        %
        % return_dot_data: if equals to 'yes' then dot data for all
        % generalized tensor multiplication operations are returned
        % in a string. In this case contract_type is expected to be
        % equal to 'optimal', operation_type is expected to be
        % equal to 'mem_analysis', iternum is expected to be equal
        % to 1 . See pltf_optimal_dot function for example usage.

            if nargin == 2
                contract_type = '';
            end
            if nargin < 4
                operation_type = 'compute';
            end
            if nargin < 5
                return_dot_data = 'no';
            end
            dot_data = '';


            % init optimal model cache
            global ocs_cache;
            ocs_cache = [];

            % initalize obj.cost with memory requirements of the
            % model elements
            obj.cost = obj.get_element_size();
            display( [ 'obj.cost ' num2str(obj.cost) ] );

            hat_X = obj.observed_factor;
            hat_X.name = 'hat_X';
            % hat_X requires extra memory
            obj.cost = obj.cost + hat_X.get_element_size();
            display( [ 'obj.cost ' num2str(obj.cost) ] );

            if strcmp( operation_type, 'compute' )
                eval( [ 'global ' obj.get_factor_data_name( obj.observed_factor) ...
                        ';' ] );
                global hat_X_data;
                hat_X.rand_init(obj.dims, 100);
            end

            mask = obj.observed_factor;
            mask.name = 'mask';
            % mask requires extra memory
            obj.cost = obj.cost + mask.get_element_size();
            display( [ 'obj.cost ' num2str(obj.cost) ] );

            if strcmp( operation_type, 'compute' )
                global mask_data;
                mask_data = ones(size(hat_X_data));
                KL=zeros(1,iternum);
                for iter = 1:iternum
                    display(['iteration' char(9) num2str(iter)]);
                    [ kl cost ] = obj.pltf_iteration(contract_type, ...
                                                     hat_X, ...
                                                     mask, ...
                                                     operation_type);
                    KL(iter) = kl;
                end

                display(['KL divergence over iterations: ']);
                display(KL);
                plot(KL);
                title('KL divergence over iterations');
                xlabel('iteration number');
                ylabel('KL divergence');

            elseif strcmp( operation_type, 'mem_analysis' )
                [ kl cost dot_data ] = ...
                    obj.pltf_iteration(contract_type, hat_X, ...
                                       mask, ...
                                       operation_type, ...
                                       return_dot_data );
            end



            obj.cost = obj.cost + cost;
            display( ['e9 ' num2str(obj.cost) ' <- ' num2str(cost) ] );

            if strcmp( contract_type, 'full' )
                global F_size;
                obj.cost = obj.cost + F_size;
                display([ 'e10 ' num2str(obj.cost) ' <- ' ...
                          num2str(F_size) ]);
            end

            display([char(10) ...
                     'data elements required: ' num2str(obj.cost) ...
                     char(10) ...
                     ['memory size with (8 byte) double precision: ' ...
                      num2str(8 * obj.cost / 1000 / 1000) ' MB' ] ] );
        end




        function [ kl cost dot_data ] = pltf_iteration(obj, ...
                                                   contract_type, ...
                                                   hat_X, mask, ...
                                                   operation_type, ...
                                                   return_dot_data )
            % helper function for the pltf inner loop
            % operation_type: 'compute' if actual computation is
            % requested, 'mem_analysis' if only memory usage
            % computation is requested
            % 
            % returns KL divergence value calculated at the end of
            % the operations. dot_data contains
            % graphviz data for the generalized tensor
            % multiplication operations if return_dot_data is equal
            % to 'yes'.

            if nargin < 6
                return_dot_data = 'no';
            end
            dot_data = '';

            cost = 0;
            for alpha=1:length(obj.latent_factor_indices)
                % access global data
                eval( [ 'global ' obj.get_factor_data_name( obj.observed_factor ) ...
                        ';' ] );
                global hat_X_data mask_data;
                X_name = obj.get_factor_data_name( obj.observed_factor );
                eval(['global ' X_name ';']);
                Z_alpha_name = obj.get_factor_data_name( ...
                    obj.factors(alpha) );
                eval( [ 'global ' Z_alpha_name ';' ] );

                % recalculate hat_X
                newmodel = obj;


                % only not used if full, compute, no (so do it
                % always) was under if below
                graph = newmodel.schedule_dp();                
                if strcmp(return_dot_data, 'yes')
                    global last_node_id;
                    [str last_node_id] = graph.print_dot(last_node_id, ...
                                                         're-calculate hat_X');
                    dot_data = [ str char(10) dot_data ];

                    global oru_models;
                    if ~length(oru_models)
                        oru_models = [ newmodel ];
                    end
                end

                % perform contraction
                % store result in hat_X_data
                [ ~ ] = ...
                    newmodel.contract_all(contract_type, ...
                                          operation_type, ...
                                          'hat_X_data', graph);
                if strcmp(contract_type, 'optimal')
                    % does not work on 'full' contraction
                    cost = cost + graph.get_optimal_path_cost();
                end
                display( ['e1 X_hat ' num2str(cost) ' <- ' ...
                          num2str(graph.get_optimal_path_cost()) ...
                         ] );


                %result_name = ...
                %    newmodel.get_first_non_observed_factor() ...
                %    .get_data_name();
                %eval(['global ' result_name ';'] );
                %eval(['hat_X_data = ' result_name ';' ] ); 



                % store X / hat_X in hat_X data
                if strcmp( operation_type, 'compute' )
                    eval( [ 'hat_X_data  =  ' ...
                            X_name ...
                            ' ./ ' ...
                            ' hat_X_data ;' ] );
                end


                % generate D1
                [ dd c ] = obj.delta(alpha, 'D1_data', ...
                                        contract_type, ...
                                        operation_type, ...
                                        hat_X, ...
                                        return_dot_data );
                if strcmp( return_dot_data, 'yes') 
                    dot_data = [ dd char(10) dot_data ];
                end

                if strcmp(contract_type, 'optimal')
                    cost = cost + c + ...
                           global_data_size(Z_alpha_name);
                else
                    % for full contraction type
                    cost = cost + global_data_size(Z_alpha_name);
                end

                display( ['e2 D1 Z_' num2str(alpha) ' ' num2str(cost) ' c ' ...
                          num2str(c) ' ' num2str(global_data_size(Z_alpha_name)) ...
                         ] );

                % generate D2
                [ dd ] = obj.delta(alpha, 'D2_data', ...
                                   contract_type, ...
                                   operation_type, ...
                                   mask, ...
                                   return_dot_data);
                if strcmp( return_dot_data, 'yes') 
                    dot_data = [ dd char(10) dot_data ];
                end

                % works for both optimal and full contraction
                cost = cost + global_data_size(Z_alpha_name);
                display( ['e2 D1 Z_' num2str(alpha) ' ' num2str(cost) ' ' ...
                          num2str(global_data_size(Z_alpha_name)) ...
                         ] );


                % update Z_alpha
                if strcmp( operation_type, 'compute' )
                    global D1_data D2_data;
                    eval( [ Z_alpha_name '=' Z_alpha_name ' .* ' ...
                            'D1_data'                     ' ./ ' ...
                            'D2_data ;' ] );
                end
            end

            if strcmp( operation_type, 'compute' )
                % calculate KL divergence
                eval ( [ 'kl = sum(sum(sum( (hat_X_data .* ' X_name ') .* ' ...
                         ' (log( (hat_X_data .* ' X_name ') ) - ' ...
                         'log(' X_name ...
                         ') ) - (hat_X_data .* ' X_name ')' ...
                         '+ ' X_name ...
                         ')));' ]);
            else
                kl = 0;
            end

        end




        function [ dot_data cost ] = delta(obj, alpha, ...
                                                output_name, ...
                                                contract_type, ...
                                                operation_type, ...
                                                A, return_dot_data)
        % PLTF delta function implementation
        % alpha: index of latent factor in PLTFModel.factors array
        % which will be updated
        % name: unique name used as the name of calculated delta
        % factor data
        % contract_type: see description in pltf function
        % A: operand element of delta function assumed all ones if
        % not given
        %
        % dot_data contains
        % graphviz data for the generalized tensor
        % multiplication operations if return_dot_data is equal
        % to 'yes'.

            
            if nargin < 7
                return_dot_data = 'no';
            end

            dot_data = '';

            % create new model for delta operation
            d_model = obj;

            % remove observed factor
            d_model.factors(d_model.observed_factor_index) = [];

            % add Z_alpha as new observed factor
            d_model.factors(alpha).isLatent = 0;
            d_model.factors(alpha).isObserved= 1;

            % if given, add A as a new latent factor
            if nargin == 7
                A.isLatent = 1;
                A.isObserved = 0;
                d_model.factors = [d_model.factors A];
            end


            % only not used if full, compute, no (so do it
            % always) was under if below
            graph = d_model.schedule_dp();
            if strcmp(return_dot_data, 'yes')
                global last_node_id;
                title = ['alpha ' num2str(alpha) ' ' ...
                         output_name ];
                [ dot_data last_node_id ] = ...
                    graph.print_dot(last_node_id, ...
                                    title );

                if strcmp(output_name, 'D1_data')
                    global oru_models;
                    oru_model = d_model;
                    oru_model.name = title;
                    oru_models = [ oru_models oru_model ];
                end
            end

            % perform contraction
            [ ~ ] = d_model.contract_all(contract_type, ...
                                              operation_type, ...
                                              output_name, graph);

            %'e4'
            if strcmp( contract_type, 'optimal' )
                cost = graph.get_optimal_path_cost();
            else
                % does not work on full contraction type
                cost = 0;
            end
        end




        function [graph] = schedule_dp(obj)
        % returns a tree of PLTFModel generated as a result of the
        % search for least memory consuming contraction sequence
        % with a dynamic programming approach

            output_dims = obj.get_contraction_dims();
            contraction_dims = obj.get_contraction_dims();
            
            graph = TFGraph;
            process_nodes = [obj];
            processing_node = 1;

            % init graph.node_list
            graph.node_list = obj;
            graph=graph.clear_edges();

            while length(process_nodes) >= processing_node
                cur_node = process_nodes(processing_node);
                %cur_node = cur_node.update_cost_from_temp();

                contraction_dims = cur_node.get_current_contraction_dims();
                for udi = 1:length(contraction_dims)

                    new_node = cur_node.contract( contraction_dims(udi), ...
                                                  'mem_analysis', ...
                                                  '' );

                    % last operation does not have memory cost data
                    % is handled outside of pltf_iteration
                    if ~length(new_node.get_current_contraction_dims())
                        new_node.cost = 0;
                    end

                    % memoization
                    nnidx = graph.exists(new_node);
                    if nnidx
                        %['old node ' new_node.name]
                        graph = graph.update_node(cur_node, new_node, nnidx);
                    else
                        %['new node ' new_node.name]
                        graph = graph.append_node(cur_node, new_node);
                        process_nodes = [ process_nodes new_node ];
                    end
                end

                processing_node = processing_node + 1;

            end

        end





        function [newmodel ] = contract_all(obj, contract_type, ...
                                                     operation_type, ...
                                                     output_name, graph )
        % Performs all necessary contraction operations for the
        % model. If contract_type argument is equal to 'optimal'
        % then schedule_dp() is used to find the optimal (least
        % memory using) sequence and the optimal sequence is used
        % to contract all necessary dimensions. Otherwise
        % get_contraction_dims() is used to order contraction
        % operation which does not order dimensions. If
        % contract_type equals to 'full', no intermediate
        % contractions are performed, full tensor is calculated and
        % then contracted into the output tensor
        % output_name: name of the global data structure to store
        % the final result into, not used if operation type is not
        % 'compute'.
        % 
        % operation_type argument may be 'compute' or
        % 'mem_analysis'. Normal operations are performed in
        % 'compute' (default), memory usage is reported in
        % 'mem_analysis' case.
        %
        % returns the model generated after contracting all
        % necessary dimension and the total extra memory required
        % by all temporary elements

            if nargin < 4
                output_name = '';
            end

            if nargin < 3
                operation_type = 'compute';
            end

            if nargin < 2
                contract_type = 'standard';
            end

            if strcmp(operation_type, 'compute') && ...
                    isempty(output_name)
                throw(MException('PLTFModel:ContractAllNoOutputName', ...
                                 ['output_name must be specified if ' ...
                                  'operation type is ''compute''' ]));
            end



            if strcmp( contract_type, 'optimal' )
                [contract_dims] = ...
                    obj.get_optimal_contraction_sequence_dims(graph);

                %for i=1:length(contract_dims)
                %    display(['optimal contracting ' ...
                %             char(contract_dims{i})]);
                %end
            elseif  ~strcmp( contract_type, 'full' )
                contract_dims = obj.get_contraction_dims();

                %for i=1:length(contract_dims)
                %    display(['contracting ' contract_dims(i).name]);
                %end
            end




            newmodel = obj;

            if strcmp( contract_type, 'full')
                [ newmodel ] = ...
                    obj.contract_full(operation_type);
                %['e5' contract_type]
            else
                for i = 1:length(contract_dims)
                    if i == length(contract_dims)
                        on = output_name;
                    else
                        on = '';
                    end

                    [ newmodel ]= ...
                        newmodel.contract(contract_dims(i), ...
                                          operation_type, ...
                                          on );

                end
            end
        end




        function [newmodel] = contract_full(obj, operation_type, ...
                                                      output_name)
        % generates a new full (temporary) tensor, multiplies all
        % latent tensors in to the full tensor and then 
        % contracts full tensor over necessary indices and returns
        % newmodel with full_tensor and contracted result in global
        % data named 'output_name'

            % generate full tensor
            F = TFFactor;
            F.name = 'full_tensor';
            F.isTemp = 1;
            F.dims = obj.dims;    % full indices
            global F_size;
            F_size = F.get_element_size();

            global full_tensor_data;

            if strcmp( operation_type, 'compute' )
                % generate global full_tensor_data

                sz = '';
                for adi = 1:length(obj.dims)
                    if adi ~= 1
                        sz = [sz ', '];
                    end
                    sz = [sz num2str(obj.dims(adi).cardinality) ];
                end
                eval( [ ' full_tensor_data = ones(' sz ');'] );
            else
                % in case of mem_analysis  make length of global data
                % greater than 0 for memory size computation
                full_tensor_data = 1;
            end

            if strcmp( operation_type, 'compute' )
                % access global data of all latent factors
                lfi = obj.latent_factor_indices;
                for lfii = 1:length(lfi)
                    eval(['global ' ...
                          obj.get_factor_data_name( obj.factors(lfi(lfii)) ) ...
                         ]);
                end

                % multiply all latent tensors, store result in data_F
                for lfii = 1:length(lfi)
                    % following tensors should be multiplied with data_F
                    eval([ 'full_tensor_data = bsxfun(@times, full_tensor_data, ' ...
                           obj.get_factor_data_name( ...
                               obj.factors(lfi(lfii)) ) ');' ]);
                end

                % contract necessary dimensions from full_tensor_data
                contract_dims = obj.get_contraction_dims();
                for cdi = 1:length(contract_dims)
                    full_tensor_data = sum( full_tensor_data,        ...
                                            obj.get_dimension_index( ...
                                                contract_dims(cdi)) );
                end
            end

            newmodel = obj;
            newmodel.factors = [ obj.observed_factor F ];
        end




        function [newmodel] = contract(obj, dim, operation_type, ...
                                                 output_name)
        % returns a new PLTFModel generated by contracting obj with
        % dim which may add new temporary factors and mem_delta
        % integer value. 
        % dim: TFDimension or char array or cell with the name of
        % the dimension
        % operation_type: if equals to 'compute' contraction is
        % calculated. if equals to 'mem_analysis' data elements are
        % not created.
        % output_name: if has length > 0 do not generate temporary
        % factor but use global data storage named 'output
        % name'. 
        %[ 'contract START dim length ' num2str(length(dim))]

            if length(dim) == 0
                throw(MException('PLTFModel:NoContractionDimensionSpecified', ...
                                 ['must specify contraction ' ...
                                  'dimension']));
            end


            if isa(dim, 'TFDimension')
                dim = dim.name;
            elseif isa(dim, 'cell')
                dim = char(dim{1});
            elseif ~isa(dim, 'char')
                display(['ERROR: unsupported dim type ' class(dim) ...
                         'was expecting TFDimension, cell or char ' ...
                         'array']);
                return
            end

            newmodel = obj;
            newmodel.name = [obj.name '_' dim];

            % remove dim from the new model's factors'
            contracted_factor_inds = [];
            for f = 1:length(newmodel.factors)
                if ~newmodel.factors(f).isObserved
                    ind = ...
                        newmodel.factors(f).got_dimension(char(dim));
                    
                    if ind ~= 0
                        % remove this dimension from this factor
                        newmodel.factors(f).dims(ind) = [];

                        contracted_factor_inds = ...
                            [ contracted_factor_inds f ];
                    end
                end
            end

            % add a temporary factor including dimensions of
            % contracted factors other than contracted dimension
            tmp=TFFactor;
            tmp.isTemp = 1;
            tmp.name = 'tmp';
            names={};
            
            for cfii = 1:length(contracted_factor_inds)
                % for each dimension of the contracted factor
                for cfi_dim = 1:length(newmodel.factors(contracted_factor_inds(cfii)).dims)
                    found=0;
                    for ti = 1:length(tmp.dims)
                        if tmp.dims(ti) == ...
                                newmodel.factors(contracted_factor_inds(cfii)).dims(cfi_dim)
                            found=1;
                            break;
                        end
                    end
                    if found == 0
                        tmp.dims = [tmp.dims ...
                                    newmodel.factors(contracted_factor_inds(cfii)) ...
                                    .dims(cfi_dim)];
                        names=[names ...
                              newmodel.factors(contracted_factor_inds(cfii)).dims(cfi_dim).name];
                        %display(['added from factor index' ...
                        %         contracted_factor_inds(cfii)])
                        %display(['added from factor' newmodel.factors(contracted_factor_inds(cfii)).name])
                        %display(['addd dim' char(newmodel.factors(contracted_factor_inds(cfii)).dims(cfi_dim).name)])
                    end
                end
            end
            names=unique(names);
            for d = 1:length(names)
                tmp.name = [tmp.name '_' char(names(d))];
                %tmp.name
            end
            tmp.name = [tmp.name '_minus_' dim];




            % calculate output data
            if isempty(output_name)
                % if output if not given store in global variable
                % with tmp.coded_name
                on = obj.get_factor_data_name(tmp);
                %eval( [ 'global ' on ';' ])
           else
                on = output_name;

                % TODO: make sure output has correct dimensions
                % eval( [ 'global ' on  ';'] );
                % tmpsz = size(tmp.get_data_name());
            end


            %['SELECTED on ' on  ' tmp.name ' tmp.name ' operation_type ' ...
            % operation_type]
            if strcmp( operation_type, 'compute' )
                eval( [ 'global ' on  ';'] );

                %[ 'on0: ' on '' ]
                %eval([ 'on_len0 = length( ' on ' )' ]);


                if length(contracted_factor_inds) == 1
                    %'sdfa111'
                    % no multiplication
                    eval( [ 'global ' ...
                            obj.get_factor_data_name( obj.factors(contracted_factor_inds(1)) ) ...
                            ';'] );
                    %['aa :' obj.get_factor_data_name( ...
                    %    obj.factors(contracted_factor_inds(1)) ) ]
                    %['aalen :']
                    eval( [ 'length( ' obj.get_factor_data_name( ...
                        obj.factors(contracted_factor_inds(1)) ) ');' ])


                    %[ obj.get_factor_data_name( ...
                    %    obj.factors(contracted_factor_inds(1)) ) ...
                    %  ' -> ' ...
                    %  on ]
                    eval( [ on ' = ' ...
                            obj.get_factor_data_name( obj.factors(contracted_factor_inds(1)) ) ...
                            ';'] );
                else
                    %'sdfa222'
                    % multiply first two into tmp data
                    eval( [ 'global' ...
                            ' ' obj.get_factor_data_name( obj.factors(contracted_factor_inds(1)) ) ...
                            ' ' obj.get_factor_data_name( obj.factors(contracted_factor_inds(2)) ) ...
                            ';'] );


                    %[ obj.get_factor_data_name( ...
                    %    obj.factors(contracted_factor_inds(1)) ) ...
                    %  ' * ' ...
                    %  obj.get_factor_data_name( ...
                    %      obj.factors(contracted_factor_inds(2)) ) ...
                    %  ' -> ' ...
                    %  on ]
                    eval( [ on ' = bsxfun (@times, ' ...
                            obj.get_factor_data_name( obj.factors(contracted_factor_inds(1)) ) ', '...
                            obj.get_factor_data_name( obj.factors(contracted_factor_inds(2)) ) ');' ...
                          ] );

                    % multiply tmp data with other factors
                    for cfii = 3:length(contracted_factor_inds)
                        eval( [ 'global '...
                                obj.get_factor_data_name( obj.factors(contracted_factor_inds(cfii)) ) ...
                                ';'] );
                        eval( [ on ' = bsxfun (@times, ' ...
                                on ','...
                                ojb.get_factor_data_name( obj.factors(contracted_factor_inds(cfii)) ) ...
                                ');' ] );
                    end
                end


                % sum contraction dimensions on tmp data
                
                con_dim_index = obj.get_dimension_index(dim);

                eval( [ on ' = sum( ' ...
                        on ', ' ...
                        num2str(con_dim_index) ');'] );

            else
                % in case of mem_analysis  make length of global data
                % greater than 0 for memory size computation
                
                % schedule_dp must not perform memory size calculation
                %stack = dbstack();
                %if ~strcmp(stack(2).name, ...
                %           'PLTFModel.schedule_dp') 
                %    eval([ on ' = 1 ;' ]);
                %end
            end


            if isempty(output_name)
                newmodel.factors = [newmodel.factors tmp];
            end

            % remove contracted factors
            % other dimensions live within the tmp factor
            removed_num=0; % removal breaks loop index
            for cfii = 1:length(contracted_factor_inds)
                newmodel.factors(contracted_factor_inds(cfii) ...
                                 - removed_num) = [];
                removed_num = removed_num + 1;
            end


            if isempty(output_name)
                newmodel = newmodel.update_cost_from_temp();
            end
        end




        function [ocs_dims] = get_optimal_contraction_sequence_dims(obj, ...
                                                              graph)
        % Runs schedule_dp function to generate graph with least
        % memory using contraction sequence information in it. Then
        % searches TFGraph.optimal_edges for the optimal path and
        % returns a cell list of contraction dimension names.

            global ocs_cache;

            found = false;
            for o = 1:length(ocs_cache)
                if ocs_cache(o).model == obj
                    found=true;
                    break
                end
            end

            if found
                %display('cache hit')

                %display([ 'ocs dims' ])
                ocs_dims = ocs_cache(o).ocs_dims;
                %for a =1:length(ocs_dims)
                %    ocs_dims{a}
                %end
                return
            %else
            %    display('cache miss')
            end


            %graph = obj.schedule_dp();
            [ocs_dims] = graph.optimal_sequence_from_graph();
            %cost
            %obj.cost = obj.cost + cost;


            %display([ 'cache store' ])
            %for a =1:length(ocs_dims)
            %    ocs_dims{a}
            %end

            ocs_cache = [ ocs_cache TFOCSCache(obj, ocs_dims) ];
        end




        function [obj] = update_cost_from_temp(obj)
            obj.cost = 0;
            lfi = obj.latent_factor_indices();
            for fi = 1:length(lfi)
                if obj.factors(lfi(fi)).isTemp
                    obj.factors(lfi(fi)).name;
                    obj.cost = obj.cost + ...
                        obj.factors(lfi(fi)).get_element_size();
                end
            end
            %['updatecost ' obj.name ' ' num2str(obj.cost)]
        end




        function [obj] = update_cost_from_latent(obj)
            obj.cost = 0;
            lfi=obj.latent_factor_indices();
            for fi = 1:length(lfi)
                obj.cost = obj.cost + ...
                    obj.factors(lfi(fi)).get_element_size();
            end
        end




        function [dn fn all_edges] = print_ubigraph(obj)
        % returns a string to be used by fgplot

        % add dimension nodes
            dn = [ '[ ''' ];
            for i =1:length(obj.dims)
                if i ~= 1
                    dn = [ dn ''',''' ];
                end
                dn = [ dn obj.dims(i).name ];
            end
            dn = [ dn ''' ] ' ];

            % add factor nodes
            fn = [ '[ ''' ];
            for i =1:length(obj.factors)
                if i ~= 1
                    fn = [ fn ''',''' ];
                end
                fn = [ fn obj.factors(i).name ];
            end
            fn = [ fn ''' ] ' ];

            % add edges
            all_edges=['[ ' ];
            for d = 1:length(obj.dims)
                if d ~= 1
                    all_edges = [ all_edges ' , ' ];
                end

                edges=['[ ' ];

                % include this dimension if a factor uses it
                for f = 1:length(obj.factors)
                    for fd = 1:length(obj.factors(f).dims)
                        if obj.factors(f).dims(fd) == obj.dims(d)
                            if length(edges) ~= 2
                                edges = [ edges ''',''' ];
                            else
                                edges = [ edges '''' ];
                            end
                            edges = [ edges obj.factors(f).name ];
                        end
                    end
                end

                all_edges = [ all_edges edges ''' ]' ];
            end
            all_edges = [ all_edges ' ]' ];
        end




        function [size] = get_element_size(obj)
        % returns number of elements for this model
            size=0;
            for f = 1:length(obj.factors)
                if obj.factors(f).isObserved == 0
                    size = size + ...
                           obj.factors(f).get_element_size();
                end
            end
        end




        function [card] = get_index_card(obj, index_char)
        % returns cardinality of a given name of a dimension
            for d = 1:length(obj.dims)
                if obj.dims(d).name == index_char
                    card = obj.dims(d).cardinality;
                    break
                end
            end
        end




        function [factor] = get_first_non_observed_factor(obj)
        % returns first non observed factor index, used to return
        % index of the output factor when all latent factors are
        % contracted out. in this case only output (observed) and a
        % single temporary non-observed factors should be present
        % in the model (ie fully contracted model).
            for f = 1:length(obj.factors)
                if obj.factors(f).isObserved == 0
                    factor = obj.factors(f);
                    return
                end
            end
        end




        function [ind] = get_first_non_observed_factor_index(obj)
        % returns first non observed factor index, used to return
        % index of the output factor
            for f = 1:length(obj.factors)
                if obj.factors(f).isObserved == 0
                    ind = f;
                    return
                end
            end
        end




        function [ind] = find_cell_char(obj, chars)
        % returns index of a given char array in object's dimension array
            for ind=1:length(obj.dims)
                if obj.dims(ind).name == chars
                    return
                end
            end
            ind=0;
        end




       function [r] = get_dimension_index(obj, dim)
        % returns index of dimension dim in obj.dims if obj
        % contains TFDimension (or char) dim returns 0 otherwise

            r=0;
            for d = 1:length(obj.dims)
                if obj.dims(d) == dim
                    r=d;
                    break;
                end
            end           
       end




       function [ordered_index_chars] = order_dims(obj, ...
                                                    dims_array)
            % order given cell of dimension names according to
            % factor.dims order

            tmp=cell(length(dims_array), 3);

            tmp(:,1) = dims_array;
            for i = 1:length(dims_array)
                tmp{i,2} = obj.get_index_card(char(dims_array(i)));
                tmp{i,3} = obj.find_cell_char(char(dims_array(i))); % order of the index
                                                                    % in the model
            end

            tmp=sortrows(tmp,3); % sort by the order of the model indices
            ordered_index_chars=tmp(:,1)';
        end




        function [contract_dims] = get_contraction_dims(obj)
        % returns cell of dimensions which must be contracted to
        % calculate output factor(s)

            output_chars = {};
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved

                    % for each dimension of this factor
                    for i=1:length(obj.factors(f).dims)
                        output_chars = [output_chars ...
                                        obj.factors(f).dims(i).name];
                    end

                end
            end

            % contraction dimensions: alldims - output_dims
            contract_dims={};
            for d_a = 1:length(obj.dims)
                found=0;
                for d_o = 1:length(output_chars)
                    if obj.dims(d_a) == char(output_chars(d_o))
                        found=1;
                        break
                    end
                end

                if found == 0
                    contract_dims = [contract_dims obj.dims(d_a)];
                    %['add '  obj.dims(d_a).name]
                end
            end

            %['return ' contract_dims.name]
            %contract_dims = obj.order_dims(unique(contract_dims));
        end




        function [contract_dims] = get_current_contraction_dims(obj)
        % returns cell of TFDimensions which must be contracted to
        % calculate output factor(s) by using current factors to
        % calculate all dimension not model.dims. (model.dims is always
        % fixed to maximum possible dimension list.)

            output_chars = {};
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved

                    % for each dimension of this factor
                    for i=1:length(obj.factors(f).dims)
                        output_chars = [output_chars ...
                                        obj.factors(f).dims(i).name];
                    end

                end
            end

            % contraction dimensions: alldims - output_dims
            alldims={};
            for f = 1:length(obj.factors)
                for d = 1:length(obj.factors(f).dims)
                    found = 0;
                    n=obj.factors(f).dims(d).name;
                    for i=1:length(alldims)
                        if char(alldims(i)) == n
                            found = 1;
                            break
                        end
                    end

                    if ~found
                        alldims = [alldims n];
                    end
                end
            end

            contract_dims={};
            for d_a = 1:length(alldims)
                found=0;
                for d_o = 1:length(output_chars)
                    if char(alldims(d_a)) == char(output_chars(d_o))
                        found=1;
                        break
                    end
                end

                if found == 0
                    contract_dims = [contract_dims alldims(d_a)];
                    %['add '  alldims(d_a).name]
                end
            end

            %['return ' contract_dims.name]
            %contract_dims = obj.order_dims(unique(contract_dims));
        end




        function [factor_inds] = latent_factor_indices(obj)
            factor_inds=[];
            for f=1:length(obj.factors)
                if obj.factors(f).isLatent
                    factor_inds = [ factor_inds f ];
                end
            end
        end




        function [factors] = latent_factors(obj)
            factors=[];
            for f=1:length(obj.factors)
                if obj.factors(f).isLatent
                    factors = [ factors obj.factors(f) ];
                end
            end
        end




        function [factor_ind] = observed_factor_index(obj)
            factor_ind=0;
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved
                    factor_ind = f;
                    return
                end
            end
        end




        function [factor] = observed_factor(obj)
        % returns first observed factor (useful for PLTF operations)
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved
                    factor = obj.factors(f);
                    return
                end
            end
        end




        function [factors] = observed_factors(obj)
            factors=[];
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved
                    factors = [ factors obj.factors(f) ];
                end
            end
        end




        function [factors] = input_factors(obj)
            factors=[];
            for f=1:length(obj.factors)
                if obj.factors(f).isInput
                    factors = [ factors obj.factors(f) ];
                end
            end
        end




        function [factors] = temp_factors(obj)
            factors=[];
            for f=1:length(obj.factors)
                if obj.factors(f).isTemp
                    factors = [ factors obj.factors(f) ];
                end
            end
        end




        function [index] = get_factor_index(obj, factor)
        % returns index of a factor in factor list
        % if factor does not exist returns 0
            index = 0;
            for f = 1:length(obj.factors)
                if factor == obj.factor(f)
                    index = f;
                    break;
                end
            end
        end




        function [name] = get_factor_data_name(obj, index)
        % returns global data name of the factor at index of
        % factors TFFactor array. index may be an TFFactor instance.
        % 
        % To preserve model elements' data, their naming structure
        % is different than temporary factors. Model data elements
        % use TFFactor.get_data_name whereas temporary elements use
        % PLTFModel.get_coded_factor_name. This it is possible to
        % re-use memory structures with same dimensions
            if isa(index, 'TFFactor')
                factor = index;
            elseif isnumeric(index)
                factor = obj.factors(index);
            end

            if factor.isTemp
                name = obj.get_coded_factor_name(factor);
            else
                % assume we have model element
                name = factor.get_data_name();
            end
        end


        function [code_name] = get_coded_factor_name(obj, index)
        % returns coded name of the factor at index, used
        % index may be a factor, in which case factor does not need
        % to exist in obj.factors
        % internally for detecting data using same dimensions
        % used with temporary factors in order to re-use same
        % dimension data structures

            if isnumeric(index)
                dims = obj.factors(index).dims;
            elseif isa(index, 'TFFactor')
                dims = index.dims;
            end

            code_name = ['factor_' ...
                char(obj.order_dims(TFDimensionList2cell(dims)))'];
        end



        function [] = rand_init_latent_factors(obj, type, imax)

            if ~strcmp(type, 'all') && ~strcmp(type, 'nonClamped')
                throw(MException('PLTFModel:WrongInitType', ...
                                 ['Supported init type values: all, ' ...
                                  'nonClamped']));
            end

            for fi=1:length(obj.latent_factor_indices)

                if strcmp(type, 'all') || ...
                        ( strcmp(type, 'nonClamped') && ...
                          obj.factors(fi).isInput == 0 )

                    data_name = [obj.get_factor_data_name(fi)];
                    
                    if nargin==2
                        obj.factors(fi).rand_init(obj.dims, 100, data_name);
                    else
                        obj.factors(fi).rand_init(obj.dims, imax, data_name);
                    end

                end
            end

        end




        function [r] = eq(a,b)
            r = false;

            % mark matched b factors
            % if there are any unmarked -> inequal
            % problematic case: 
            % a.factors ( ip, jpi ) , b.factors (  ip, pi )
            % b==a matches all b objects with a.factors(1)
            % but a~=b !

            b_marks = zeros(size(b.factors));

            if length(a.factors) == length(b.factors)
                for f_a = 1:length(a.factors)
                    found = 0;
                    for f_b = 1:length(b.factors)
                        if a.factors(f_a) == b.factors(f_b) && ...
                                b_marks(f_b) == 0
                            found = 1;
                            b_marks(f_b) = 1;
                            break
                        end
                    end

                    if found == 0
                        return
                    end
                end

                r = true;
            end
        end




        function [] = full_pltf_mem_analysis(obj)
            obj.pltf(1, 'full', 'mem_analysis');
        end




        function [] = optimal_pltf_mem_analysis(obj)
            obj.pltf(1, 'optimal', 'mem_analysis');
        end




    end




end