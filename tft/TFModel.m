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
%   parafac_model = TFModel('name', 'Parafac', 'factors', [p_A p_B p_C X], 'dims', [dim_i dim_j dim_k dim_r]);
%   
%   parafac_model.rand_init_latent_factors('all');
%
%   See also TFDimension, TFFactor

classdef TFModel

    properties
        name='';          % name of the model
        factors=TFFactor; % array of TFFactor

        dims=TFDimension; % array of TFDimension used by factors defines
                          % order of dimensions on memory

        cost=0;

    end

    methods

        function obj = TFModel(varargin)
            p = inputParser;
            addParamValue(p, 'name', '', @isstr);
            addParamValue(p, 'factors', [], @isvector);
            addParamValue(p, 'dims', [], @isvector);

            parse(p,varargin{:});

            % check if all dims elements are TFDimension objects
            for i = 1:length(p.Results.dims)
                if ~isa(p.Results.dims(i), 'TFDimension')
                    err = MException( ...
                        ['TFModel:DimensionNotTFDimension'], ...
                        ['Dimensions of TFModel must be ' ...
                         'TFDimension objects']);
                    throw(err);
                end
            end
            obj.dims = p.Results.dims;

            % check if all factors are TFFactor objects
            for i = 1:length(p.Results.factors)
                if ~isa(p.Results.factors(i), 'TFFactor')
                    err = MException( ...
                        ['TFModel:FactorsNotTFFactor'], ...
                        ['Factors of TFModel must be ' ...
                         'TFFactor objects']);
                    throw(err);
                end
            end
            obj.factors = p.Results.factors;

            obj.name = p.Results.name;

        end

        function [] = pltf(obj, iternum, contract_type)
        % performs PLTF update operations on the model
        % returns estimated TFFactor (\hat(X))
        % iternum: number of iterations
        % contract_type: if optimal contractions are performed in
        % least memory using sequence
        % defines order of contraction operations

            if nargin == 2
                contract_type = '';
            end

            % init optimal model cache
            global ocs_cache;
            ocs_cache = [];

            hat_X = obj.observed_factor;
            hat_X.name = 'hat_X';

            eval( [ 'global ' obj.observed_factor.get_data_name() ...
                    ';' ] );
            global hat_X_data;
            hat_X.rand_init(obj.dims, 100);

            mask = obj.observed_factor;
            mask.name = 'mask';
            global mask_data;
            mask_data = ones(size(hat_X_data));

            KL=zeros(1,iternum);
            for iter = 1:iternum
                iter
                for alpha=1:length(obj.latent_factor_indices)
                    % access global data
                    X_name = ...
                        obj.factors(obj.observed_factor_index) ...
                        .get_data_name();
                    eval(['global ' X_name ';']);
                    Z_alpha_name = ...
                        obj.factors(alpha).get_data_name();
                    eval( [ 'global ' Z_alpha_name ';' ] );




                    % recalculate hat_X
                    newmodel = obj;

                    if iter==1 && alpha==1
                        %g = newmodel.schedule_dp();
                        %system([ 'rm /tmp/img.eps; echo '' ' g.print_dot  [' '' |' ...
                        %                    ' dot -o /tmp/img.eps; ' ...
                        %                    ' display  /tmp/img.eps ' ...
                        %                    ' ' ] ] );
                    end

                    % perform contraction
                    newmodel = newmodel.contract_all(contract_type);

                    % store result in hat_X_data
                    result_name = ...
                        newmodel.get_first_non_observed_factor() ...
                        .get_data_name();
                    eval(['global ' result_name ';'] );
                    eval(['hat_X_data = ' result_name ';' ] ); 



                    % store X / hat_X in hat_X data
                    eval( [ 'hat_X_data  =  ' ...
                            X_name ...
                            ' ./ ' ...
                            ' hat_X_data ;' ] );


                    % generate D1
                    obj.delta(alpha, 'D1_data', contract_type, hat_X);

                    % generate D2
                    obj.delta(alpha, 'D2_data', contract_type, mask);

                    % update Z_alpha
                    global D1_data D2_data;
                    eval( [ Z_alpha_name '=' Z_alpha_name ' .* ' ...
                            'D1_data'                     ' ./ ' ...
                            'D2_data ;' ] );


                    % calculate KL divergence
                    eval ( [ 'KL(iter) = sum(sum(sum( (hat_X_data .* ' X_name ') .* ' ...
                             ' (log( (hat_X_data .* ' X_name ') ) - ' ...
                             'log(' X_name ...
                             ') ) - (hat_X_data .* ' X_name ')' ...
                             '+ ' X_name ...
                             ')));' ]);

                end
            end
            display(['KL divergence over iterations: ']);
            display(KL);
            plot(KL);
            title('KL divergence over iterations');
            xlabel('iteration number');
            ylabel('KL divergence');
        end

        function [] = delta(obj, alpha, output_name, contract_type, ...
                            A)
        % PLTF delta function implementation
        % alpha: index of latent factor in TFModel.factors array
        % which will be updated
        % name: unique name used as the name of calculated delta
        % factor data
        % contract_type: if optimal contractions are performed in
        % least memory using sequence
        % A: operand element of delta function assumed all ones if
        % not given

            
            % create new model for delta operation
            d_model = obj;

            % remove observed factor
            d_model.factors(d_model.observed_factor_index) = [];

            % add Z_alpha as new observed factor
            d_model.factors(alpha).isLatent = 0;
            d_model.factors(alpha).isObserved= 1;

            % if given, add A as a new latent factor
            if nargin == 5
                A.isLatent = 1;
                A.isObserved = 0;
                d_model.factors = [d_model.factors A];
            end


            %g = d_model.schedule_dp();
            %system( [ 'rm /tmp/img.eps; echo '' ' g.print_dot  [' '' |' ...
            %                    ' dot -o /tmp/img.eps ;  display  /tmp/img.eps; ' ] ] );
            % perform contraction
            d_model=d_model.contract_all(contract_type);

            eval( [ 'global ' output_name ';'] );
            eval( [ 'global ' ...
                    d_model.get_first_non_observed_factor().get_data_name() ';'...
                  ] );

            eval( [ output_name '=' ...
                    d_model.get_first_non_observed_factor().get_data_name() ...
                    ';' ]);
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

        function [uncontraction_dims] = get_uncontraction_dims(obj)
        % returns list of TFDimension objects representing
        % dimensions which are required to be added to the current
        % TFModel object in order to reach initial model where
        % uncontraction_dims = {}
        % Calculation is performed as follows: 
        % uncontraction_dims = contraction_dims - current_contraction_dims

            contraction_dims = obj.get_contraction_dims();
            current_contraction_dims = ...
                obj.get_current_contraction_dims();
            uncontraction_dims = [];

            for cdi = 1:length(contraction_dims)
                found=0;
                for ccdi = 1:length(current_contraction_dims)
                    if contraction_dims(cdi) == ...
                            char(current_contraction_dims(ccdi))
                        found=1;
                        break;
                    end
                end
                if ~found
                    uncontraction_dims = [ uncontraction_dims ...
                                        contraction_dims(cdi) ];
                end
            end
        end

        function [ocs_dims] = get_optimal_contraction_sequence_dims(obj)
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


            graph = obj.schedule_dp();
            t = graph.optimal_edges;
            t(t==0) = Inf;
            ocs_models = [];
            i = length(t);
            while i ~= 1
                ocs_models = [ ocs_models graph.node_list(i) ];
                i = find( t(:,i) == min(t(:, i)) );
            end
            ocs_models = [ ocs_models graph.node_list(i) ];

            ocs_dims = [];
            for i = 1:(length(ocs_models)-1)
                ocs_dims = [ ocs_dims ...
                             { setdiff( ...
                                 ocs_models(i)...
                                 .get_current_contraction_dims, ...
                                 ocs_models(i+ ...
                                            1)...
                                 .get_current_contraction_dims) }; ...
                           ];
            end

            %display([ 'cache store' ])
            %for a =1:length(ocs_dims)
            %    ocs_dims{a}
            %end
            ocs_cache = [ ocs_cache TFOCSCache(obj, ocs_dims) ];
        end
            

        function [graph] = schedule_dp(obj)
        % returns a tree of TFModel generated as a result of the
        % search for least memory consuming contraction sequence
        % with a dynamic programming approach

            output_dims = obj.get_contraction_dims();
            contraction_dims = obj.get_contraction_dims();
            
            % generate final state of contraction operation to be
            % used as the initial state of dp search
            end_node = obj.contract_all(); % must not be optimal,
                                           % infinite loop!

            graph = TFGraph;
            process_nodes = [end_node];
            processing_node = 1;

            while length(process_nodes) >= processing_node
                cur_node = process_nodes(processing_node);
                cur_node = cur_node.update_cost_from_latent();

                % init graph.node_list
                if length(graph.node_list) == 0
                    graph.node_list = [cur_node];
                    graph=graph.clear_edges();
                end

                uncontraction_dims = cur_node.get_uncontraction_dims();
                for udi = 1:length(uncontraction_dims)
                    new_node = cur_node.uncontract(obj, ...
                                                   uncontraction_dims(udi));
                    new_node = new_node.update_cost_from_latent();

                    % memoization
                    nnidx = graph.exists(new_node);
                    if nnidx
                        graph = graph.update_node(cur_node, new_node, nnidx);
                    else
                        graph = graph.append_node(cur_node, new_node);
                        process_nodes = [ process_nodes new_node ];
                    end
                end

                processing_node = processing_node + 1;

            end

        end


        function [newmodel] = uncontract(obj, orig_model, dim)
        % returns a new TFModel generated by adding given
        % TFDimension object dim to the model.
        % dim: TFDimension object to uncontract
        % orig_model: provides original non-contracted model data

            newmodel = obj;

            newmodel.name = [ newmodel.name  ...
                              '_uncontract_' dim.name ];

            uncontract_dims = obj.get_uncontraction_dims();
            other_dims = []; % already contracted dimensions
            for i=1:length(uncontract_dims)
                if uncontract_dims(i) ~= dim
                    other_dims = [ other_dims ...
                                   uncontract_dims(i) ];
                end
            end

            % reset newmodel's factors
            newmodel.factors = [];

            % stores factors which will contribute to the temporary
            % factor
            tmp_factor_parents = [];


            % populate newmode.factors with factors independant
            % from uncontraction operation
            for fi = 1:length(orig_model.factors)
                found = 0;
                for fdi = 1:length(orig_model.factors(fi).dims)
                    for odi = 1:length(other_dims)
                        % if any factor contains any one of the
                        % other_dims can not use it as it is, must
                        % create a temporary factor for those
                        % here we identify factors we will use
                        % without modification

                        if other_dims(odi) == ...
                                orig_model.factors(fi).dims(fdi)
                            found=1;
                            break
                        end
                    end
                    if found, break; end
                end

                if ~found
                    % if other_dims are not found in this factor then
                    % use this factor as it is in the new node

                    newmodel.factors = [newmodel.factors ...
                                        orig_model.factors(fi) ];
                else
                    % in other case this factor will be inspected
                    % further to generate temporary factor
                    tmp_factor_parents = [ tmp_factor_parents ...
                                        orig_model.factors(fi) ];
                end
            end

            % make sure factors are unique
            %newmodel.factors = unique(newmodel.factors);

            % inspect tmp_factor_parents and generate a temporary
            % model
            tmpf = TFFactor;
            tmpf.isTemp = 1;
            tmpf.name = 'tmp';
            names={};
            tmpf.dims = [];

            % add all dimensions of all parent factors
            for tfpi = 1:length(tmp_factor_parents)
                for di = 1:length(tmp_factor_parents(tfpi).dims)
                    if tmp_factor_parents(tfpi).isLatent
                        % if dimension is not one of the other_dims
                        % then add it to the temporary factor
                        found = 0;
                        for odi = 1:length(other_dims)
                            if other_dims(odi) == ...
                                    tmp_factor_parents(tfpi).dims(di)
                                found = 1;
                                break;
                            end
                        end


                        if ~found
                            % if not already added
                            found2 = 0;
                            for tmpfdind = 1:length(tmpf.dims)
                                if tmpf.dims(tmpfdind) == ...
                                        tmp_factor_parents(tfpi).dims(di)
                                    found2 = 1;
                                    break;
                                end
                            end

                            if ~found2

                                tmpf.dims = [ tmpf.dims ...
                                              tmp_factor_parents(tfpi) ...
                                              .dims(di)];
                                names = [ names ...
                                          tmp_factor_parents(tfpi) ...
                                          .dims(di).name];
                            end
                        end
                    end
                end
            end

            % if no names are found then there is no temporary
            % factor added
            if length(names)
                % make sure tmp.factor dims are unique
                %'önce'
                %tmpf.dims.name
                %tmpf.dims = unique(tmpf.dims);
                %'sonra'
                %tmpf.dims.name
                
                names=obj.order_dims(unique(names));
                for d = 1:length(names)
                    tmpf.name = [ tmpf.name '_' char(names(d)) ];
                end
                %tmpf.name = [tmpf.name '_minus_' ? ];
                
                newmodel.factors = [ newmodel.factors tmpf ];

                newmodel = newmodel.update_cost_from_latent();
            end

        end

        function [newmodel] = contract(obj, dim)
        % returns a new TFModel generated by contracting obj with
        % dim which may add new temporary factors. 
        % dim: TFDimension or char array or cell with the name of
        % the dimension

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
            end
            tmp.name = [tmp.name '_minus_' dim];




            eval( [ 'global ' tmp.get_data_name()  ';'] );

            if length(contracted_factor_inds) == 1
                % no multiplication
                eval( [ 'global ' ...
                        obj.factors(contracted_factor_inds(1)) ...
                        .get_data_name() ';'] );
                eval( [ tmp.get_data_name() ' = ' ...
                        obj.factors(contracted_factor_inds(1)) ...
                        .get_data_name() ';'] );
            else

                % multiply first two into tmp data
                eval( [ 'global' ...
                        ' ' obj.factors(contracted_factor_inds(1)).get_data_name() ...
                        ' ' obj.factors(contracted_factor_inds(2)).get_data_name() ';'] );

                eval( [ tmp.get_data_name() ' = bsxfun (@times, ' ...
                        obj.factors(contracted_factor_inds(1)).get_data_name() ', '...
                        obj.factors(contracted_factor_inds(2)).get_data_name() ');' ...
                      ] );

                % multiply tmp data with other factors
                for cfii = 3:length(contracted_factor_inds)
                    eval( [ 'global '...
                            obj.factors(contracted_factor_inds(cfii)) ...
                            .get_data_name() ';'] );
                    eval( [ tmp.get_data_name() ' = bsxfun (@times, ' ...
                            tmp.get_data_name() ','...
                            obj.factors(contracted_factor_inds(cfii)) ...
                            .get_data_name() ');' ] );
                end
            end


            % sum contraction dimensions on tmp data
            
            con_dim_index = obj.get_dimension_index(dim);

            eval( [ tmp.get_data_name() ' = sum( ' ...
                    tmp.get_data_name() ', ' ...
                    num2str(con_dim_index) ');'] );
            
            newmodel.factors = [newmodel.factors tmp];


            % remove contracted factors
            % other dimensions live within the tmp factor
            removed_num=0; % removal breaks loop index
            for cfii = 1:length(contracted_factor_inds)
                newmodel.factors(contracted_factor_inds(cfii) ...
                                 - removed_num) = [];
                removed_num = removed_num + 1;
            end
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
        % index of the output factor
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


        function [newmodel] = contract_all(obj, contract_type)
        % performs all necessary contraction operations for the
        % model. if contract_type argument is equal to 'optimal'
        % then schedule_dp() is used to find the optimal (least
        % memory using) sequence and the optimal sequence is used
        % to contract all necessary dimensions. Otherwise
        % get_contraction_dims() is used to order contraction
        % operation which does not order dimensions

            if nargin == 2 && strcmp(contract_type, 'optimal')
                contract_dims = ...
                    obj.get_optimal_contraction_sequence_dims();

                %for i=1:length(contract_dims)
                %    display(['optimal contracting ' ...
                %             char(contract_dims{i})]);
                %end
            else
                contract_dims = obj.get_contraction_dims();

                %for i=1:length(contract_dims)
                %    display(['contracting ' contract_dims(i).name]);
                %end
            end

            newmodel = obj;
            for i = 1:length(contract_dims)
                newmodel = newmodel.contract(contract_dims(i));
            end
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

        function [] = rand_init_latent_factors(obj, type, imax)

            if ~strcmp(type, 'all') && ~strcmp(type, 'nonClamped')
                throw(MException('TFModel:WrongInitType', ...
                                 ['Supported init type values: all, ' ...
                                  'nonClamped']));
            end

            for fi=1:length(obj.latent_factor_indices)

                if strcmp(type, 'all') || ...
                        ( strcmp(type, 'nonClamped') && ...
                          obj.factors(fi).isInput == 0 )
                    
                    if nargin==2
                        obj.factors(fi).rand_init(obj.dims);
                    else
                        obj.factors(fi).rand_init(obj.dims, imax);
                    end

                end
            end

        end

    end

end