classdef TFModel
% Class representing data required to describe a tensor
% factorization model

    properties
        name='';    % name of the model
        factors=[]; % array of TFFactor



        dims=[];    % array of TFDimension used by factors defines
                    % order of dimensions on memory


        tree_index = 0; % index of current model in externally
                        % stored model list

        parent_tree_indices=[]; % array of integers representing
                                % indices of parents of this
                                % model. list of models are assumed
                                % to be stored externally
        children_tree_indices=[];
        cost=0;
    end

    methods

        function [hat_X] = pltf(obj, iternum)
        % performs PLTF update operations on the model
        % returns estimated TFFactor (\hat(X))
        % iternum: number of iterations
        % defines order of contraction operations

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

                    % perform contraction
                    newmodel=newmodel.contract_all();

                    % store result in hat_X_data
                    result_name = ...
                        newmodel.get_first_non_observed_factor() ...
                        .get_data_name();
                    eval(['global ' result_name]);
                    eval(['hat_X_data = ' result_name ';' ] ); 



                    % store X / hat_X in hat_X data
                    eval( [ 'hat_X_data  =  ' ...
                            X_name ...
                            ' ./ ' ...
                            ' hat_X_data ;' ] );


                    % generate D1
                    obj.delta(alpha, 'D1_data', hat_X);

                    % generate D2
                    obj.delta(alpha, 'D2_data', mask);

                    % update Z_alpha
                    global D1_data D2_data;
                    'bura'
                    eval( [ Z_alpha_name '=' Z_alpha_name ' .* ' ...
                            'D1_data'                     ' ./ ' ...
                            'D2_data ' ] );


                    % calculate KL divergence
                    eval ( [ 'KL(iter) = sum(sum(sum( (hat_X_data .* ' X_name ') .* ' ...
                             ' (log( (hat_X_data .* ' X_name ') ) - ' ...
                             'log(' X_name ...
                             ') ) - (hat_X_data .* ' X_name ')' ...
                             '+ ' X_name ...
                             ')));' ]);

                end
            end

            plot(KL)
        end

        function [] = delta(obj, alpha, output_name, A)
        % PLTF delta function implementation
        % alpha: index of latent factor in TFModel.factors array
        % which will be updated
        % name: unique name used as the name of calculated delta
        % factor data
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
            if nargin == 4
                A.isLatent = 1;
                A.isObserved = 0;
                d_model.factors = [d_model.factors A];
            end

            % perform contraction
            'delta contract_all'
            d_model.factors.name
            d_model.factors.isObserved
            d_model=d_model.contract_all();
            d_model.factors.dims

            eval( [ 'global ' output_name ] );
            eval( [ 'global ' ...
                    d_model.get_first_non_observed_factor().get_data_name() ...
                  ] );

            eval( [ output_name '=' ...
                    d_model.get_first_non_observed_factor().get_data_name() ...
                    ';' ]);
        end

        function [r] = eq(a,b)
            r = false;

            if length(a.factors) == length(b.factors)
                for f_a = 1:length(a.factors)
                    found = 0;
                    for f_b = 1:length(b.factors)
                        if a.factors(f_a) == b.factors(f_b)
                            found = 1;
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

        function [newmodel] = contract(obj, dim)
        % returns a new TFModel generated by contracting obj with
        % dim which may add new temporary factors. 
        % dim: TFDimension or char array or cell with the name of
        % the dimension

            if isa(dim, 'TFDimension')
                dim = dim.name;
            elseif isa(dim, 'char') || isa(dim, 'cell')
                dim = char(dim);
            else
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
            name={};
            ['contracting dim ' num2str(dim)]
            ['factors ']
            obj.factors.name
            ['contracted factor inds ' num2str(contracted_factor_inds)]
            
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
                        name=[name ...
                              newmodel.factors(contracted_factor_inds(cfii)).dims(cfi_dim).name];
                        %display(['added from factor index' ...
                        %         contracted_factor_inds(cfii)])
                        %display(['added from factor' newmodel.factors(contracted_factor_inds(cfii)).name])
                        %display(['addd dim' char(newmodel.factors(contracted_factor_inds(cfii)).dims(cfi_dim).name)])
                    end
                end
            end
            name=unique(name);
            for d = 1:length(name)
                tmp.name = [tmp.name '_' char(name(d))];
            end
            tmp.name = [tmp.name '_minus_' dim];




            eval( [ 'global ' tmp.get_data_name()  ] );
            ['tmp name ' tmp.get_data_name ]

            if length(contracted_factor_inds) == 1
                % no multiplication
                eval( [ 'global ' ...
                        obj.factors(contracted_factor_inds(1)) ...
                        .get_data_name() ] );
                eval( [ tmp.get_data_name() ' = ' ...
                        obj.factors(contracted_factor_inds(1)) ...
                        .get_data_name() ';'] );
            else

                % multiply first two into tmp data
                ['multiply 1 ' ...
                 obj.factors(contracted_factor_inds(1)) ...
                 .get_data_name() ' ' ...
                 obj.factors(contracted_factor_inds(2)).get_data_name() ...
                ]

                eval( [ 'global' ...
                        ' ' obj.factors(contracted_factor_inds(1)).get_data_name() ...
                        ' ' obj.factors(contracted_factor_inds(2)).get_data_name() ] );

                ['size of operands 1 ']
                eval(['size(' ...
                      obj.factors(contracted_factor_inds(1)) ...
                      .get_data_name() ')']);
                eval(['size(' ...
                      obj.factors(contracted_factor_inds(2)) ...
                      .get_data_name() ')']);



                eval( [ tmp.get_data_name() ' = bsxfun (@times, ' ...
                        obj.factors(contracted_factor_inds(1)).get_data_name() ', '...
                        obj.factors(contracted_factor_inds(2)).get_data_name() ');' ...
                      ] );

                % multiply tmp data with other factors
                for cfii = 3:length(contracted_factor_inds)
                    ['multiply 2' ...
                     obj.factors(contracted_factor_inds(cfii)) ...
                     .get_data_name()]

                    ['size of operands 2']
                    eval(['size(' tmp.get_data_name() ')']);
                    eval(['size(' ...
                          obj.factors(contracted_factor_inds(cfii)) ...
                          .get_data_name() ')']);


                    eval( [ 'global '...
                            obj.factors(contracted_factor_inds(cfii)) ...
                            .get_data_name() ] );
                    eval( [ tmp.get_data_name() ' = bsxfun (@times, ' ...
                            tmp.get_data_name() ','...
                            obj.factors(contracted_factor_inds(cfii)) ...
                            .get_data_name() ');' ] );
                end

                ['size of tmp data 1']
                eval(['size(' tmp.get_data_name ')']);

            end


            % sum contraction dimensions on tmp data

            %['tmp dims' tmp.dims.name]
            %['observed dims' obj.observed_factor.dims.name]

            
            con_dim_index = obj.get_dimension_index(dim)

            eval( [ tmp.get_data_name() ' = sum( ' ...
                    tmp.get_data_name() ', ' ...
                    num2str(con_dim_index) ');'] );

            ['size of tmp data 2']
            eval(['size(' tmp.get_data_name ')']);
            
            newmodel.factors = [newmodel.factors tmp];


            % remove dim from newmodel's dims array
%            newdims=[];
%            for d = 1:length(newmodel.dims)
%                if newmodel.dims(d) == char(dim)
%                else
%                    newdims = [ newdims newmodel.dims(d) ];
%                end
%            end
%            newmodel.dims = newdims;


            % remove contracted factors
            % other dimensions live within the tmp factor
            removed_num=0; % removal breaks loop index
            for cfii = 1:length(contracted_factor_inds)
                newmodel.factors(contracted_factor_inds(cfii) ...
                                 - removed_num) = [];
                removed_num = removed_num + 1;
            end
        end


        function [dn fn all_edges] = print(obj)
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
            all_edges = [ all_edges ' ]' ]
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


        function [newmodel] = contract_all(obj)
            contract_dims = obj.get_contraction_dims();
            ['contract dims name ' contract_dims.name]
            newmodel = obj;
            for i = 1:length(contract_dims)
                newmodel = newmodel.contract(contract_dims(i));
            end



            % store final result in observed factor's global data
            %result_name = ...
            %    newmodel.get_first_non_observed_factor() ...
            %    .get_data_name();

            % observed_name = obj.observed_factor.get_data_name();

            %eval([ 'global ' result_name ' ' observed_name ]);
            %eval([ observed_name ' = ' result_name ';']);
            %eval([ observed_name ]);

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

        function [factor_inds] = latent_factor_indices(obj)
            factor_inds=[];
            for f=1:length(obj.factors)
                if obj.factors(f).isLatent
                    factor_inds = [ factor_inds f ];
                end
            end
        end


        function [factors] = latent_factors(obj)
            factors=[]
            for f=1:length(obj.factors)
                if obj.factors(f).isLatent
                    factors = [ factors obj.factors(f) ]
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

    end

end