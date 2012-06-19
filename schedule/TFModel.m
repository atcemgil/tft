classdef TFModel
    properties
        % array of TFFactor
        factors=[];

        % array of TFDimension used by factors
        % defines order of dimensions on memory
        dims=[];
    end

    methods

        % returns a string to be used by fgplot
        function [dn fn all_edges] = print(obj)

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

        % order given cell of dimension names according to
        % factor.dims order
        function [ordered_index_chars] = order_dim_chars(obj, ...
                                                         dims_array)
            tmp=cell{length(dims_array), 3};
            default_order = get_default_order(tf_model);

            for i = 1:length(index_chars_array)
                tmp(i,1) = index_chars_array(i);
                tmp(i,2) = get_index_card(tf_model, index_chars_array(i));
                tmp(i,3) = find(default_order==index_chars_array(i)); % order of the index
                                                                      % in the model
            end

            tmp=sortrows(tmp,3); % sort by the order of the model indices
            ordered_index_chars=char(tmp(:,1)');

        end


        function [contract_dims] = get_contraction_dims(obj)
            output_chars = {};
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved

                    % for each dimension of this factor
                    for i=1:length(obj.factors(f).dims)
                        output_chars = [output_chars ...
                                        obj.factors(f).dims(i).name]
                    end

                end
            end

            obj.order_dims(unique(output_chars))
        end

        function [factors] = latent_factors(obj)
            factors=[]
            for f=1:length(obj.factors)
                if obj.factors(f).isLatent
                    factors = [ factors obj.factors ]
                end
            end
        end

        function [factors] = observed_factors(obj)
            factors=[]
            for f=1:length(obj.factors)
                if obj.factors(f).isObserved
                    factors = [ factors obj.factors ]
                end
            end
        end

        function [factors] = input_factors(obj)
            factors=[]
            for f=1:length(obj.factors)
                if obj.factors(f).isInput
                    factors = [ factors obj.factors ]
                end
            end
        end

        function [factors] = temp_factors(obj)
            factors=[]
            for f=1:length(obj.factors)
                if obj.factors(f).isTemp
                    factors = [ factors obj.factors ]
                end
            end
        end
        
    end

end