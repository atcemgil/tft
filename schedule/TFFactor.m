classdef TFFactor

    properties
        name='';

        dims=[]; % array of TFDimension
        data;    % contains data of this factor

        isLatent=0;
        isObserved=0;
        isInput=0;
        isTemp=0;
    end

    methods

        function [size] = get_element_size(obj)
        % returns number of elements for this factor
            size=1;
            for d = 1:length(obj.dims)
                size = size * obj.dims(d).cardinality;
            end
        end

        function r = eq(a,b)
            r=false;

            %if a.name == b.name && ...
            %a.isLatent == b.isLatent && ...
            %a.isObserved == b.isObserved && ...
            %a.isInput == b.isInput && ...
            %a.isTemp == b.isTemp

            if length(a.dims) == length(b.dims)% && ...
                for d_a = 1:length(a.dims)
                    found = 0;
                    for d_b = 1:length(b.dims)
                        if a.dims(d_a) == b.dims(d_b)
                            found = 1;
                            break;
                        end
                    end

                    if found == 0
                        return
                    end
                end

                r=true;
            end
        end

        function [r] = got_dimension(obj, dim)
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

        function [name] = get_data_name(obj)
            name = [obj.name '_data'];
        end

        function [] = rand_init(obj, all_dims, imax)

            eval( [ 'global ' obj.get_data_name() ] );
            sz = '';
            for ad = 1:length(all_dims)
                if ad ~= 1
                    sz = [sz ', '];
                end

                found=0;
                for od = 1:length(obj.dims)
                    if all_dims(ad) == obj.dims(od)
                        found=1;
                        break
                    end
                end

                if found
                    sz = [sz num2str(all_dims(ad).cardinality) ];
                else
                    sz = [sz num2str(1) ];
                end
            end

            if nargin == 2
                eval( [ obj.get_data_name() ' = rand(' sz ');'] );
            else
                eval( [ obj.get_data_name() ...
                        ' = randi(' num2str(imax) ', ' sz ');' ] );
            end
        end


        function [contract_dims] = ...
                get_contraction_to(obj, sub_dims)
            contract_dims = [];
            for od = 1:length(obj.dims)
                found=0;
                for sd = 1:length(sub_dims)
                    if obj.dims(od) == sub_dims(sd)
                        found=1;
                        break;
                    end
                end

                if ~found
                    contract_dims = [ contract_dims obj.dims(od) ];
                end
            end
        end
        

    end

end