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
            r=logical(0);

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

                r=logical(1);
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
                    
    end

end