classdef TFDimension

    properties
        name='';
        cardinality=0;
    end

    methods
        function r = eq(a,b)
            r=logical(0);
            if a.name == b.name
                if a.cardinality == b.cardinality
                    r=logical(1);
                else
                    display(['ERROR: inconsistent dimensions! If ' ...
                             'dimension names are same cardinalities ' ...
                             'can not be different!']);
                    r=logical(0);
                end
            else
                r=logical(0);
            end
        end
    end

end