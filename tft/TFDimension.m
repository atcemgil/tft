classdef TFDimension

    properties
        name;
        cardinality;
    end

    methods

        function obj = TFDimension(varargin)
            p = inputParser;
            addParamValue(p, 'name', '', @isstr);
            addParamValue(p, 'cardinality', 0, @isnumeric);
            parse(p,varargin{:});
            obj.name = p.Results.name;
            obj.cardinality = p.Results.cardinality;
        end


        %function [obj,idx,varargout]=sort(obj,varargin)
        %varargout=cell(1,nargout-2);
        %    [~,idx,varargout{:}]=sort([obj.name],varargin{:});
        %        obj=obj(idx);
        %end


        function r = eq(a,b)
            r=logical(0);

            if isa(a, 'TFDimension') && isa(b, 'TFDimension')
                aname=a.name;
                bname=b.name;
                acard=a.cardinality;
                bcard=b.cardinality;
            elseif isa(a, 'TFDimension') && isa(b, 'char')
                aname=a.name;
                bname=b;
                acard=0;
                bcard=0;
            elseif isa(b, 'TFDimension') && isa(a, 'char')
                aname=a;
                bname=b.name;
                acard=0;
                bcard=0;
            else
                throw(MException('TFDimension:TypeError', ...
                ['ERROR: unsupported input classes ' ...
                 'a:' class(a) ' '...
                 'b:' class(b) ' '] ));
            end


            if aname == bname
                if acard == bcard
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


        function r = ne(a,b)
            r = ~(a==b);
        end


    end

end