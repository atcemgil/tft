% Represents a single dimension
%   Dimensions are represented with two attributes. Name of the
%   dimension and cardinality of the dimension.
%   
%   Examples: 
%   dim_i = TFDimension('name', 'i', 'cardinality', 5);
%   dim_j = TFDimension('cardinality', 6, 'name', 'j'); 
%
% See also TFFactor, PLTFModel
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


        function r = eq_TFDimension(a,b)
        % written for not callins isa() for performance
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


        function r = ne(a,b)
            r = ~(a==b);
        end


    end

end