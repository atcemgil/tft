classdef GCTFModel
    properties
        % array of TFModel objects
        tfmodels=[];
    end

    methods

        % returns a string to be used by fgplot
        function [dn fn all_edges] = print(obj)
            dn=['[ '];
            fn=['[ '];
            dims={};
            factors={};

            for m=1:length(obj.tfmodels)
    
                for i =1:length(obj.tfmodels(m).dims)
                    txt = obj.tfmodels(m).dims(i).name;
                    if ~length(strfind(dn, ['''' txt '''']))
                        if  ~(i == 1 && m == 1)
                            dn = [ dn ',' ];
                        end

                        dn = [ dn '''' txt ''''];
                        dims = [ dims txt ];
                    end
                end
    
                for i =1:length(obj.tfmodels(m).factors)
                    txt = obj.tfmodels(m).factors(i).name;
                    if ~length(strfind(fn, ['''' txt '''']))
                        if  ~(i == 1 && m == 1)
                            fn = [ fn ',' ];
                        end

                        fn = [ fn '''' txt '''' ];
                        factors = [ factors txt ];
                    end
                end
            end

            dn = [ dn ' ] ' ];
            fn = [ fn ' ] ' ];


            all_edges=['[ '];
            for d = 1:length(dims)
                if  ~(d == 1)
                    all_edges = [ all_edges ' , ' ];
                end
                
                edges=['[ ' ];
                
                % include this dimension if a factor uses it
                for m = 1:length(obj.tfmodels)
                    for f = 1:length(obj.tfmodels(m).factors)
                        for fd = 1:length(obj.tfmodels(m).factors(f).dims)
                            if obj.tfmodels(m).factors(f).dims(fd).name == dims{d}
                                if length(edges) ~= 2
                                    edges = [ edges ''',''' ];
                                else
                                    edges = [ edges '''' ];
                                end
                                edges = [ edges obj.tfmodels(m).factors(f).name ];
                            end
                        end
                    end
                end
                all_edges = [ all_edges edges ''' ]' ];
            end
            all_edges = [ all_edges ' ]' ];
    
        end
    
    end
end