% Represents a PLTF factor 
%   Factors are identified with their names of any length
%   character array and dimension they occupy data in. dims array
%   contains an array of TFDimension objects.
%
%   Depending on the input arguments 'latent', 'type' and
%   'isClamped' internal flag variables are set. See examples for
%   details
%
%   Examples:
%   A = TFFactor('name', 'A', 'type', 'latent', 'dims', [dim_i dim_p]);
%   C = TFFactor('name', 'C', 'type', 'latent', 'dims', [dim_k dim_r], 'isClamped', true);
%   X = TFFactor('name', 'X', 'type', 'observed', 'dims', [dim_i, dim_j, dim_k]);
%
%   See also TFDimension, PLTFModel

classdef TFFactor

    properties
        name = '';

        dims = TFDimension;  % array of TFDimension

        data_mat_file;       % name of the mat file containing ...
                             % a variable with obj.name with correct dimensions

        isLatent=0;
        isObserved=0;
        isInput=0;
        isTemp=0;
        isReUsed=0; % true if this tensor is re-used (temporary) factor


        size = -1;

        % used for temporary tensors
        % stores names of source factors
        source_factor_names = {};
        contracted_index_names = {};
    end

    methods

        function obj = TFFactor(varargin)
            p = inputParser;
            addParamValue(p, 'name', '', @isstr);
            types={'latent', 'observed', 'temp'};
            addParamValue(p, 'type', 'latent', @(x) any(validatestring(x,types)));
            addParamValue(p, 'isClamped', 0, @islogical);
            addParamValue(p, 'dims', [], @isvector);
            addParamValue(p, 'data_mat_file', '', @isstr);

            parse(p,varargin{:});

            % check if all dims elements are TFDimension objects
            for i = 1:length(p.Results.dims)
                if ~isa(p.Results.dims(i), 'TFDimension')
                    err = MException( ...
                        ['TFFactor:DimensionNotTFDimension'], ...
                        ['Dimensions of TFFactor must be ' ...
                         'TFDimension objects']);
                    throw(err);
                end
            end
            obj.dims = p.Results.dims;

            obj.name = p.Results.name;

            if strcmp(p.Results.type, 'latent')
                obj.isLatent = 1;
            elseif strcmp(p.Results.type, 'observed')
                obj.isObserved = 1;
            elseif strcmp(p.Results.type, 'temp')
                obj.isTemp = 1;
            end

            obj.isInput = p.Results.isClamped;

            obj.data_mat_file = p.Results.data_mat_file;
        end


        function [sn] = get_short_name(obj)
            if obj.isTemp
                sn = '';
                for i=1:length(obj.source_factor_names)
                    sn = [ sn obj.source_factor_names{i} ];
                end

                sn = [ sn '_' ];
                for i=1:length(obj.contracted_index_names)
                    sn = [ sn char(obj.contracted_index_names(i)) ];
                end
            else
                sn = obj.name;
            end
        end


        function [size] = get_element_size(obj)
        % returns number of elements for this factor
            if obj.size ~= -1
                size = obj.size;
            elseif length(obj.dims) == 0
                err = MException( ...
                    ['TFFactor:Dimension list size 0!'] );
                throw(err);
            else
                size=1;
                for d = 1:length(obj.dims)
                    size = size * obj.dims(d).cardinality;
                end
            end
        end


        function [obj, idx] = sort(obj, varargin)
            [~,idx] = sort([obj.name],varargin{:}); 
            obj = obj(idx);
        end


        function r = eq(a,b)
            r=false;

            %if a.name == b.name && ...
            %a.isLatent == b.isLatent && ...
            %a.isObserved == b.isObserved && ...
            %a.isInput == b.isInput && ...
            %a.isTemp == b.isTemp

            if a.isLatent ~= b.isLatent
                r=false;
                return
            end


            % from PLTFModel.eq:
            % mark matched b factors
            % if there are any unmarked -> inequal
            % problematic case: 
            % a.factors ( ip, jpi ) , b.factors (  ip, pi )
            % b==a matches all b objects with a.factors(1)
            % but a~=b !

            b_marks = zeros(size(b.dims));

            if length(a.dims) == length(b.dims)% && ...
                for d_a = 1:length(a.dims)
                    found = 0;
                    for d_b = 1:length(b.dims)
                        if a.dims(d_a) == b.dims(d_b) && ...
                                b_marks(d_b) == 0
                            found = 1;
                            b_marks(d_b) = 1;
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

        function r = eq_TFDimension(a,b)
        % written for not callins isa() for performance
            r=false;

            %if a.name == b.name && ...
            %a.isLatent == b.isLatent && ...
            %a.isObserved == b.isObserved && ...
            %a.isInput == b.isInput && ...
            %a.isTemp == b.isTemp

            if a.isLatent ~= b.isLatent
                r=false;
                return
            end


            % from PLTFModel.eq:
            % mark matched b factors
            % if there are any unmarked -> inequal
            % problematic case: 
            % a.factors ( ip, jpi ) , b.factors (  ip, pi )
            % b==a matches all b objects with a.factors(1)
            % but a~=b !

            b_marks = zeros(size(b.dims));

            if length(a.dims) == length(b.dims)% && ...
                for d_a = 1:length(a.dims)
                    found = 0;
                    for d_b = 1:length(b.dims)
                        if a.dims(d_a).eq_TFDimension(b.dims(d_b)) && ...
                                b_marks(d_b) == 0
                            found = 1;
                            b_marks(d_b) = 1;
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
            


        function r = ne(a,b)
            r = ~(a==b);
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
        % used with model elements' data
            name = [obj.name '_data'];
        end


        function [] = rand_init(obj, all_dims, imax, data_name)

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

            if nargin < 3
                imax = 100;
            end

            if nargin < 4
                data_name = obj.get_data_name();
            end

            eval( [ 'global ' data_name ';' ] );
            if nargin == 2
                eval( [ data_name ' = rand(' sz ');'] );
            else
                eval( [ data_name  ...
                        ' = randi(' num2str(imax) ', ' sz ');' ] );
            end
        end


        function [] = zero_init(obj, all_dims, data_name)

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

            if nargin < 3
                data_name = obj.get_data_name();
            end

            eval( [ 'global ' data_name ';' ] );
            eval( [ data_name ' = zeros(' sz ');'] );
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


        function [] = remote_gpu_upload(obj, host_url, username, dataset)
            display(['uploading factor ' obj.name]);
            url = [host_url '/upload_data'];

            dims = '';
            for di = 1:length(obj.dims)
                dims = [ dims char(10) obj.dims(di).name ];
            end

            % if data_mat_file is specified use that
            %tfn = false;
            if length(obj.data_mat_file) ~= 0
                data_file_name = obj.data_mat_file
                %f = fopen(obj.data_mat_file);
                %data = fread(f,Inf,'*uint8');
                %fclose(f);
            else
                % if global storage is defined use that
                eval([ 'global ' obj.get_data_name() ';']);
                if eval([ 'length(' obj.get_data_name() ')' ]) ~= 0
                    data_file_name = [tempname '.mat'];
                    save(data_file_name, obj.get_data_name());

                    % there may not be enough space for both mat file and the variable itself
                    %eval([ 'clear ' obj.get_data_name() ';' ]);
                    %f = fopen(tfn);
                    %data = fread(f,Inf,'*uint8');
                    %fclose(f);

                else
                    throw(MException( 'TFFactor:DataError', ...
                                      ['ERROR: no data specified for this factor: ' obj.name] ))
                end
            end

            
            urlread(url, 'Post', ...
                    { 'user', username, 'dataset', dataset, ...
                      'type', 'factor', ...
                      'name', obj.name, ...
                      'dims', dims, ...
                      'isLatent', num2str(obj.isLatent), ...
                      'isObserved', num2str(obj.isObserved), ...
                      'isInput', num2str(obj.isInput), ...
                      'isTemp', num2str(obj.isTemp), ...
                      'isReUsed', num2str(obj.isReUsed), ...
                    });

            f = ftp('localhost:2121', username, '12345');
            mput(f, data_file_name);
            close(f);

            fname = strread(data_file_name, '%s', 'delimiter', '/');
            url = [host_url '/fix_ftp_upload?' ...
                   'user=' username ...
                   '&dataset=' dataset ...
                   '&factor=' obj.name ...
                   '&uploadname=' fname{end}
                  ];
            res = urlread(url);

            display([ ' ' res]);
            % reload data element if it was cleared
            %if ~tfn
            %    load(tfn);
            %end
        end


        function [] = load_data_from_file(obj, all_dims)
        % loads data of factor from file

            factor_data_name = obj.get_data_name();

            var_list = who('-file', obj.data_mat_file);
            display([ 'init factor ' obj.name ': found ' num2str(length(var_list))  ' variable(s) in file ' obj.data_mat_file ]);

            if length(var_list) == 1
                eval([ 'global ' factor_data_name ]);
                cmd = [ 'load( ''' obj.data_mat_file ''' , ''' var_list{1} '''  )' ];
                eval(cmd);
                eval([ factor_data_name ' = ' var_list{1} ';' ]);
                data_target_name = var_list{1};
            else
                found = false;
                for vli = 1:length(var_list)
                    if strcmp(var_list{vli}, factor_data_name)
                        found = true;
                        eval([ factor_data_name ' = ' var_list{vli} ';' ]);
                        data_target_name = var_list{vli};
                        break;
                    end
                end
                if ~found
                    err = MException( ...
                        ['TFFactor:NoSuitableVariableFound'], ...
                        ['please either provide a mat file with a single variable or make sure the mat file contains a variable named ' factor_data_name]);
                    throw(err);
                end
            end

            display([ ' -> using ' data_target_name ' to initialize ' factor_data_name ]);
            for adi = 1:length(all_dims)
                if size(eval(factor_data_name), adi) ~= 1 && size(eval(factor_data_name), adi) ~= all_dims(adi).cardinality
                    err = MException( ...
                        ['TFFactor:WrongDataSize'], ...
                        ['wrong data size on dimension ' char(all_dims(adi).name) ' for variable ' factor_data_name '. Expecting ' num2str(all_dims(adi).cardinality) ' found ' num2str(size(eval(factor_data_name), adi)) char(10)...
                        'size(' data_target_name ') = ' num2str( size(eval(data_target_name)) ) ]);
                    throw(err);
                end
            end
        end

    end

end