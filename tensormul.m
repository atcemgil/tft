
% VERBOSITY flags for output vebosity
% 0 : none
% 1 : some
% 2 : some more

% prev_full_result is used for displayin purposes, may be left out

function [C, full_result] = tensormul(A,A_card,B,B_card,C_card, VERBOSITY, prev_full_result)
    % assume all tensors have same number of dimensions (may be zero)
    dims=numel(A_card);

    max_cardinalities = max(A_card,B_card);
    % if A,B do not have data on a dimension, then C can not have it either
    %max_cardinalities = max(max_cardinalities,C_card);
    
    % matlab does not like zero indices
    max_cardinalities( max_cardinalities == 0 ) = 1;
    total_cardinality = prod(max_cardinalities);

    full_result = zeros(max_cardinalities);

	% zero indexed
    global_index=zeros(1,dims);
    for element = 1:total_cardinality
        if VERBOSITY > 2
            display(['global_index ' num2str(global_index)]);
        end
        full_result = set_element( full_result, max_cardinalities, global_index, ...
                                   get_element(A, A_card, global_index,VERBOSITY) * ...
                                   get_element(B, B_card, global_index,VERBOSITY) , VERBOSITY );
        
        if element ~= total_cardinality
            global_index=increment_index(max_cardinalities, global_index);
        end
        %display(['done' num2str(element)])
    end

	ndims=length(max_cardinalities);
    
    if VERBOSITY>0
        if nargin > 6 && ...
            isAllDimsSameCard(prev_full_result, full_result) == 1
            % display if prev_full_result is different than full_result
            csum=full_result ~= prev_full_result;
            for i=1:ndims
                csum = sum(csum);
            end
            if csum ~= 0
                display(full_result);
            else
                display('full_result is same as above');
            end
        else
            display(full_result);
        end
    end


	C=full_result;
    % primitive summation along zero result tensor dimensions
    for dim=1:ndims
        if C_card(dim)==0
            %display(['summing over dimension ' num2str(dim)]);
            C = sum(C,dim);
        end
    end
    

end

% index starts from zero
function next_index = increment_index(max_cardinalities, cur_index)
    next_index = cur_index;
    ndims=length(max_cardinalities);
    for dims=1:ndims
        % if we have NOT reached limit of this dimension
        if next_index(dims) ~= max_cardinalities(dims)-1
            % increment this dimension
            next_index(dims) = next_index(dims)+1;
            return
        else
            % we have reached limit of this dimension

            % if next dimension is at limit as well, skip this dimension, operation will take place in next dimension
            if dims ~= ndims && next_index(dims+1) == max_cardinalities(dims+1)-1
                %std::cout << "skip" << std::endl;
                continue
            else
                % if this is the last dimension (and it is full) no increment is possible increment error
                %if (dim == h_ctc->ndims-1){
                %  h_ct->increment_error = 1;
                %  break;
                %}

                % make this and all previous dimensions zero
                for dim_prev=dims:-1:1
                    next_index(dim_prev) = 0;
                end
                % increment next dimension
                next_index(dims+1)=next_index(dims+1)+1;
                break;
            end
        end
    end
end

function el = get_element(tensor_data, cardinalities, index, VERBOSITY)
    strides=get_strides(cardinalities,VERBOSITY);
    cur_ind = sum(strides .* index);
    el = tensor_data(cur_ind+1);
    if VERBOSITY > 1
        display(['get: index ' num2str(cur_ind+1)  ' val ' num2str(el) ]);
    end
end

function tensor_data = set_element(tensor_data, cardinalities, index, val, VERBOSITY)
    strides=get_strides(cardinalities,VERBOSITY);
    cur_ind = sum(strides .* index);
    tensor_data(cur_ind+1) = val;
    if VERBOSITY > 1
        display(['set: index ' num2str(cur_ind+1)  ' val ' num2str(val) ]);
        display(' ');
    end
end

function strides = get_strides(cardinalities,VERBOSITY)
    ndims=length(cardinalities);
    cum_sum=1;
    strides=zeros(size(cardinalities));
    for dims=1:ndims
        if cardinalities(dims) == 0
            strides(dims)=0;
        else
            strides(dims)=cum_sum;
            cum_sum = cum_sum * cardinalities(dims);
        end
    end
    if VERBOSITY > 2
        display(['strides ' num2str(strides)])
    end
end


function cond = isAllDimsSameCard(card_0, card_1)
    cond=1;

    if numel(card_0) ~= numel(card_1) ...
       || length(size(card_0)) ~= length(size(card_1))
        cond = 0;
    else
        % compare each dimension
        for dim = 1:ndims(card_0)
            if size(card_0,dim) ~= size(card_1,dim)
                %display(['!!!!' num2str(size(card_0,dim)) , num2str(size(card_1,dim))])
                cond=0;
                return;
            end
        end
    end
end