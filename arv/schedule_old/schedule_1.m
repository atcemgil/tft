
% model: cell containing following keys
% 1: factor characters -> 1xN cell of factor names
% 2: index characters -> 1xM array of index characters
% 3: index cardinalities -> 1xM array of index cardinalities
% 4: 1x? array of output factor index characters
% 5: 1x? array of factor index characters for factor 1
% 6: 1x? array of factor index characters for factor 2
% ...
% N: 1x? array of factor index characters for factor N
function [order elimination_seqs] = schedule_1(model)

    factor_chars=model{1};
    factor_count=length(factor_chars);

    index_chars=model{2};
    index_cards=model{3};

    output_chars=model{4};


    contract_dims='';
    for i = 1:length(index_chars)
        if sum( index_chars(i) == output_chars ) == 0
            contract_dims = [contract_dims index_chars(i)];
        end
    end

    % all possible eliminations
    elimination_seqs=perms(contract_dims)

    % calculate memory requirement for each possible elimination sequence
    orig_model = model;
    for p = 1:length(elimination_seqs)
        display( [ 'test elimination sequence ' elimination_seqs(p,:) ] )
        model = orig_model;
        mem = 0;
        for e = 1:length(contract_dims)
            [m model] = calc_elimination_mem_cost(model, elimination_seqs(p,e));
            mem = mem + m;
        end
        display( [ 'mem required ' num2str(mem) ] )
    end
