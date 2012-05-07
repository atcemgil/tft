

% gctf model: cell containing following elements
% 1 : gctf_model : array of model structures with following elements
%                  latent_factors: example : { 'ABC' }
%                  input_factors: (not updated, may be empty) example : { 'C' }
%                  observed_factor: example : { 'X' }
% 2 : index chracters array : example : 'xyzfl'
% 3 : index cardinalities : example : [ 1 2 3 4 5 ]
% 4 : factor indices cell : pairs of factor and factor's indices
%                           example : { 'A', 'klm', 'B', 'mno', ...
%                                       factor_M, factor_M_indices }

function [elimination_seqs] = schedule_2(gctf_model)


    models           = gctf_model{1};
    index_chars      = gctf_model{2};
    index_cards      = gctf_model{3};
    observed_factors = gctf_model{4};


    rules=gen_gctf_rules(gctf_model)

    all_contract_dims = {};
    for r = 1:(length(rules)/2)

        cd = get_contraction_dims(gctf_model, ...
                                  rules{(r-1)*2+1}, ...
                                  rules{(r-1)*2+2} );

        all_contract_dims = [ all_contract_dims cd ];
    end

    display(all_contract_dims)


    % all possible eliminations -> exponential
    % elimination_seqs = get_all_sequences(all_contract_dims);
    % calculate memory requirement for each possible elimination
    % sequence -> NOT PRACTICAL


    % greedy approach
    % given a list of contract_dims select least memory requirement

    orig_model = gctf_model;
    for p = 1:length(elimination_seqs)
        display( [ 'test elimination sequence ' elimination_seqs(p,:) ] )
        gctf_model = orig_model;
        mem = 0;
        for e = 1:length(all_contract_dims)
            [m gctf_model] = calc_elimination_mem_cost(gctf_model, elimination_seqs(p,e));
            mem = mem + m;
        end
        display( [ 'mem required ' num2str(mem) ] )
    end
