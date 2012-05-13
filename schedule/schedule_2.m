

% gctf model: cell containing following elements
% 1 : gctf_model : array of model structures with following elements
%                  latent_factors: example : { 'ABC' }
%                  input_factors: (not updated, may be empty) example : { 'C' }
%                  observed_factor: example : { 'X' }
%                  temps: example : { }
% 2 : index chracters array : example : 'xyzfl'
% 3 : index cardinalities : example : [ 1 2 3 4 5 ]
% 4 : factor indices cell : pairs of factor and factor's indices
%                           example : { 'A', 'klm', 'B', 'mno', ...
%                                       factor_M, factor_M_indices }

function [] = schedule_2(gctf_model)


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
    opnum = 1;
    for el = 1:length(all_contract_dims)
        display( [ 10, 10 , 10, 10, 'decide on elimination operation ' num2str(opnum) ...
                   ' with elimination indices ' all_contract_dims{el} ] )
        opnum = opnum + 1;
        elim_choices=perms(all_contract_dims{el})
        mems={};
        for i = 1:size(elim_choices, 1)
            display( [10,10,'ELIM CHOICE ', elim_choices(i,:)] )
            gctf_model = orig_model
            mem = 0;
            for j = 1:size(elim_choices, 2)
                display([10, 'CALL calc_elimination_mem_cost with ' elim_choices(i,j)])
                [m gctf_model] = calc_elimination_mem_cost(gctf_model, elim_choices(i,j));
                mem = mem + m;
            end
            mems = [mems; {elim_choices(i,:) mem}]
        end
        display( [ 'mem required ' num2str(mem), 10, 10, 10 ] )
    end
