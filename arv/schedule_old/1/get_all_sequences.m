% given a list of contraction indices produces all possible
% contraction sequences

function all_seqs = get_all_sequences(contract_dims, cur_seq, all_seqs)

if nargin==1
    cur = 1;
    cur_seq = [];
    all_seqs = {};
end


if length(contract_dims) > 0

    for i = 1:length(contract_dims{1})
        all_seqs = get_all_sequences( contract_dims(2:end), ...
                                      [cur_seq contract_dims{1}(i)], ...
                                      all_seqs );
    end

else

    %all_seqs = [ all_seqs unique(cur_seq) ];
    if sum(not(cellfun('isempty', (strfind(all_seqs, unique(cur_seq)))))) == 0
        all_seqs = [ all_seqs unique(cur_seq) ]
    end

end