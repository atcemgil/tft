
% calculate memory cost of generalized multiplication (gm) for the operation
% which removes given index from the input model elements
function [mem, newmodel] = calc_elimination_mem_cost(  model,  elimination_index_char)
factor_chars=model{1};
factor_count = length(factor_chars);

mem=0;
elsize=8; % assume double precision memory

% scan all input factors
removed_factor_chars=[]; % for display purposes
not_removed_factor_inds=[];
removed_factor_index_chars=[];
temp_name=[];
for i = 1:factor_count

    % if factor uses elimination_index mark this factor
    if sum(model{4+i} == elimination_index_char) == 1
        
        % must perform a gm_mops including all elimiation factors
        mem = mem + elsize * factor_numel(model,i);

        removed_factor_index_chars = [ removed_factor_index_chars ...
                                       model{4+i} ];
        temp_name = [ temp_name char(factor_chars(i)) ];
        removed_factor_chars = [ char(removed_factor_chars) ' ' char(factor_chars(i)) ];
    else
        not_removed_factor_inds = [ not_removed_factor_inds i ];
    end
end


% create new model with elimination factors removed 
% and new temporary added

% must contain indices included in any not_removed_factor_inds
nm_factor_chars=[];
nm_index_chars=[];
for i = 1:length(not_removed_factor_inds)
    nm_factor_chars = [ nm_factor_chars factor_chars(not_removed_factor_inds(i)) ];
    nm_index_chars = [ nm_index_chars model{4+not_removed_factor_inds(i)} ] ;
end
nm_index_chars=unique(nm_index_chars);


removed_factor_index_chars=unique(removed_factor_index_chars);
removed_factor_index_chars_noelim=[];
temp_elnum=1;
for i = 1:length(removed_factor_index_chars)
    if removed_factor_index_chars(i) ~= elimination_index_char
        removed_factor_index_chars_noelim = [removed_factor_index_chars_noelim ...
                                             removed_factor_index_chars(i)];
        temp_elnum = temp_elnum * get_index_card(model,removed_factor_index_chars(i));
    end
end

nm_index_chars=unique([ nm_index_chars removed_factor_index_chars_noelim ]);

% add current output to mem
mem = mem + elsize * temp_elnum;

display(['  operation: eliminate ' elimination_index_char ': ' removed_factor_chars ' -> ' temp_name ' mem ' num2str(mem)]);

% generate new model
newmodel=cell(1, 4+length(not_removed_factor_inds)+1 );
class(nm_index_chars);
class(temp_name);
newmodel{1}=[ nm_factor_chars temp_name ];
[ newmodel{2} newmodel{3} ]=order_index_chars(model, nm_index_chars);
newmodel{4}=model{4};
for i = 1:length(not_removed_factor_inds)
    newmodel{4+i}=model{4+not_removed_factor_inds(i)};
end
newmodel{4+length(not_removed_factor_inds)+1}= order_index_chars(model,removed_factor_index_chars_noelim);

