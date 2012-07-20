
% calculate memory cost of generalized multiplication (gm) for the operation
% which removes given index from the input model elements
function [mem, newmodel] = calc_elimination_mem_cost(  gctf_model,  elimination_index_char)


models=gctf_model{1};
newmodel={};
newmodel{1} = [];
newmodel{2} = [];
newmodel{3} = [];
newmodel{4} = [];

for m = 1:length(models)

    factor_chars={};
    for i = 1:length(models{m}.latent_factors)
        factor_chars=[ factor_chars models{m}.latent_factors(i) ]
    end

    % assume single temp
    if isfield(models{m}, 'temps') == 1
        %for i = 1:length(models{m}.temps)
        factor_chars=[ factor_chars models{m}.temps ]
        %end
    end

    mem=0;
    elsize=8; % assume double precision memory

    % scan all input factors
    removed_factor_chars=[]; % for display purposes
    not_removed_factor_inds=[];
    removed_factor_index_chars=[];
    temp_name=[];
    removed_factor_num=0;
    for i = 1:length(factor_chars)
        factor_index_chars = get_factor_indices(gctf_model, char(factor_chars{i}));

        % if factor uses elimination_index mark this factor
        if sum(factor_index_chars == elimination_index_char) == 1
            removed_factor_num = removed_factor_num + 1

            % must perform a gm_mops including all elimiation factors
            mem = mem + elsize * factor_numel(gctf_model, char(factor_chars{i}));

            removed_factor_index_chars = [ removed_factor_index_chars ...
                                           factor_index_chars ];
            temp_name = [ temp_name char(factor_chars{i}) ];
            removed_factor_chars = [ char(removed_factor_chars) ' ' char(factor_chars{i}) ]
        else
            not_removed_factor_inds = [ not_removed_factor_inds i ]
        end
    end







    % create new model with elimination factors removed 
    % and new temporary added

    % must contain indices included in any not_removed_factor_inds
    nm_factor_chars=[];
    nm_index_chars = newmodel{2};
    for i = 1:length(not_removed_factor_inds)
        nm_factor_chars = [ nm_factor_chars factor_chars(not_removed_factor_inds(i)) ];
        nm_index_chars = [ nm_index_chars ...
                           get_factor_indices(gctf_model, ...
                                              factor_chars{not_removed_factor_inds(i)}) ] ;
    end
    nm_index_chars=unique(nm_index_chars);


    display(['bune' num2str(removed_factor_num)])
    % single factor contractions are not considered
    if removed_factor_num < 2
        mem = 0;
        display( ['single element operation for elimination of : ' ...
                  elimination_index_char] )

        % generate new model

        % remove fully contracted factors
        tmpmodel=models{m};
        tmplf={};
        for i = 1:length(tmpmodel.latent_factors) 
            rfc_elim = setdiff(removed_factor_index_chars, ...
                               elimination_index_char);
            if length(rfc_elim) ~= 0
                tmplf=[ tmplf tmpmodel.latent_factors(i) ];
            end
        end
        tmpmodel.latent_factors = tmplf;

        if length( newmodel{1} ) == 0
            newmodel{1}= tmpmodel;
        else
            newmodel{1}={ newmodel{1}, tmpmodel; }
        end

        [ newmodel{2} newmodel{3} ]=order_index_chars(gctf_model, nm_index_chars);


        for i = 1:length(not_removed_factor_inds)
            if not_removed_factor_inds(i) > length(models{m}.latent_factors())
                factor_char = char(models{m}.temps);
            else
                factor_char = char(models{m}.latent_factors(not_removed_factor_inds(i)));
            end


            % if factor is fully contracted do not add
            if strcmp(removed_factor_index_chars, ...
                      get_factor_indices(gctf_model, factor_char)) ...
                    ~= 1
                if length( newmodel{4} ) == 0
                    newmodel{4} = {...
                        factor_char,...
                        get_factor_indices(gctf_model, ...
                                           factor_char)};
                else
                    newmodel{4} = [newmodel{4}, ...
                                   factor_char,...
                                   get_factor_indices(gctf_model, ...
                                                      factor_char)];
                end
            end
        end

        rfc_elim = setdiff(removed_factor_index_chars, ...
                           elimination_index_char);
        if length(rfc_elim) ~= 0 
            newmodel{4} = [ newmodel{4}, ...
                            removed_factor_chars, ...
                            rfc_elim];
        end
    
    else

        removed_factor_index_chars=unique(removed_factor_index_chars);
        removed_factor_index_chars_noelim=[];
        temp_elnum=1;
        temp_ind_chars=[];
        for i = 1:length(removed_factor_index_chars)
            if removed_factor_index_chars(i) ~= elimination_index_char
                removed_factor_index_chars_noelim = [removed_factor_index_chars_noelim ...
                                    removed_factor_index_chars(i)];
                temp_elnum = temp_elnum * get_index_card(gctf_model, ...
                                                         removed_factor_index_chars(i));
                temp_ind_chars=[temp_ind_chars removed_factor_index_chars(i)];
            end
        end

        nm_index_chars=unique([ nm_index_chars removed_factor_index_chars_noelim ]);

        % add current output to mem
        mem = mem + elsize * temp_elnum;

        display(['  operation: eliminate ' elimination_index_char ': ' ...
                 removed_factor_chars ' -> ' temp_name ' mem ' num2str(mem)]);

        %temps={};
        %if isfield(models{m}, 'temps') == 0
        %    temps = { temp_name }
        %else
        %    temps = { models{m}.temps{:} , temp_name }
        %end

        % generate new model
        if length( newmodel{1} ) == 0
            newmodel{1}=struct('latent_factors', { nm_factor_chars }, ...
                               'input_factors', models{m}.input_factors, ...
                               'observed_factor', models{m}.observed_factor, ...
                               'temps', {temp_name} );
        else
            newmodel{1}={ newmodel{1} ,
                          struct('latent_factors', { nm_factor_chars }, ...
                                 'input_factors', models{m}.input_factors, ...
                                 'observed_factor', models{m}.observed_factor, ...
                                 'temps', {temp_name} ) };
        end

        [ newmodel{2} newmodel{3} ]=order_index_chars(gctf_model, nm_index_chars);


        for i = 1:length(not_removed_factor_inds)

            if not_removed_factor_inds(i) > length(models{m}.latent_factors())
                factor_char = char(models{m}.temps);
            else
                factor_char = char(models{m}.latent_factors(not_removed_factor_inds(i)));
            end

            %factor_char = ...
            %    char(models{m}.latent_factors(not_removed_factor_inds(i)));

            if length( newmodel{4} ) == 0
                newmodel{4} = {...
                    factor_char,...
                    get_factor_indices(gctf_model, ...
                                       factor_char)};
            else
                newmodel{4} = [newmodel{4}, ...
                               factor_char,...
                               get_factor_indices(gctf_model, ...
                                                  factor_char)];
            end
        end

        newmodel{4} = [ newmodel{4} , ...
                        temp_name , ...
                        temp_ind_chars];
    end
end

display('newmodel')
newmodel

newmodel{1}{1}
newmodel{1}{2}
newmodel{2}
newmodel{3}
newmodel{4}