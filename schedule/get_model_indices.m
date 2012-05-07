% returns list of indices used by the selected model
% index_type : only latent -> 'latent'
%              only input -> 'input'
%              only observed -> 'observed'
%              all -> 'all'
function [inds] = get_model_indices(gctf_model, n, index_type)

models=gctf_model{1}
factor_indices=gctf_model{4}


if strcmp(index_type, 'all') || strcmp(index_type, 'observed')
    % assume only 1 observed factor
    inds=get_factor_indices(gctf_model, models{n}(1).observed_factor)
else
    inds=[]
end


if strcmp(index_type, 'all') || strcmp(index_type, 'input')
    for i = 1:length(models{n})
        inds = [ inds ...
                 get_factor_indices(gctf_model, models{n}(i).input_factors) ...
               ]
    end
    
    inds=order_index_chars(gctf_model, unique(inds))
end

if strcmp(index_type, 'all') || strcmp(index_type, 'latent')
    for i = 1:length(models{n})
        inds = [ inds ...
                 get_factor_indices(gctf_model, models{n}(i).latent_factors) ...
               ]
    end
    
    inds=order_index_chars(gctf_model, unique(inds))
end