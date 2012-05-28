% given a tf_model returns indices of all factors in a cell
% observed factors are not included

function [latent_factors latent_factor_indices] = get_all_latent_factor_indices(tf_model, model)
 
if nargin==1
    model=1;
end

latent_factors = {};

for all_alpha = 1:length(tf_model{model}{1}{2})
    for observed_alpha = 1:length(tf_model{model}{1}{6})
        if ( char(tf_model{model}{1}{2}(all_alpha)) == ...
             char(tf_model{model}{1}{6}(observed_alpha)) )
            break
        end
        latent_factors{end+1} = char(tf_model{model}{1}{2}(all_alpha));
    end
end

latent_factor_indices = {};
for i = 1:length(latent_factors)
    latent_factor_indices{end+1} = get_factor_indices(tf_model, ...
                                                      latent_factors{i});
end
