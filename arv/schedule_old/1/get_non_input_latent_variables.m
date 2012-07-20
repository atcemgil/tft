% returns all latent factors of a gctf model, which should be
% updated, ie. non-input variables

% if v is given only v th model's non input latent variables are
% returned
% otherwise all models' non input latent variables are returned
function [latent_factors] = ...
    get_non_input_latent_variables(gctf_model, v)

latent_factors = [];
models = gctf_model{1};

if nargin == 1
    for v = 1:length(models)
        latent_factors = [ latent_factors ...
                           setdiff(models{v}.latent_factors, ...
                                   models{v}.input_factors) ];
    end
elseif nargin == 2
        latent_factors = setdiff(models{v}.latent_factors, ...
                                 models{v}.input_factors);
end

latent_factors = unique(latent_factors);
