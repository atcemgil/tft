% returns a cell listing all of the required generalized
% tensor multiplication operations for an iteration of gctf

% output format
% { 'output_tensor', 
%   ['latent/input tensor 1' 'latent/input tensor 2'...],
%   ...  }

function [operations] = gen_gctf_rules(gctf_model)

operations = [];

models = gctf_model{1};

non_input_latent_variables = get_non_input_latent_variables(gctf_model);

operation = [];

for alpha = 1:length(non_input_latent_variables)

    for v = 1:length(models)
        operation = {models{v}.observed_factor, ...
                     models{v}.latent_factors};
        operations = [ operations operation ];
    end


    for v = 1:length(models)
        % d1
        operation = { non_input_latent_variables(alpha), ...
                      [models{v}.observed_factor ...
                       models{v}.latent_factors] };
        operations = [ operations operation ];

        % d2
        operation = { non_input_latent_variables(alpha), ...
                      models{v}.latent_factors };
        operations = [ operations operation ];
        
    end

end