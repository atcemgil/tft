% returns a cell listing all of the required generalized
% tensor multiplication operations for an iteration of gctf

% output format
% { 'output_tensor', 
%   ['latent/input tensor 1' 'latent/input tensor 2'...],
%   ...  }

function [operations] = gen_tf_rules(tf_model)

operations = [];

models = tf_model{1};

non_input_factors = get_non_input_factors(tf_model);

operation = [];

for alpha = 1:length(non_input_factors)

    for v = 1:length(models)
        operation = {models{v}{6}, ...
                     models{v}{2}};
        operations = [ operations operation ];
    end


    for v = 1:length(models)
        % d1
        operation = { non_input_factors(alpha), ...
                      [models{v}{6} ...
                       setdiff(models{v}{2}, non_input_factors(alpha)) ...
                      ] };
        operations = [ operations operation ]

        % d2
        operation = { non_input_factors(alpha), ...
                      setdiff(models{v}{2}, non_input_factors(alpha)) ...
                    };
        operations = [ operations operation ]
        
    end

end