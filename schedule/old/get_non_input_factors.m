% returns latent factors which are not given as input

function [non_latent_factors] = get_non_input_factors(tf_model, model)

if nargin==1
    model=1
end

non_latent_factors = {};

for all_alpha = 1:length(tf_model{1}{model}{2})
    for input_alpha = 1:length(tf_model{1}{model}{4})
        if ( char(tf_model{1}{model}{2}(all_alpha)) == ...
             char(tf_model{1}{model}{4}(input_alpha)) )
            break
        end
        non_latent_factors{end+1} = char(tf_model{1}{model}{2}(all_alpha));
    end
end
