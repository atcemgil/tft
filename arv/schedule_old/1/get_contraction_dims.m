% returns contraction dimensions given a gctf_model and operation
function contract_dims = get_contraction_dims(gctf_model, ...
                                              operation_output, ...
                                              operation_latent)
contract_dims=[];
output_indices=get_factor_indices(gctf_model, operation_output);

latent_indices=[];
for i = 1:length(operation_latent)
    latent_indices = [ latent_indices ...
                       get_factor_indices(gctf_model, operation_latent(i)) ...
                     ];
end

for i = 1:length(latent_indices)
    if sum( latent_indices(i) == output_indices ) == 0
        contract_dims = [contract_dims latent_indices(i)];
    end
end

contract_dims=unique(contract_dims);