function [ KL ] = get_KL_div( p_mat, q_mat )

    KL = -sum(sum( p_mat .* (log( q_mat ) - log(p_mat))  ));

end

