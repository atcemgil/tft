function [Z1, Z2, kl_data] = m_nmf(iter, Z1, Z2, M, X, diff_limit)

    kl_data = zeros(iter,1);
    
    for i=1:iter
        % find new Z1 estimate
        
        X_hat = Z1*Z2;
        
        A  = X./X_hat;
        A  = M.*A;
        D1 = A * Z2';
        D2 = M * Z2';
        D1 = D1 ./ D2;
        Z1 = Z1 .* D1;

        X_hat = Z1*Z2;
        
        
        A = X./X_hat;
        A = M.*A;
        D1 = A' * Z1;
        D2 = M' * Z1;
        D1 = D1 ./ D2;
        Z2 = Z2 .* D1';

        kl_data(i)=get_KL_div(X,Z1*Z2);
        
        if get_mean_diff(X,Z1*Z2) < diff_limit
            return
        end
    end
    

end