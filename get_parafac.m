
function [X1] = get_parafac(A_data,B_data,C_data,I,J,K,A,X_dims)

  X1=zeros(X_dims);
  for i=1:I
    for j=1:J
      for k=1:K
        for a=1:A
            X1(i,j,k) = X1(i,j,k) + ...
                        A_data(i,1,1,a)*B_data(1,j,1,a)*C_data(1,1,k,a);
        end
      end
    end
  end
end
