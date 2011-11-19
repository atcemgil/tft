
function [X1] = get_parafac(A,B,C,I,J,K,a,X_dims)
%A1=reshape(A,I,a);
%B1=reshape(B,J,a);
%C1=reshape(C,K,a);

  X1=zeros(X_dims);
  for i=1:I
    for j=1:J
      for k=1:K
        X1(i,j,k) = X1(i,j,k) + A(i,1,1,a)*B(1,j,1,a)*C(1,1,k,a);
      end
    end
  end
end
