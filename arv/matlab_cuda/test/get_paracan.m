
function [X1] = get_paracan(A_data,B_data,I,J,A,X_dims)

  X1=zeros(X_dims);
  for i=1:I
    for j=1:J
        for a=1:A
            X1(i,j) = X1(i,j) + ...
                      A_data(i,1,1,a)*B_data(1,j,1,a);
        end
    end
  end
end
