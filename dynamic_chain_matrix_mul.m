A=rand(50,20);
B=rand(20,1);
C=rand(1,10);
D=rand(10,100);

m=[size(A,1) size(B,1) size(C,1) size(D,1) size(D,2)];

n=4;

cost=zeros(n,n);
% left and right selections for each cell
seperator=zeros(n, n);

for s=1:n-1
    for i=1:n-s
        j=i+s;

        vals=zeros(1, i-j);
        for k=i:(j-1)
            vals(k-i+1) = cost(i, k) + ...
                          cost(k+1, j) + ...
                          m(i) * m(k+1) * m(j+1);
        end
        cost(i, j) = min(vals);
        seperator(i, j) = find(vals==min(vals));
    end
end
