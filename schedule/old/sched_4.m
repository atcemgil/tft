rng(1)

i=2;
j=3;
k=4;
z=5;
f=6;
l=7;

rand_max=10;

A=randi(rand_max,i,z);
B=randi(rand_max,j,f);
C=randi(rand_max,k,l);
D=randi(rand_max,z,f,l);



AD=zeros(z,f,l);
for ind_z=1:z
    for ind_f=1:f
        for ind_l=1:l
            AD(ind_i, ind_z, ind_f, ind_l) = A(ind_i, ind_z) * D(ind_z, ind_f, ind_l);
        end
    end
end

D1=sum(AD,2); % i,f,l
D1=reshape(D1,[i,f,l]);

BD1=zeros(j,f,i,l);
for ind_j=1:j
    for ind_f=1:f
        for ind_i=1:i
            for ind_l=1:l
                BD1(ind_j, ind_f, ind_i, ind_l) = B(ind_j, ind_f) * D1(ind_i, ind_f, ind_l);
            end
        end
    end
end

D2=sum(BD1,2); % j,i,l
D2=reshape(D2,[j,i,l]);

CD2=zeros(i,l,j,k);
for ind_k=1:k
    for ind_l=1:l
        for ind_i=1:i
            for ind_j=1:j
                CD2(ind_i, ind_l, ind_j, ind_k) = B(ind_j, ind_f) * D(ind_i, ind_f, ind_l);
            end
        end
    end
end

D3=sum(CD2,2); % i,j,k
D3=reshape(D3,[i,j,k]);
