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




AB=zeros(i,j,z,f);
for ind_i=1:i
    for ind_j=1:j
        for ind_z=1:z
            for ind_f=1:f
                AB(ind_i,ind_j,ind_z,ind_f) = A(ind_i,ind_z) * B(ind_j,ind_f);
            end
        end
    end
end

AB_ij=sum(AB, 4);
AB_ij=sum(AB_ij, 3);

AB_ijC=zeros(i,j,k,l);
for ind_i=1:i
    for ind_j=1:j
        for ind_k=1:k
            for ind_l=1:l
                AB_ijC(ind_i,ind_j,ind_k, ind_l) = AB_ij(ind_i, ind_j) * C(ind_k, ind_l);
            end
        end
    end
end

ABC_ijk=sum(AB_ijC, 4);

ABCD1=zeros(i,j,k,z,f,l);
for ind_i=1:i
    for ind_j=1:j
        for ind_k=1:k
            for ind_z=1:z
                for ind_f=1:f
                    for ind_l=1:l
                        ABCD1(ind_i,ind_j,ind_k, ind_z, ind_f, ind_l) = ABC_ijk(ind_i,ind_j,ind_k) * D(ind_z, ind_f, ind_l);
                    end
                end
            end
        end
    end
end

X1 = sum(ABCD1, 6);
X1 = sum(X1, 5);
X1 = sum(X1, 4);
