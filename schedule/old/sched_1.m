function X = sched_1(A,B,C,D,i,j,k,z,f,l)

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

ABC=zeros(i,j,k,z,f,l);
for ind_i=1:i
    for ind_j=1:j
        for ind_k=1:k
            for ind_z=1:z
                for ind_f=1:f
                    for ind_l=1:l
                        ABC(ind_i,ind_j,ind_k, ind_z, ind_f, ind_l) = AB(ind_i, ind_j, ind_z, ind_f) * C(ind_k, ind_l);
                    end
                end
            end
        end
    end
end

ABCD=zeros(i,j,k,z,f,l);
for ind_i=1:i
    for ind_j=1:j
        for ind_k=1:k
            for ind_z=1:z
                for ind_f=1:f
                    for ind_l=1:l
                        ABCD(ind_i,ind_j,ind_k, ind_z, ind_f, ind_l) = ABC(ind_i,ind_j,ind_k, ind_z, ind_f, ind_l) * D(ind_z, ind_f, ind_l);
                    end
                end
            end
        end
    end
end

X = sum(ABCD, 6);
X = sum(X, 5);
X = sum(X, 4);
