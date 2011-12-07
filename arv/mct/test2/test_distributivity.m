format 'compact'
clear

S = RandStream('mt19937ar');
RandStream.setDefaultStream(S);

I=3;
J=4;
K=5;
a=2;

V=['i','j','k','a'];

A_cards=['i','a'];
A_true = round(10*rand(I,a));

B_cards=['j','a'];
B_true = round(20*rand(J,a));

C_cards=['k','a'];
C_true = round(30*rand(K,a));

X_cards = ['i','j','k'];

A1=reshape(A_true,I,a);
B1=reshape(B_true,J,a);
C1=reshape(C_true,K,a);
F_1=zeros(I,J,K,a);
for a_i=1:a
    for i=1:I
        for j=1:J
            for k=1:K
                F_1(i,j,k,a_i) = A1(i,a_i)*B1(j,a_i)*C1(k,a_i);
            end
        end
    end
end

F_2=zeros(I,J,K,a);
for a_i=1:a
    for i=1:I
        for j=1:J
            for k=1:K
                F_2(i,j,k,a_i) = A1(i,a_i)*B1(j,a_i);
            end
        end
    end
end
for a_i=1:a
    for i=1:I
        for j=1:J
            for k=1:K
                F_2(i,j,k,a_i) = F_2(i,j,k,a_i)*C1(k,a_i);
            end
        end
    end
end
