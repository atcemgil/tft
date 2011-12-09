I=100 ;
J=200 ;
K=500 ;

A = rand(I, J);
B = rand(J, K);

tic; C = gmult_seq(A, [I J 0], ...
                   B, [0 J K], ...
                   [I 0 K], ...
                   1); toc;

diff=sum(sum(A*B ~= C))

tic; C = gmult_par(A, [I J 0], ...
                   B, [0 J K], ...
                   [I 0 K], ...
                   1); toc;

diff=sum(sum(A*B ~= C))