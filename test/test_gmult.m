rng(1);

I=2 ;
J=3 ;
K=4 ;

A = randi(5, [I, J]);
B = randi(5, [J, K]);

% dikkat ikisi (_seq, _par) birden calismiyor ???

%tic; C = gmult_seq(A, [I J 0], ...
%                   B, [0 J K], ...
%                   [I 0 K], ...
%                   1); toc;

%diff=sum(sum(A*B ~= C))

tic; C = gmult_par(A, [I J 0], ...
                   B, [0 J K], ...
                   [I 0 K], ...
                   1); toc;

diff=sum(sum(A*B ~= C))
