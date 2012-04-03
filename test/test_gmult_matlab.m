function [] = test_gmult_matlab()

rng(1);
format 'compact'

[ts_mops r]= doit(0)
[ts_F r]= doit(1)

plot(r, ts_mops, '-r', r, ts_F, '-b')
legend('mops', 'with F')
xlabel('J size')
ylabel('seconds')

end


function [ts r] = doit(OP)
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

% tic; C = gmult_par(A, [I J 0], ...
%                    B, [0 J K], ...
%                    [I 0 K], ...
%                    1); toc;

%OP=1
iter_num=5
r=3:10000:100000

episode_num = 5 % dummy
ts=zeros(1,length(r))
for i=1:length(r)
    J=r(i)

    A = randi(5, [I, J]);
    B = randi(5, [J, K]);

    display(i)
    tic; [C t] = test_gmult(A, [I J 0], ...
                            B, [0 J K], ...
                            [I 0 K], ...
                            1, ...
                            OP, episode_num, iter_num); toc;

    diff=sum(sum(A*B ~= C))
    ts(i) = t;
end

end