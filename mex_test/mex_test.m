% mex -setup 
% select 1 for gcc

% compile c file with mex
mex helloworld.c
% run mexfunction in helloworld.c
helloworld

mex twice.c
[a,b]=twice([1 2 3 4; 5 6 7 8], 8)