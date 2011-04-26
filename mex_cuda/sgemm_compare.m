A=[1.5 2 3;4 4 4]; 
B=[44 55 66;11 11 11]';
% Let's use some large arrays... 
A=randn(4000,2000); 
B=randn(2000,4000);
[m n]=size(A);
[mm nn]=size(B);
C=randn(m,nn);
alpha=-1;
beta=0;
disp('Matlab:') 
tic 
C1d=alpha*A*B + beta*C; 
toc

% In single precision, Matlab is twice as fast! (go figure...) 
tic 
A1=single(A); 
B1=single(B); 
C1=single(C);
C1s=alpha*A1*B1 + beta*C1;
toc

% The call here is testing out the transposes of the code. 
disp('CUDA:') 
tic 
C2=sgemm_cu(0,1,single(alpha),single(beta),single(A),single(B'),single(C)); 
toc

% Compare the CUDA results with the Matlab results

min(min(C2-C1s))/min(min(C1s)) 
max(max(C2-C1s))/max(max(C1s))