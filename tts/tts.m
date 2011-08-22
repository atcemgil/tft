% tensor test set
% generates results of all possible factorizations of two [3 3 3] tensors

% cardinality of 3 indicates tensors have 3 elements in the corresponding
% dimension.

% cardinality of 0 indicates
%    - contraction on corresponding dimension for the result tensor
%    - no data is available on corresponding dimension for the input tensor

% all possible factorization of two [3 3 3] tensors is described as such:
% maximum cardinality:
% A [3 3 3] * B [3 3 3] = C [3 3 3]
% this case reduces to the hadamard product

% We could generate multiplications of smaller cardinalities such as:
% A [3 3 3] * B [3 3 3] = C [3 3 0]
% in this case we have to sum up corresponding elements of the complete
% product vector to calculate the result tensor.

% We could move the 0 cardinality to any index of any tensor object in the
% above mentioned multiplication operation.

% We could also have two 0 cardinality dimensions in this factorization.
% Any tensor with two 0 cardinality dimensions would reduce to a vector in
% this case.

% The following code generates all possible combinations of cardinalities
% for one and two 0 cardinality tensors. Results are stored and published
% as a test set.

% if input opnumber is specified, only numbers in the opnumber vector will
% be processed

function [] = tts(opnumber)
% tts function

    TEST=3;
    % by specifying TEST=1 and a test folder, can perform checks
    %               TEST=2 can perform tests with the cudatensor3 output
    %               TEST=3 can perform tests with the C code output

    TEST_FOLDER='set4/'; % do not forget the last /

    % by specifying OUTPUT=1 and an output folder, can store generated results
    % (overwrites)

    OUTPUT=0;
    OUTPUT_FOLDER='theset/'; % do not forget the last /

    % defines output verbosity
    VERBOSITY=0;












    if TEST~=0
        display(['test mode:' num2str(TEST)]);
        display(['sourcing test set from folder: ' TEST_FOLDER]);
        if TEST == 1
            display('testing folder contents with matlab output')
        elseif TEST == 2
            display('testing folder contents with GPU output')
        elseif TEST == 3
            display('testing folder contents with C code output')
        end
    end

    if OUTPUT==1
        display(['storing output to folder: ' OUTPUT_FOLDER]);
    end


    d=3;
    A_card=[unique(perms([d 0 d]), 'rows'); unique(perms([d 0 0]), 'rows')];
    B_card=A_card;
    C_card=B_card;

    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);

    A_orig=round(rand(d,d,d)*5);
    B_orig=round(rand(d,d,d)*5);

    if VERBOSITY>0
        display(A_orig);
        display(B_orig);
    end

    opnum=1;
    full_result=0;
    errcount=0;
    
    for a_c=1:size(A_card,1)
        for b_c=1:size(B_card,1)
            for c_c=1:size(C_card,1)

                if nargin==1 && sum( opnumber == opnum ) == 0
                    opnum = opnum + 1;
                else

                    if VERBOSITY>0
                        display(['op ' num2str(opnum) ': ' ...
                                 ' A_card ' num2str(A_card(a_c,:)) ...
                                 ' B_card ' num2str(B_card(b_c,:)) ...
                                 ' C_card ' num2str(C_card(c_c,:)) ])
                    end

                    A=get_tensor_part(A_orig,A_card(a_c,:),VERBOSITY);
                    B=get_tensor_part(B_orig,B_card(b_c,:),VERBOSITY);
                    A_crd = A_card(a_c,:);
                    B_crd = B_card(b_c,:);
                    C_crd = C_card(c_c,:);

                    if TEST == 1
                        % perform the operation on cpu with matlab code
                        [op, full_result] = ...
                            tensormul( A, A_crd, B, B_crd, C_crd, VERBOSITY , full_result );
                    elseif TEST == 2
                        % perform operation on gpu
                        addpath('../mex_cudatensor3')
                        tic; op=cudatensor3(A,A_crd,B,B_crd,C_crd,0); toc;
                    elseif TEST == 3
                        % perform operation on cpu with C code
                        addpath('../mex_cudatensor3')
                        tic; op=cudatensor3(A,A_crd,B,B_crd,C_crd,1); toc;
                    end


                    if VERBOSITY>0
                        display(op);
                    end




                    if TEST~=0
                        test=load([TEST_FOLDER 'op' num2str(opnum)]);

                        numeldiff = numel(op) - numel(test.op);
                        allequal=0;
                        if numeldiff == 0
                            display(['numeldiff ok : ' num2str(numeldiff)]);
                            for n=1:numel(op)
                                if op(n) ~= test.op(n)
                                    allequal = allequal+1;
                                end
                            end
                        end

                        if allequal ~= 0 || numeldiff ~= 0
                            display(['op ' num2str(opnum) ': test failed (' num2str(errcount) ')']);
                            display('saved tensor');
                            display(test.op);
                            %return
                            errcount=errcount+1;
                        else
                            display(['op ' num2str(opnum) ': ok']);
                        end
                    end


                    if OUTPUT == 1
                        save([OUTPUT_FOLDER 'op' num2str(opnum)], 'op');
                    end


                    if VERBOSITY==1
                        display(' ');
                        display(' ');
                        display(' ');
                    end

                    opnum = opnum + 1;
                end
            end
        end
    end
    display(['error count' num2str(errcount)]);
end



% helper function to cut parts of complete tensor objects
% example A may have 3 dimensions all full with data
%         but to test [0 3 3] we need to crop the first dimension
function parts = get_tensor_part(tensor, cardinalities, VERBOSITY)
    ndims=length(cardinalities);
    str='tensor(';
    for i=1:ndims
        if cardinalities(i) == 0
            str=strcat(str,'1');
        else
            str=strcat(str,':');
        end

        if i ~= ndims
            str=strcat(str,',');
        end
    end
    str=strcat(str,')');
    parts = eval( str );
    if VERBOSITY > 2
        display([str ': '])
        display(parts);
    end
end
